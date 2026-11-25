#!/usr/bin/env python3
# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0
"""
Unified External Baseline Comparison Runner (Step 4 & Step 5).

Runs reproducible trajectory and global-search comparisons under isolated ROS 2
localhost domains against external unmodified binaries installed outside this repo:
  - gn10_pointcloud_localization (3D+2D and 3D-only)
  - robotics-upo/dll (Humble Ceres SDF solver via OctoMap .bt)
  - rsasaki0109/lidar_localization_ros2 (NDT_OMP and GICP_OMP)
  - kiss-icp (pure LiDAR odometry baseline anchored to initial reference pose)
  - KOKIAOKI/3d_bbs (CPU and GPU Branch-and-Bound global search)
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "runners"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "analysis"))

from evaluate_trajectories import compute_ate_rpe_metrics, save_tum_file
from ros_eval_common import (
    EvalListener,
    load_reference_from_bag,
    resolve_localization_paths,
    terminate_process_group,
    wrap_angle,
)
from run_ablation_suite import SystemGPUMonitor


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_repo_or_user_path(raw: str) -> Path:
    p = Path(os.path.expanduser(raw))
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _ros_param_cli_args(params: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, val in params.items():
        if isinstance(val, bool):
            val_str = "true" if val else "false"
        else:
            val_str = str(val)
        args.extend(["-p", f"{key}:={val_str}"])
    return args


def _quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _rot_matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return (float(qx), float(qy), float(qz), float(qw))


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class ExternalPoseListener(Node):
    """Collects trajectories, static TFs, and diagnostic timings from external localizers."""

    def __init__(
        self,
        tf_static_msgs: list[Any],
        listen_tf_parent: str | None = None,
        listen_tf_child: str | None = None,
        shift_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        listen_pcl_pose: bool = False,
        listen_diag_status: bool = False,
        stamp_offset_ns: int = 0,
        imu_cv_odom_topic: str | None = None,
    ) -> None:
        super().__init__("baseline_compare_listener")
        self._lock = threading.Lock()
        self.poses: list[tuple[int, float, float, float, float, float, float, float]] = []
        self.latencies_ms: list[float] = []
        self.accepted_frames = 0
        self.rejected_frames = 0
        self._tf_parent = listen_tf_parent
        self._tf_child = listen_tf_child
        self._shift = shift_xyz
        self._stamp_offset_ns = int(stamp_offset_ns)

        # Lightweight symmetric IMU-yaw + EMA-CV odom->base_link bridge for DLL
        self._use_imu_cv_odom = bool(imu_cv_odom_topic)
        self._static_rot_map: dict[str, np.ndarray] = {}
        for msg in tf_static_msgs:
            for tf in msg.transforms:
                child = tf.child_frame_id.lstrip("/")
                r = tf.transform.rotation
                self._static_rot_map[child] = _quat_to_rot_matrix(r.x, r.y, r.z, r.w)
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0
        self._prev_imu_ns: int | None = None
        self._prev_wz = 0.0
        self._odom_hist_ns: list[int] = []
        self._odom_hist_xyy: list[tuple[float, float, float]] = []
        self._last_R_map_odom = np.eye(3, dtype=np.float64)
        self._map_base_hist: list[tuple[int, float, float]] = []
        self._vx_world = 0.0
        self._vy_world = 0.0
        self._has_vel = False
        self._last_tf_pose_key: tuple[float, float, float] | None = None

        qos_static = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub_tf_static = self.create_publisher(TFMessage, "/tf_static", qos_static)
        merged = TFMessage()
        for msg in tf_static_msgs:
            merged.transforms.extend(msg.transforms)
        self._merged_static = merged
        self.create_timer(0.05, self._publish_static_tf)

        if self._use_imu_cv_odom:
            self._pub_odom_tf = self.create_publisher(
                TFMessage, "/tf", QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE)
            )
            self._imu_sub = self.create_subscription(
                Imu,
                imu_cv_odom_topic,
                self._on_imu_for_odom,
                QoSProfile(depth=400, reliability=ReliabilityPolicy.BEST_EFFORT),
            )

        if listen_tf_parent and listen_tf_child:
            self._tf_sub = self.create_subscription(
                TFMessage,
                "/tf",
                self._on_tf,
                QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT),
            )

        if listen_pcl_pose:
            pose_qos = QoSProfile(
                depth=50,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._pose_sub = self.create_subscription(
                PoseWithCovarianceStamped,
                "/pcl_pose",
                self._on_pcl_pose,
                pose_qos,
            )

        if listen_diag_status:
            diag_qos = QoSProfile(
                depth=50,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._diag_sub = self.create_subscription(
                DiagnosticArray,
                "/alignment_status",
                self._on_diag,
                diag_qos,
            )

    def _publish_static_tf(self) -> None:
        if self._merged_static.transforms:
            self._pub_tf_static.publish(self._merged_static)

    def _lookup_odom_at(self, t_ns: int) -> tuple[float, float, float]:
        if not self._odom_hist_ns:
            return (0.0, 0.0, 0.0)
        idx = bisect.bisect_left(self._odom_hist_ns, t_ns)
        if idx == 0:
            return self._odom_hist_xyy[0]
        if idx >= len(self._odom_hist_ns):
            return self._odom_hist_xyy[-1]
        t0, t1 = self._odom_hist_ns[idx - 1], self._odom_hist_ns[idx]
        p0, p1 = self._odom_hist_xyy[idx - 1], self._odom_hist_xyy[idx]
        if t1 <= t0:
            return p1
        a = float(t_ns - t0) / float(t1 - t0)
        ox = p0[0] + a * (p1[0] - p0[0])
        oy = p0[1] + a * (p1[1] - p0[1])
        oyaw = wrap_angle(p0[2] + a * wrap_angle(p1[2] - p0[2]))
        return (ox, oy, oyaw)

    def _on_imu_for_odom(self, msg: Imu) -> None:
        t_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        if t_ns <= 0:
            return
        frame = msg.header.frame_id.lstrip("/")
        w = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=np.float64)
        if frame != "base_link" and frame in self._static_rot_map:
            w = self._static_rot_map[frame] @ w
        wz = float(w[2])

        with self._lock:
            if self._prev_imu_ns is not None:
                dt = (t_ns - self._prev_imu_ns) * 1e-9
                if 0.0 < dt < 0.5:
                    self._odom_yaw = wrap_angle(self._odom_yaw + 0.5 * (self._prev_wz + wz) * dt)
                    if self._has_vel:
                        v_odom = self._last_R_map_odom.T @ np.array([self._vx_world, self._vy_world, 0.0])
                        self._odom_x += float(v_odom[0]) * dt
                        self._odom_y += float(v_odom[1]) * dt
            self._prev_imu_ns = t_ns
            self._prev_wz = wz
            # Predict 50 ms ahead so TF stamped at (t_ns + 0.05 s) + 0.10 s always brackets
            # DLL's lookup at cloud_stamp + 0.10 s with zero time lag and zero extrapolation error.
            lead_s = 0.05
            ox_lead = self._odom_x
            oy_lead = self._odom_y
            if self._has_vel:
                v_odom = self._last_R_map_odom.T @ np.array([self._vx_world, self._vy_world, 0.0])
                ox_lead += float(v_odom[0]) * lead_s
                oy_lead += float(v_odom[1]) * lead_s
            oyaw_lead = wrap_angle(self._odom_yaw + wz * lead_s)
            t_lead_ns = t_ns + int(lead_s * 1e9)
            self._odom_hist_ns.append(t_lead_ns)
            self._odom_hist_xyy.append((ox_lead, oy_lead, oyaw_lead))
            ox, oy, oyaw = ox_lead, oy_lead, oyaw_lead

        pub_ns = t_lead_ns - self._stamp_offset_ns
        tf_msg = TFMessage()
        tr = TransformStamped()
        tr.header.stamp.sec = int(pub_ns // 10**9)
        tr.header.stamp.nanosec = int(pub_ns % 10**9)
        tr.header.frame_id = "odom"
        tr.child_frame_id = "base_link"
        tr.transform.translation.x = ox
        tr.transform.translation.y = oy
        tr.transform.translation.z = 0.0
        qx, qy, qz, qw = _yaw_to_quat(oyaw)
        tr.transform.rotation.x = qx
        tr.transform.rotation.y = qy
        tr.transform.rotation.z = qz
        tr.transform.rotation.w = qw
        tf_msg.transforms.append(tr)
        self._pub_odom_tf.publish(tf_msg)

    def _on_tf(self, msg: TFMessage) -> None:
        sx, sy, sz = self._shift
        with self._lock:
            for tr in msg.transforms:
                if (
                    tr.header.frame_id.lstrip("/") == self._tf_parent
                    and tr.child_frame_id.lstrip("/") == self._tf_child
                ):
                    stamp_ns = int(tr.header.stamp.sec) * 10**9 + int(tr.header.stamp.nanosec) + self._stamp_offset_ns
                    if stamp_ns <= 0:
                        continue
                    t = tr.transform.translation
                    q = tr.transform.rotation
                    if self._use_imu_cv_odom:
                        R_mo = _quat_to_rot_matrix(q.x, q.y, q.z, q.w)
                        t_mo = np.array([t.x + sx, t.y + sy, t.z + sz], dtype=np.float64)
                        self._last_R_map_odom = R_mo
                        ox, oy, oyaw = self._lookup_odom_at(stamp_ns)
                        c_o, s_o = math.cos(oyaw), math.sin(oyaw)
                        R_ob = np.array([[c_o, -s_o, 0.0], [s_o, c_o, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                        t_mb = t_mo + R_mo @ np.array([ox, oy, 0.0], dtype=np.float64)
                        R_mb = R_mo @ R_ob
                        qx, qy, qz, qw = _rot_matrix_to_quat(R_mb)
                        mx, my, mz = float(t_mb[0]), float(t_mb[1]), float(t_mb[2])
                        self.poses.append((stamp_ns, mx, my, mz, qx, qy, qz, qw))

                        # Update EMA planar velocity (same 0.20 s window, alpha=0.6, max 4.0 m/s as gn10)
                        pose_key = (round(t.x, 5), round(t.y, 5), round(q.z, 5))
                        if pose_key != self._last_tf_pose_key:
                            self._last_tf_pose_key = pose_key
                            if self._map_base_hist:
                                prev_t, prev_x, prev_y = self._map_base_hist[-1]
                                sdt = (stamp_ns - prev_t) * 1e-9
                                if sdt <= 1e-3 or math.hypot(mx - prev_x, my - prev_y) / sdt > 4.0:
                                    self._map_base_hist.clear()
                                    self._has_vel = False
                            self._map_base_hist.append((stamp_ns, mx, my))
                            while len(self._map_base_hist) > 2 and (stamp_ns - self._map_base_hist[1][0]) * 1e-9 > 0.20:
                                self._map_base_hist.pop(0)
                            old_t, old_x, old_y = self._map_base_hist[0]
                            dt = (stamp_ns - old_t) * 1e-9
                            if 0.15 <= dt < 0.50:
                                vx = (mx - old_x) / dt
                                vy = (my - old_y) / dt
                                if math.hypot(vx, vy) < 4.0:
                                    if self._has_vel:
                                        self._vx_world = 0.6 * vx + 0.4 * self._vx_world
                                        self._vy_world = 0.6 * vy + 0.4 * self._vy_world
                                    else:
                                        self._vx_world, self._vy_world = vx, vy
                                        self._has_vel = True
                                else:
                                    self._has_vel = False
                    else:
                        self.poses.append((stamp_ns, t.x + sx, t.y + sy, t.z + sz, q.x, q.y, q.z, q.w))

    def _on_pcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        stamp_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        if stamp_ns <= 0:
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        with self._lock:
            self.poses.append((stamp_ns, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    def _on_diag(self, msg: DiagnosticArray) -> None:
        with self._lock:
            for status in msg.status:
                kv = {item.key: item.value for item in status.values}
                if "alignment_time_sec" in kv:
                    try:
                        ms = float(kv["alignment_time_sec"]) * 1000.0
                        if ms > 0.0:
                            self.latencies_ms.append(ms)
                    except ValueError:
                        pass
                action = kv.get("recovery_action", "")
                if "reject" in action:
                    self.rejected_frames += 1
                elif "accept" in action:
                    self.accepted_frames += 1

    def snapshot(self) -> tuple[list[tuple[int, float, float, float, float, float, float, float]], list[float], int, int]:
        with self._lock:
            dedup: dict[int, tuple[int, float, float, float, float, float, float, float]] = {}
            for row in self.poses:
                dedup[row[0]] = row
            sorted_poses = [dedup[k] for k in sorted(dedup.keys())]
            return sorted_poses, list(self.latencies_ms), self.accepted_frames, self.rejected_frames


def _spin_executor_thread(node: Node) -> tuple[MultiThreadedExecutor, threading.Event, threading.Thread]:
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    stop_event = threading.Event()

    def _loop() -> None:
        while not stop_event.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.05)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return executor, stop_event, t


def _summarize_resources(res_df: pd.DataFrame | None, cum_cpu_pct: float | None = None) -> dict[str, Any]:
    if res_df is None or len(res_df) == 0:
        return {}
    out: dict[str, Any] = {}
    for col, out_key, fn in [
        ("process_cpu_pct", "cpu_node_mean_pct", np.nanmean),
        ("process_rss_mb", "ram_rss_max_mb", np.nanmax),
        ("gpu_util_pct", "gpu_util_mean_pct", np.nanmean),
        ("gpu_power_w", "gpu_power_mean_w", np.nanmean),
        ("cpu_gpu_power_w", "cpu_gpu_power_mean_w", np.nanmean),
    ]:
        if col in res_df.columns:
            vals = res_df[col].dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                out[out_key] = float(fn(vals))
    if cum_cpu_pct is not None and np.isfinite(cum_cpu_pct):
        out["cpu_node_mean_pct"] = float(cum_cpu_pct)
    return out


def _summarize_latencies(latencies_ms: list[float]) -> dict[str, Any]:
    if not latencies_ms:
        return {"latency_mean_ms": None, "latency_p95_ms": None, "latency_max_ms": None, "latency_samples": 0}
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "latency_mean_ms": float(np.mean(arr)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_max_ms": float(np.max(arr)),
        "latency_samples": int(len(arr)),
    }


def _build_env(domain_id: int, ext_ws: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["RCUTILS_LOGGING_BUFFERED_STREAM"] = "0"
    octo_lib = str(ext_ws / "install_octomap" / "lib")
    ext_lib_dirs = [
        octo_lib,
        str(ext_ws / "install" / "pcl_ros" / "lib"),
        str(ext_ws / "install" / "ndt_omp_ros2" / "lib"),
        str(ext_ws / "install" / "lidar_localization_ros2" / "lib"),
        str(ext_ws / "install" / "dll" / "lib"),
    ]
    prev_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([d for d in ext_lib_dirs if os.path.isdir(d)] + ([prev_ld] if prev_ld else []))
    return env


def _start_odom_identity_tf(env: dict[str, str]) -> subprocess.Popen:
    cmd = [
        "ros2",
        "run",
        "tf2_ros",
        "static_transform_publisher",
        "--frame-id",
        "odom",
        "--child-frame-id",
        "base_link",
        "--ros-args",
        "-p",
        "use_sim_time:=true",
    ]
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _play_bag(
    bag_path: Path,
    offset_s: float,
    rate: float,
    env: dict[str, str],
    topics: list[str] | None = None,
) -> subprocess.Popen:
    cmd = [
        "ros2",
        "bag",
        "play",
        str(bag_path),
        "--clock",
        "-r",
        str(rate),
        "--start-offset",
        str(offset_s),
    ]
    if topics:
        cmd.extend(["--topics", *topics])
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_bag_or_duration(bag_proc: subprocess.Popen, wall_timeout_s: float) -> None:
    deadline = time.monotonic() + wall_timeout_s
    while time.monotonic() < deadline:
        if bag_proc.poll() is not None:
            break
        time.sleep(0.2)


def _lookup_init_pose_3d(ref_data: dict[str, Any], offset_s: float) -> tuple[float, float, float, float]:
    """Return (x, y, z, yaw) at offset_s from load_reference_from_bag dict."""
    poses = ref_data["poses"]
    poses_3d = ref_data["poses_3d"]
    if not poses:
        return (0.0, 0.0, 0.0, 0.0)
    best_i = min(range(len(poses)), key=lambda i: abs(poses[i][3] - offset_s))
    return (float(poses[best_i][0]), float(poses[best_i][1]), float(poses_3d[best_i][3]), float(poses[best_i][2]))


def _run_gn10_method(
    method: dict[str, Any],
    dataset_cfg: dict[str, Any],
    bag_path: Path,
    ref_data: dict[str, Any],
    init_pose: tuple[float, float, float, float],
    offset_s: float,
    duration_s: float,
    rate: float,
    out_dir: Path,
    suite_name: str,
    env: dict[str, str],
) -> dict[str, Any]:
    node_bin, _, default_params = resolve_localization_paths()
    init_x, init_y, _, init_yaw = init_pose
    map_type = method.get("map_type", dataset_cfg.get("map_type", "json"))
    map_file = _resolve_repo_or_user_path(method.get("map_file", dataset_cfg["map_file"]))
    pipeline_csv = out_dir / f"{suite_name}__{method['label']}_pipeline.csv"

    merged_params: dict[str, Any] = {
        "use_sim_time": True,
        "map_source_type": map_type,
        "map_file_path": str(map_file),
        "benchmark.enable": True,
        "benchmark.csv_output_path": str(pipeline_csv.resolve()),
        "initial_pose.use_for_local_search": True,
        "initial_pose.x": init_x,
        "initial_pose.y": init_y,
        "initial_pose.yaw": init_yaw,
    }
    if "lidar_2d" in dataset_cfg:
        merged_params["lidar_2d.enable"] = bool(dataset_cfg["lidar_2d"])
    merged_params.update(dataset_cfg.get("params", {}))
    merged_params.update(method.get("params", {}))

    remaps: list[str] = []
    for dst, src in (dataset_cfg.get("remaps") or {}).items():
        remaps.extend(["-r", f"{dst}:={src}"])

    cmd = [
        node_bin,
        "--ros-args",
        "--params-file",
        default_params,
        *_ros_param_cli_args(merged_params),
        *remaps,
    ]
    log_path = out_dir / f"{suite_name}__{method['label']}.log"
    listener = EvalListener(ref_data["tf_static_msgs"])
    executor, stop_evt, th = _spin_executor_thread(listener)
    log_f = open(log_path, "w", encoding="utf-8")
    node_proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    monitor = SystemGPUMonitor(
        target_process_name="gn10_pointcloud_localization_node",
        sample_interval_s=0.1,
        target_pid=node_proc.pid,
    )
    monitor.start()
    bag_proc = None
    try:
        ready = False
        wait_deadline = time.monotonic() + 15.0
        while time.monotonic() < wait_deadline:
            log_f.flush()
            if "Publishing localization telemetry on:" in log_path.read_text(encoding="utf-8", errors="ignore"):
                ready = True
                break
            if node_proc.poll() is not None:
                break
            time.sleep(0.05)
        if not ready:
            raise RuntimeError(f"gn10 node failed to initialize for {suite_name}::{method['label']}")
        time.sleep(0.2)
        listener.active = True
        play_topics = dataset_cfg.get("play_topics") or ["/livox/lidar", "/livox/imu", "/lakibeam/scan"]
        bag_proc = _play_bag(bag_path, offset_s, rate, env, play_topics)
        _wait_bag_or_duration(bag_proc, duration_s / max(rate, 0.1) + 3.0)
        time.sleep(0.5)
    finally:
        terminate_process_group(bag_proc)
        terminate_process_group(node_proc)
        log_f.close()
        monitor.stop()
        stop_evt.set()
        th.join(timeout=2.0)
        executor.remove_node(listener)
        listener.destroy_node()

    raw_poses = listener.poses_3d
    dedup_poses: dict[int, tuple[int, float, float, float, float, float, float, float]] = {}
    for row in raw_poses:
        dedup_poses[row[0]] = row
    if len(dedup_poses) != len(raw_poses):
        raise RuntimeError(
            f"Duplicate pose timestamps detected in {suite_name}::{method['label']} "
            f"({len(raw_poses)} raw vs {len(dedup_poses)} unique) — aborting to prevent contaminated metrics!"
        )
    est_poses = [dedup_poses[k] for k in sorted(dedup_poses.keys())]
    est_tum = out_dir / f"{suite_name}__{method['label']}_est_tum.txt"
    save_tum_file(est_poses, est_tum)
    ate = compute_ate_rpe_metrics(est_poses, ref_data["poses_3d"])
    lat_vals: list[float] = []
    match_lat_vals: list[float] = []
    filter_lat_vals: list[float] = []
    if pipeline_csv.exists() and pipeline_csv.stat().st_size > 0:
        df_pipe = pd.read_csv(pipeline_csv)
        if "time_total_ms" in df_pipe.columns:
            lat_vals = [float(x) for x in df_pipe["time_total_ms"].dropna().tolist() if x > 0.0]
        if "time_matching_ms" in df_pipe.columns:
            match_lat_vals = [float(x) for x in df_pipe["time_matching_ms"].dropna().tolist() if x > 0.0]
        if "time_ground_filter_ms" in df_pipe.columns:
            filter_lat_vals = [float(x) for x in df_pipe["time_ground_filter_ms"].dropna().tolist() if x > 0.0]
    res_df = monitor.to_dataframe()
    res_df.to_csv(out_dir / f"{suite_name}__{method['label']}_resources.csv", index=False)

    match_summary = _summarize_latencies(match_lat_vals)
    filter_summary = _summarize_latencies(filter_lat_vals)
    return {
        "method": method["label"],
        "type": "gn10",
        "est_poses_count": len(est_poses),
        **ate,
        **_summarize_latencies(lat_vals),
        "match_latency_mean_ms": match_summary["latency_mean_ms"],
        "match_latency_p95_ms": match_summary["latency_p95_ms"],
        "filter_latency_mean_ms": filter_summary["latency_mean_ms"],
        **_summarize_resources(res_df, monitor.get_cumulative_cpu_pct()),
    }


def _parse_dll_solve_latencies_ms(log_path: Path) -> list[float]:
    """Parse '[INFO] [sec.nsec] [dll_node]: Solving' -> 'Solved!' delta times in ms."""
    stamp_re = re.compile(r"\[(\d+\.\d+)\]\s+\[dll_node\]:\s+(Solving|Solved!)")
    t_start: float | None = None
    lat_ms: list[float] = []
    if not log_path.exists():
        return lat_ms
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = stamp_re.search(line)
            if not m:
                continue
            ts = float(m.group(1))
            tag = m.group(2)
            if tag == "Solving":
                t_start = ts
            elif tag == "Solved!" and t_start is not None:
                dt = (ts - t_start) * 1000.0
                if 0.0 <= dt < 5000.0:
                    lat_ms.append(dt)
                t_start = None
    return lat_ms


def _run_dll_method(
    method: dict[str, Any],
    bag_path: Path,
    ref_data: dict[str, Any],
    init_pose: tuple[float, float, float, float],
    offset_s: float,
    duration_s: float,
    rate: float,
    suite_cfg: dict[str, Any],
    ext_ws: Path,
    bin_rel: str,
    out_dir: Path,
    suite_name: str,
    env: dict[str, str],
) -> dict[str, Any]:
    dll_bin = str(ext_ws / bin_rel)
    map_bt = _resolve_repo_or_user_path(method["map_file"])
    min_x, min_y, min_z = [float(v) for v in method.get("shift_xyz", [0.0, 0.0, 0.0])]
    init_x, init_y, init_z, init_yaw = init_pose
    init_a = init_yaw if abs(init_yaw) > 1e-4 else 1e-4

    cloud_topic = method.get("cloud_topic", suite_cfg["cloud_topic"])
    params: dict[str, Any] = {
        "use_sim_time": True,
        "map_path": str(map_bt),
        "in_cloud": cloud_topic,
        "global_frame_id": "map",
        "odom_frame_id": "odom",
        "base_frame_id": "base_link",
        "initial_x": init_x - min_x,
        "initial_y": init_y - min_y,
        "initial_z": init_z - min_z,
        "initial_a": init_a,
        "publish_point_cloud_rate": 0.1,
    }
    params.update(method.get("params", {}))
    cmd = [dll_bin, "--ros-args", *_ros_param_cli_args(params)]
    log_path = out_dir / f"{suite_name}__{method['label']}.log"

    use_imu_cv_odom = bool(method.get("use_imu_cv_odom", False))
    imu_topic = method.get("imu_topic", suite_cfg.get("imu_topic", "/livox/imu"))

    # Upstream dllnode.hpp (L309-310) adds +100,000 us (+0.10 s) to cloud->header.stamp
    # when publishing map->odom TF; subtract 0.10 s here without modifying upstream binaries.
    listener = ExternalPoseListener(
        ref_data["tf_static_msgs"],
        listen_tf_parent="map",
        listen_tf_child="odom",
        shift_xyz=(min_x, min_y, min_z),
        stamp_offset_ns=-100_000_000,
        imu_cv_odom_topic=imu_topic if use_imu_cv_odom else None,
    )
    executor, stop_evt, th = _spin_executor_thread(listener)
    odom_tf_proc = None if use_imu_cv_odom else _start_odom_identity_tf(env)
    node_proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    monitor = SystemGPUMonitor(
        target_pid=node_proc.pid, target_process_name="dll_node", sample_interval_s=0.1
    )
    monitor.start()
    bag_proc = None
    try:
        time.sleep(2.5)
        default_topics = [cloud_topic, imu_topic] if use_imu_cv_odom else [cloud_topic]
        topics = list(method.get("play_topics") or suite_cfg.get("play_topics") or default_topics)
        bag_proc = _play_bag(bag_path, offset_s, rate, env, topics)
        _wait_bag_or_duration(bag_proc, duration_s / max(rate, 0.1) + 4.0)
        time.sleep(0.5)
    finally:
        terminate_process_group(bag_proc)
        terminate_process_group(node_proc)
        terminate_process_group(odom_tf_proc)
        monitor.stop()
        stop_evt.set()
        th.join(timeout=2.0)
        executor.remove_node(listener)
        listener.destroy_node()

    est_poses, _, _, _ = listener.snapshot()
    est_tum = out_dir / f"{suite_name}__{method['label']}_est_tum.txt"
    save_tum_file(est_poses, est_tum)
    ate = compute_ate_rpe_metrics(est_poses, ref_data["poses_3d"])
    lat_vals = _parse_dll_solve_latencies_ms(log_path)
    res_df = monitor.to_dataframe()
    res_df.to_csv(out_dir / f"{suite_name}__{method['label']}_resources.csv", index=False)

    return {
        "method": method["label"],
        "type": "dll",
        "est_poses_count": len(est_poses),
        **ate,
        **_summarize_latencies(lat_vals),
        **_summarize_resources(res_df, cum_cpu_pct=monitor.get_cumulative_cpu_pct()),
    }


def _run_lidar_loc_method(
    method: dict[str, Any],
    bag_path: Path,
    ref_data: dict[str, Any],
    init_pose: tuple[float, float, float, float],
    offset_s: float,
    duration_s: float,
    rate: float,
    suite_cfg: dict[str, Any],
    ext_ws: Path,
    bin_rel: str,
    out_dir: Path,
    suite_name: str,
    env: dict[str, str],
) -> dict[str, Any]:
    loc_bin = str(ext_ws / bin_rel)
    map_pcd = _resolve_repo_or_user_path(method["map_file"])
    init_x, init_y, init_z, init_yaw = init_pose
    qx, qy, qz, qw = _yaw_to_quat(init_yaw)

    params: dict[str, Any] = {
        "use_sim_time": True,
        "use_pcd_map": True,
        "map_path": str(map_pcd),
        "global_frame_id": "map",
        "odom_frame_id": "odom",
        "base_frame_id": "base_link",
        "set_initial_pose": True,
        "initial_pose_x": init_x,
        "initial_pose_y": init_y,
        "initial_pose_z": init_z,
        "initial_pose_qx": qx,
        "initial_pose_qy": qy,
        "initial_pose_qz": qz,
        "initial_pose_qw": qw,
        "enable_scan_voxel_filter": True,
        "enable_local_map_crop": False,
        "publish_tf": True,
    }
    params.update(method.get("params", {}))
    cloud_topic = method.get("cloud_topic", suite_cfg["cloud_topic"])
    imu_topic = method.get("imu_topic", suite_cfg["imu_topic"])
    remaps = ["-r", f"/cloud:={cloud_topic}", "-r", f"/imu:={imu_topic}"]
    cmd = [loc_bin, "--ros-args", *_ros_param_cli_args(params), *remaps]
    log_path = out_dir / f"{suite_name}__{method['label']}.log"

    listener = ExternalPoseListener(
        ref_data["tf_static_msgs"],
        listen_pcl_pose=True,
        listen_diag_status=True,
    )
    executor, stop_evt, th = _spin_executor_thread(listener)
    odom_tf_proc = _start_odom_identity_tf(env)
    node_proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    monitor = SystemGPUMonitor(
        target_pid=node_proc.pid,
        target_process_name="lidar_localization_node",
        sample_interval_s=0.1,
    )
    monitor.start()
    bag_proc = None
    try:
        # Wait until /lidar_localization exposes lifecycle get_state service (can take >2s on ARM64)
        ready_deadline = time.monotonic() + 15.0
        while time.monotonic() < ready_deadline:
            probe = subprocess.run(
                ["ros2", "lifecycle", "get", "/lidar_localization"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            if probe.returncode == 0 and "unconfigured" in probe.stdout.lower():
                break
            time.sleep(0.4)
        for tr_name in ("configure", "activate"):
            subprocess.run(
                ["ros2", "lifecycle", "set", "/lidar_localization", tr_name],
                env=env,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        topics = [cloud_topic]
        if params.get("use_imu_preintegration", False) or params.get(
            "use_continuous_time_deskew", False
        ):
            topics.append(imu_topic)
        bag_proc = _play_bag(bag_path, offset_s, rate, env, topics)
        _wait_bag_or_duration(bag_proc, duration_s / max(rate, 0.1) + 4.0)
        time.sleep(0.5)
    finally:
        terminate_process_group(bag_proc)
        terminate_process_group(node_proc)
        terminate_process_group(odom_tf_proc)
        monitor.stop()
        stop_evt.set()
        th.join(timeout=2.0)
        executor.remove_node(listener)
        listener.destroy_node()

    est_poses, lat_vals, accepted_cnt, rejected_cnt = listener.snapshot()
    est_tum = out_dir / f"{suite_name}__{method['label']}_est_tum.txt"
    save_tum_file(est_poses, est_tum)
    ate = compute_ate_rpe_metrics(est_poses, ref_data["poses_3d"])
    res_df = monitor.to_dataframe()
    res_df.to_csv(out_dir / f"{suite_name}__{method['label']}_resources.csv", index=False)

    return {
        "method": method["label"],
        "type": "lidar_loc",
        "est_poses_count": len(est_poses),
        "accepted_frames": accepted_cnt,
        "rejected_frames": rejected_cnt,
        **ate,
        **_summarize_latencies(lat_vals),
        **_summarize_resources(res_df, cum_cpu_pct=monitor.get_cumulative_cpu_pct()),
    }


def _extract_bag_static_transforms(bag_path: Path) -> dict[str, np.ndarray]:
    """Return dict mapping child_frame_id -> 4x4 T_base_child from /tf_static."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    out: dict[str, np.ndarray] = {}
    with AnyReader([bag_path], default_typestore=ts) as r:
        for conn, _, raw in r.messages(connections=[c for c in r.connections if c.topic == "/tf_static"]):
            msg = r.deserialize(raw, conn.msgtype)
            for tf in msg.transforms:
                child = tf.child_frame_id.lstrip("/")
                tr = tf.transform.translation
                q = tf.transform.rotation
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = _quat_to_rot_matrix(q.x, q.y, q.z, q.w)
                T[:3, 3] = [tr.x, tr.y, tr.z]
                out[child] = T
    return out


def _extract_point_timestamps_ns(msg: Any, valid_mask: np.ndarray, fallback_ns: int) -> np.ndarray:
    """Return per-point timestamps in integer nanoseconds for PointCloud2 messages."""
    n_pts = len(valid_mask)
    ts_field = next((f for f in msg.fields if f.name in ("timestamp", "time", "t")), None)
    if ts_field is None or ts_field.offset + (8 if ts_field.datatype == 8 else 4) > msg.point_step:
        return np.full(int(valid_mask.sum()), fallback_ns, dtype=np.int64)
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)[valid_mask]
    hdr_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
    if ts_field.datatype == 8:  # FLOAT64
        vals = np.frombuffer(arr[:, ts_field.offset : ts_field.offset + 8].tobytes(), dtype=np.float64)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        # Heuristic: > 1e12 is nanoseconds (epoch or >1000s), > 1e5 is microseconds/relative seconds? Check magnitude vs hdr_ns
        if np.median(vals) > 1e12 or (hdr_ns < 1e12 and np.median(vals) > 1e6):
            pts_ns = np.rint(vals).astype(np.int64)
        elif np.median(vals) < 10.0 and hdr_ns > 0:
            pts_ns = hdr_ns + np.rint(vals * 1e9).astype(np.int64)
        else:
            pts_ns = np.rint(vals * 1e9).astype(np.int64)
        return pts_ns
    if ts_field.datatype == 7:  # FLOAT32 relative seconds
        vals = np.frombuffer(arr[:, ts_field.offset : ts_field.offset + 4].tobytes(), dtype=np.float32).astype(np.float64)
        return hdr_ns + np.rint(np.where(np.isfinite(vals), vals, 0.0) * 1e9).astype(np.int64)
    if ts_field.datatype == 6:  # UINT32 nanoseconds offset
        vals = np.frombuffer(arr[:, ts_field.offset : ts_field.offset + 4].tobytes(), dtype=np.uint32).astype(np.int64)
        return hdr_ns + vals
    return np.full(int(valid_mask.sum()), fallback_ns, dtype=np.int64)


def _run_kiss_icp_method(
    method: dict[str, Any],
    bag_path: Path,
    ref_data: dict[str, Any],
    init_ref_3d: tuple[int, float, float, float, float, float, float, float],
    offset_s: float,
    duration_s: float,
    suite_cfg: dict[str, Any],
    out_dir: Path,
    suite_name: str,
) -> dict[str, Any]:
    from kiss_icp.config import load_config
    from kiss_icp.kiss_icp import KissICP
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    cfg = load_config(None)
    cfg.data.min_range = float(method.get("min_range", 0.45))
    cfg.data.max_range = float(method.get("max_range", 30.0))
    use_deskew = bool(method.get("deskew", False))
    cfg.data.deskew = use_deskew
    if "voxel_size" in method:
        cfg.mapping.voxel_size = float(method["voxel_size"])
    od = KissICP(cfg)

    tfs = _extract_bag_static_transforms(bag_path)
    ts = get_typestore(Stores.ROS2_HUMBLE)
    T_map_base0 = np.eye(4, dtype=np.float64)
    T_map_base0[:3, :3] = _quat_to_rot_matrix(init_ref_3d[4], init_ref_3d[5], init_ref_3d[6], init_ref_3d[7])
    T_map_base0[:3, 3] = [init_ref_3d[1], init_ref_3d[2], init_ref_3d[3]]

    est_poses: list[tuple[int, float, float, float, float, float, float, float]] = []
    lat_vals: list[float] = []
    cloud_topic = method.get("cloud_topic", suite_cfg["cloud_topic"])

    with AnyReader([bag_path], default_typestore=ts) as reader:
        conns = [c for c in reader.connections if c.topic == cloud_topic]
        t_min_ns = reader.start_time + int(offset_s * 1e9)
        t_max_ns = t_min_ns + int(duration_s * 1e9)
        for conn, bag_t_ns, raw in reader.messages(connections=conns, start=t_min_ns):
            if bag_t_ns < t_min_ns:
                continue
            if bag_t_ns > t_max_ns:
                break
            msg = reader.deserialize(raw, conn.msgtype)
            stamp_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
            xyz = np.frombuffer(arr[:, :12].tobytes(), dtype=np.float32).reshape(-1, 3)
            valid = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) >= 0.15)
            xyz = xyz[valid]
            if len(xyz) == 0:
                continue
            frame_id = msg.header.frame_id.lstrip("/")
            if frame_id != "base_link" and frame_id in tfs:
                T_bl = tfs[frame_id]
                pts_base = (T_bl[:3, :3] @ xyz.astype(np.float64).T).T + T_bl[:3, 3]
            else:
                pts_base = xyz.astype(np.float64)
            if use_deskew:
                pts_ns = _extract_point_timestamps_ns(msg, valid, stamp_ns)
                t0_ns = int(pts_ns.min())
                t1_ns = int(pts_ns.max())
                span_ns = max(t1_ns - t0_ns, 1)
                timestamps = (pts_ns - t0_ns).astype(np.float64) / float(span_ns)
                # KISS-ICP deskews points onto the end timestamp (normalized t=1.0).
                out_stamp_ns = t1_ns
            else:
                timestamps = np.zeros(len(pts_base), dtype=np.float64)
                out_stamp_ns = stamp_ns
            t0 = time.perf_counter()
            od.register_frame(pts_base, timestamps)
            lat_vals.append((time.perf_counter() - t0) * 1000.0)
            T_odom = od.last_pose
            T_map = T_map_base0 @ T_odom
            qx, qy, qz, qw = _rot_matrix_to_quat(T_map[:3, :3])
            est_poses.append((out_stamp_ns, float(T_map[0, 3]), float(T_map[1, 3]), float(T_map[2, 3]), qx, qy, qz, qw))

    est_tum = out_dir / f"{suite_name}__{method['label']}_est_tum.txt"
    save_tum_file(est_poses, est_tum)
    ate = compute_ate_rpe_metrics(est_poses, ref_data["poses_3d"])
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "method": method["label"],
        "type": "kiss_icp",
        "est_poses_count": len(est_poses),
        **ate,
        **_summarize_latencies(lat_vals),
        "ram_rss_max_mb": float(rss_mb),
    }


def _enter_isolated_domain(domain_id: int, ext_ws: Path) -> dict[str, str]:
    env = _build_env(domain_id, ext_ws)
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    if rclpy.ok():
        rclpy.shutdown()
    rclpy.init()
    return env


def _leave_isolated_domain() -> None:
    if rclpy.ok():
        rclpy.shutdown()


def run_tracking_command(args: argparse.Namespace) -> None:
    baselines = _load_yaml(REPO_ROOT / "benchmarks" / "manifests" / "baselines.yaml")
    datasets = _load_yaml(REPO_ROOT / "benchmarks" / "manifests" / "datasets.yaml")["datasets"]
    ext_ws = Path(os.path.expanduser(args.external_ws)).resolve()
    out_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = baselines["external_binaries"]

    requested_suites = args.suites.split(",") if args.suites else list(baselines["tracking_suites"].keys())
    requested_methods = set(args.methods.split(",")) if args.methods else None
    results: list[dict[str, Any]] = []

    domain_base = int(args.domain_id)
    idx = 0
    for suite_name in requested_suites:
        suite_cfg = baselines["tracking_suites"][suite_name]
        ds_name = suite_cfg["dataset"]
        ds_cfg = datasets[ds_name]
        bag_path = _resolve_repo_or_user_path(ds_cfg["default_bag_path"])
        ref_topic = ds_cfg.get("ref_topic", "/platform_constraint")
        ref_data = load_reference_from_bag(bag_path, ref_topic=ref_topic)
        offset_s = float(suite_cfg.get("offset_s", 0.0))
        duration_s = float(args.duration if args.duration is not None else suite_cfg.get("duration_s", 30.0))
        rate = float(suite_cfg.get("rate", 1.0))

        init_pose = _lookup_init_pose_3d(ref_data, offset_s)
        ref_poses = ref_data["poses"]
        best_i = min(range(len(ref_poses)), key=lambda i: abs(ref_poses[i][3] - offset_s))
        init_ref_3d = ref_data["poses_3d"][best_i]
        save_tum_file(ref_data["poses_3d"], out_dir / f"{suite_name}__ref_tum.txt")

        for method in suite_cfg["methods"]:
            if requested_methods and method["label"] not in requested_methods:
                continue
            idx += 1
            mtype = method["type"]
            play_bag_path = _resolve_repo_or_user_path(
                method.get("bag_file", suite_cfg.get("bag_file", ds_cfg["default_bag_path"]))
            )
            print(f"[compare] suite={suite_name} method={method['label']} domain={domain_base + idx}", flush=True)
            if mtype == "kiss_icp":
                row = _run_kiss_icp_method(
                    method, play_bag_path, ref_data, init_ref_3d, offset_s, duration_s, suite_cfg, out_dir, suite_name
                )
            else:
                env = _enter_isolated_domain(domain_base + idx, ext_ws)
                try:
                    if mtype == "gn10":
                        row = _run_gn10_method(
                            method, ds_cfg, play_bag_path, ref_data, init_pose, offset_s, duration_s, rate, out_dir, suite_name, env
                        )
                    elif mtype == "dll":
                        row = _run_dll_method(
                            method,
                            play_bag_path,
                            ref_data,
                            init_pose,
                            offset_s,
                            duration_s,
                            rate,
                            suite_cfg,
                            ext_ws,
                            bins["dll_node"],
                            out_dir,
                            suite_name,
                            env,
                        )
                    elif mtype == "lidar_loc":
                        row = _run_lidar_loc_method(
                            method,
                            play_bag_path,
                            ref_data,
                            init_pose,
                            offset_s,
                            duration_s,
                            rate,
                            suite_cfg,
                            ext_ws,
                            bins["lidar_localization_node"],
                            out_dir,
                            suite_name,
                            env,
                        )
                    else:
                        raise ValueError(f"Unknown method type: {mtype}")
                finally:
                    _leave_isolated_domain()
            row["suite"] = suite_name
            row["dataset"] = ds_name
            row["ground_truth_kind"] = ds_cfg.get("ground_truth_kind", "unknown")
            results.append(row)
            print(
                f"  -> matched={row.get('matched_frames')} xy_rmse={_fmt(row.get('xy_rmse_m'))} "
                f"xy_mean={_fmt(row.get('xy_mean_m'))} lat_ms={_fmt(row.get('latency_mean_ms'), '.2f')}",
                flush=True,
            )

    json_out = out_dir / "tracking_compare_results.json"
    json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_tracking_reports(results, out_dir)
    print(f"Wrote tracking comparison results -> {json_out}")


def _fmt(v: float | int | None, fmt: str = ".4f") -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return format(v, fmt)


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def _write_tracking_reports(results: list[dict[str, Any]], out_dir: Path) -> None:
    md_lines = [
        "# Baseline Tracking Comparison Results",
        "",
        "| Suite | GT Kind | Method | Matched | XY RMSE (m) | XY Mean (m) | Yaw RMSE (rad) | >0.10m | Lat Mean (ms) | Lat P95 (ms) | CPU (%) | RAM Max (MB) | VDD_IN (W) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tex_lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\hline",
        r"Suite & Method & Matched & XY RMSE (m) & Yaw RMSE (rad) & $>$0.10m & Lat (ms) & CPU (\%) & RAM (MB) \\",
        r"\hline",
    ]
    for r in results:
        md_lines.append(
            f"| {r['suite']} | {r['ground_truth_kind']} | {r['method']} | "
            f"{r.get('matched_frames', 0)} | {_fmt(r.get('xy_rmse_m'))} | {_fmt(r.get('xy_mean_m'))} | "
            f"{_fmt(r.get('yaw_rmse_rad'))} | {r.get('frames_gt_010m', 0)} | "
            f"{_fmt(r.get('latency_mean_ms'), '.2f')} | {_fmt(r.get('latency_p95_ms'), '.2f')} | "
            f"{_fmt(r.get('cpu_node_mean_pct'), '.1f')} | {_fmt(r.get('ram_rss_max_mb'), '.1f')} | "
            f"{_fmt(r.get('gpu_power_mean_w'), '.2f')} |"
        )
        tex_lines.append(
            f"{_tex_escape(r['suite'])} & {_tex_escape(r['method'])} & "
            f"{r.get('matched_frames', 0)} & {_fmt(r.get('xy_rmse_m'))} & "
            f"{_fmt(r.get('yaw_rmse_rad'))} & {r.get('frames_gt_010m', 0)} & "
            f"{_fmt(r.get('latency_mean_ms'), '.2f')} & {_fmt(r.get('cpu_node_mean_pct'), '.1f')} & "
            f"{_fmt(r.get('ram_rss_max_mb'), '.1f')} \\\\"
        )
    tex_lines.extend([r"\hline", r"\end{tabular}"])
    (out_dir / "tracking_compare_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (out_dir / "tracking_compare_summary.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def _write_pcd_xyz(path: Path, pts: np.ndarray) -> None:
    pts32 = pts.astype(np.float32, copy=False)
    hdr = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(pts32)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(pts32)}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(hdr)
        f.write(pts32.tobytes())


def _integrate_base_imu_yaws(
    reader: Any,
    imu_topic: str,
    t_min_ns: int,
    t_max_ns: int,
    tfs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate base-frame yaw angle (rad) from /livox/imu over [t_min_ns, t_max_ns]."""
    imu_conns = [c for c in reader.connections if c.topic == imu_topic]
    if not imu_conns:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    stamps: list[int] = []
    omegas: list[float] = []
    for c, _, raw in reader.messages(connections=imu_conns, start=t_min_ns):
        msg = reader.deserialize(raw, c.msgtype)
        stamp_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        if stamp_ns < t_min_ns:
            continue
        if stamp_ns > t_max_ns:
            break
        w = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            dtype=np.float64,
        )
        frame = msg.header.frame_id.lstrip("/")
        if frame != "base_link" and frame in tfs:
            w = tfs[frame][:3, :3] @ w
        stamps.append(stamp_ns)
        omegas.append(float(w[2]))
    if len(stamps) < 2:
        return np.array(stamps, dtype=np.int64), np.zeros(len(stamps), dtype=np.float64)
    st_arr = np.asarray(stamps, dtype=np.int64)
    om_arr = np.asarray(omegas, dtype=np.float64)
    order = np.argsort(st_arr, kind="mergesort")
    st_arr = st_arr[order]
    om_arr = om_arr[order]
    dt_s = np.diff(st_arr) * 1e-9
    yaws = np.zeros_like(om_arr)
    yaws[1:] = np.cumsum(0.5 * (om_arr[:-1] + om_arr[1:]) * dt_s)
    return st_arr, yaws


def _extract_single_scan_pcd(
    bag_path: Path,
    offset_s: float,
    out_pcd: Path,
    robot_radius: float,
    min_z_filter: float,
    accum_window_s: float = 0.0,
    imu_topic: str = "/livox/imu",
) -> bool:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    tfs = _extract_bag_static_transforms(bag_path)
    ts = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_path], default_typestore=ts) as r:
        t_target = r.start_time + int(offset_s * 1e9)
        conns = [c for c in r.connections if c.topic in ("/livox/lidar", "/filtered_points")]
        if accum_window_s <= 0.0:
            for c, t, raw in r.messages(connections=conns, start=t_target):
                if t < t_target:
                    continue
                msg = r.deserialize(raw, c.msgtype)
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
                xyz = np.frombuffer(arr[:, :12].tobytes(), dtype=np.float32).reshape(-1, 3)
                xyz = xyz[np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) >= 0.15)]
                frame = msg.header.frame_id.lstrip("/")
                if frame != "base_link" and frame in tfs:
                    T_bl = tfs[frame]
                    pts = (T_bl[:3, :3] @ xyz.astype(np.float64).T).T + T_bl[:3, 3]
                else:
                    pts = xyz.astype(np.float64)
                r_xy = np.linalg.norm(pts[:, :2], axis=1)
                in_chassis = (r_xy <= robot_radius) & (pts[:, 2] >= 0.0) & (pts[:, 2] <= 1.2)
                pts = pts[(~in_chassis) & (pts[:, 2] >= min_z_filter) & (pts[:, 2] <= 2.9)]
                _write_pcd_xyz(out_pcd, pts)
                return len(pts) > 0
            return False

        accum_ns = int(round(accum_window_s * 1e9))
        # Match gn10: integrate packets ending at the first scan at/after t_target over [ref - accum_ns, ref].
        packets: list[tuple[np.ndarray, np.ndarray]] = []
        t_ref_ns = 0
        for c, t, raw in r.messages(connections=conns, start=max(r.start_time, t_target - accum_ns - 50_000_000)):
            msg = r.deserialize(raw, c.msgtype)
            hdr_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
            xyz = np.frombuffer(arr[:, :12].tobytes(), dtype=np.float32).reshape(-1, 3)
            valid = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) >= 0.15)
            xyz = xyz[valid]
            if len(xyz) == 0:
                if t >= t_target and packets:
                    break
                continue
            frame = msg.header.frame_id.lstrip("/")
            if frame != "base_link" and frame in tfs:
                T_bl = tfs[frame]
                pts = (T_bl[:3, :3] @ xyz.astype(np.float64).T).T + T_bl[:3, 3]
            else:
                pts = xyz.astype(np.float64)
            pts_ns = _extract_point_timestamps_ns(msg, valid, hdr_ns)
            packets.append((pts, pts_ns))
            t_ref_ns = max(t_ref_ns, int(pts_ns.max()))
            if t >= t_target:
                break
        if not packets:
            return False
        all_pts = np.vstack([p for p, _ in packets])
        all_ns = np.concatenate([ts_arr for _, ts_arr in packets])
        keep = all_ns >= (t_ref_ns - accum_ns)
        all_pts = all_pts[keep]
        all_ns = all_ns[keep]
        if len(all_pts) == 0:
            return False

        imu_st, imu_yaws = _integrate_base_imu_yaws(
            r, imu_topic, int(all_ns.min()) - 100_000_000, t_ref_ns + 100_000_000, tfs
        )
        if len(imu_st) >= 2:
            yaw_ref = float(np.interp(t_ref_ns, imu_st, imu_yaws))
            yaw_pts = np.interp(all_ns, imu_st, imu_yaws)
            dtheta = yaw_pts - yaw_ref
            c_th = np.cos(dtheta)
            s_th = np.sin(dtheta)
            x_new = c_th * all_pts[:, 0] - s_th * all_pts[:, 1]
            y_new = s_th * all_pts[:, 0] + c_th * all_pts[:, 1]
            all_pts[:, 0] = x_new
            all_pts[:, 1] = y_new

        r_xy = np.linalg.norm(all_pts[:, :2], axis=1)
        in_chassis = (r_xy <= robot_radius) & (all_pts[:, 2] >= 0.0) & (all_pts[:, 2] <= 1.2)
        all_pts = all_pts[(~in_chassis) & (all_pts[:, 2] >= min_z_filter) & (all_pts[:, 2] <= 2.9)]
        _write_pcd_xyz(out_pcd, all_pts)
        return len(all_pts) > 0
    return False


def run_global_compare_command(args: argparse.Namespace) -> None:
    baselines = _load_yaml(REPO_ROOT / "benchmarks" / "manifests" / "baselines.yaml")
    datasets = _load_yaml(REPO_ROOT / "benchmarks" / "manifests" / "datasets.yaml")["datasets"]
    ext_ws = Path(os.path.expanduser(args.external_ws)).resolve()
    out_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bbs_bin = str(ext_ws / baselines["external_binaries"]["bbs3d_bench"])
    modes = [m.strip() for m in args.bbs_modes.split(",") if m.strip()]

    requested_suites = args.gs_suites.split(",") if args.gs_suites else list(baselines["global_search_suites"].keys())
    all_rows: list[dict[str, Any]] = []
    ref_cache: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="bbs3d_scans_") as tmpdir:
        tmp_path = Path(tmpdir)
        for gs_name in requested_suites:
            sc_cfg = baselines["global_search_suites"][gs_name]
            manifest = _load_yaml(REPO_ROOT / sc_cfg["manifest"])
            scenarios = manifest["scenarios"]
            map_pcd = str(_resolve_repo_or_user_path(sc_cfg["map_pcd"]))

            scan_records: list[dict[str, Any]] = []
            list_file = tmp_path / f"{gs_name}_scans.txt"
            scan_paths: list[str] = []
            for sc in scenarios:
                if sc.get("name", "").startswith("C5"):
                    continue  # C5 is dynamic exit/re-entry tracking, not static trigger localization
                ds_key = sc_cfg.get("dataset_full", sc_cfg.get("dataset")) if sc.get("requires_out_of_arena") else sc_cfg.get("dataset_in", sc_cfg.get("dataset"))
                ds_cfg = datasets[ds_key]
                bag_path = _resolve_repo_or_user_path(ds_cfg["default_bag_path"])
                if not bag_path.exists():
                    continue
                ref_topic = ds_cfg.get("ref_topic", "/platform_constraint")
                if ds_key not in ref_cache:
                    ref_cache[ds_key] = load_reference_from_bag(bag_path, ref_topic=ref_topic)
                ref_data = ref_cache[ds_key]
                rx, ry, rz, ryaw = _lookup_init_pose_3d(ref_data, float(sc["offset"]))
                scan_pcd = tmp_path / f"{gs_name}_{sc['name']}.pcd"
                ok = _extract_single_scan_pcd(
                    bag_path,
                    float(sc["offset"]),
                    scan_pcd,
                    float(sc_cfg.get("robot_radius", 0.45)),
                    float(sc_cfg.get("min_z_filter", 0.05)),
                    accum_window_s=float(sc_cfg.get("accum_window_s", 0.0)),
                    imu_topic=str(sc_cfg.get("imu_topic", "/livox/imu")),
                )
                if not ok:
                    continue
                scan_paths.append(str(scan_pcd))
                scan_records.append(
                    {
                        "suite": gs_name,
                        "name": sc["name"],
                        "group": sc["group"],
                        "expected": sc.get("expected", "recover"),
                        "scan_pcd": str(scan_pcd),
                        "ref_x": rx,
                        "ref_y": ry,
                        "ref_z": rz,
                        "ref_yaw": ryaw,
                    }
                )
            list_file.write_text("\n".join(scan_paths) + "\n", encoding="utf-8")

            for mode in modes:
                cmd = [
                    bbs_bin,
                    mode,
                    map_pcd,
                    str(list_file),
                    str(sc_cfg["min_res"]),
                    str(sc_cfg["max_level"]),
                    str(args.score_thresh if args.score_thresh is not None else sc_cfg["score_thresh_pct"]),
                    str(sc_cfg["timeout_ms"]),
                    str(sc_cfg["rp_deg"]),
                    "0.0",
                    str(sc_cfg["src_voxel"]),
                    "8",
                ]
                proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
                json_lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip().startswith("{")]
                by_scan = {row["scan"]: row for row in json_lines}
                for rec in scan_records:
                    res = by_scan.get(rec["scan_pcd"], {})
                    pose = res.get("pose", [0, 0, 0, 0, 0, 0])
                    localized = bool(res.get("has_localized", False))
                    xy_err = math.hypot(pose[0] - rec["ref_x"], pose[1] - rec["ref_y"]) if localized else None
                    yaw_err = abs(wrap_angle(pose[5] - rec["ref_yaw"])) if localized else None
                    if rec["expected"] == "recover":
                        passed = bool(
                            localized and xy_err is not None and xy_err <= 0.50 and yaw_err is not None and yaw_err <= 0.26
                        )
                    else:
                        passed = not localized
                    all_rows.append(
                        {
                            "suite": gs_name,
                            "method": f"3d_bbs_{mode}",
                            "scenario": rec["name"],
                            "group": rec["group"],
                            "expected": rec["expected"],
                            "passed": passed,
                            "has_localized": localized,
                            "has_timed_out": bool(res.get("has_timed_out", False)),
                            "init_ms": res.get("init_ms"),
                            "search_ms": res.get("bbs_elapsed_ms"),
                            "wall_ms": res.get("wall_ms"),
                            "best_score": res.get("best_score"),
                            "best_score_pct": res.get("best_score_pct"),
                            "xy_err_m": xy_err,
                            "yaw_err_rad": yaw_err,
                        }
                    )

    json_out = out_dir / "global_compare_results.json"
    json_out.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    summary_lines = [
        "# Global Localization Comparison (3D-BBS vs gn10)",
        "",
        "| Suite | Method | Expected | Pass / Total | Init (ms) | Mean Search (ms) | Mean XY Err (m) | Mean Yaw Err (rad) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    tex_lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"Suite & Method & Pass / Total & Init (ms) & Search (ms) & XY Err (m) & Yaw Err (rad) \\",
        r"\hline",
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        groups.setdefault((row["suite"], row["method"], row["expected"]), []).append(row)
    for (s_name, meth, exp), rows in groups.items():
        passes = sum(1 for r in rows if r["passed"])
        init_ms = rows[0].get("init_ms")
        search_vals = [r["search_ms"] for r in rows if r.get("search_ms") is not None]
        xy_vals = [r["xy_err_m"] for r in rows if r.get("xy_err_m") is not None and r["passed"]]
        yaw_vals = [r["yaw_err_rad"] for r in rows if r.get("yaw_err_rad") is not None and r["passed"]]
        m_search = float(np.mean(search_vals)) if search_vals else None
        m_xy = float(np.mean(xy_vals)) if xy_vals else None
        m_yaw = float(np.mean(yaw_vals)) if yaw_vals else None
        summary_lines.append(
            f"| {s_name} | {meth} | {exp} | {passes}/{len(rows)} | "
            f"{_fmt(init_ms, '.2f')} | {_fmt(m_search, '.2f')} | "
            f"{_fmt(m_xy, '.4f')} | {_fmt(m_yaw, '.4f')} |"
        )
        tex_lines.append(
            f"{_tex_escape(s_name)} & {_tex_escape(meth)} ({exp}) & {passes}/{len(rows)} & "
            f"{_fmt(init_ms, '.2f')} & {_fmt(m_search, '.2f')} & "
            f"{_fmt(m_xy, '.4f')} & {_fmt(m_yaw, '.4f')} \\\\"
        )
    tex_lines.extend([r"\hline", r"\end{tabular}"])
    (out_dir / "global_compare_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (out_dir / "global_compare_summary.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"Wrote global search comparison results -> {json_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified Step 4 External SOTA Baseline Comparison Runner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("tracking", help="Run trajectory tracking comparisons")
    tr.add_argument("--external-ws", default="~/external_baselines_ws")
    tr.add_argument("--output-dir", required=True)
    tr.add_argument("--suites", default="", help="Comma-separated suite names")
    tr.add_argument("--methods", default="", help="Comma-separated method labels")
    tr.add_argument("--duration", type=float, default=None)
    tr.add_argument("--domain-id", type=int, default=71)

    gs = sub.add_parser("global-compare", help="Run 3D-BBS vs gn10 global search comparisons")
    gs.add_argument("--external-ws", default="~/external_baselines_ws")
    gs.add_argument("--output-dir", required=True)
    gs.add_argument("--gs-suites", default="", help="Comma-separated global_search_suites names")
    gs.add_argument("--bbs-modes", default="gpu,cpu")
    gs.add_argument("--score-thresh", type=float, default=None, help="Override 3D-BBS score_thresh_pct")

    args = ap.parse_args()
    if args.cmd == "tracking":
        run_tracking_command(args)
    elif args.cmd == "global-compare":
        run_global_compare_command(args)


if __name__ == "__main__":
    main()

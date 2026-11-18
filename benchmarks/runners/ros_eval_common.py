#!/usr/bin/env python3
# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0

"""
Shared ROS 2 bag playback, telemetry capture, and process management helpers
for reproducible localization benchmarks.
"""

import bisect
import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import yaml
import rosbag2_py
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from gn10_pointcloud_localization.msg import LocalizationTelemetry
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_angle(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def open_rosbag_reader(bag_path: Path):
    """Opens SequentialReader or SequentialCompressionReader based on metadata.yaml."""
    storage_opts = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    conv_opts = rosbag2_py.ConverterOptions("cdr", "cdr")

    meta_file = bag_path / "metadata.yaml"
    is_compressed = False
    if meta_file.exists():
        with open(meta_file, "r") as mf:
            meta = yaml.safe_load(mf) or {}
        bag_info = meta.get("rosbag2_bagfile_information", {})
        comp_mode = str(bag_info.get("compression_mode", "")).strip().lower()
        comp_fmt = str(bag_info.get("compression_format", "")).strip().lower()
        is_compressed = bool(comp_fmt) and comp_mode not in ("", "none")

    reader = (
        rosbag2_py.SequentialCompressionReader() if is_compressed else rosbag2_py.SequentialReader()
    )
    reader.open(storage_opts, conv_opts)
    return reader


def load_reference_from_bag(bag_path: Path, ref_topic: str = "/platform_constraint"):
    """Reads reference poses (TUM-compatible tuples) and /tf_static from a ROS 2 bag."""
    reader = open_rosbag_reader(bag_path)
    metadata = reader.get_metadata()
    bag_duration_s = metadata.duration.nanoseconds * 1e-9
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    ref_type = topic_types.get(ref_topic, "geometry_msgs/msg/PoseWithCovarianceStamped")

    stamps = []
    poses = []
    poses_3d = []
    tf_static_msgs = []
    ts_first = None

    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        if ts_first is None:
            ts_first = ts_ns
        if topic == ref_topic:
            if ref_type == "geometry_msgs/msg/PoseStamped":
                msg = deserialize_message(data, PoseStamped)
                pose_obj = msg.pose
            elif ref_type == "nav_msgs/msg/Odometry":
                msg = deserialize_message(data, Odometry)
                pose_obj = msg.pose.pose
            else:
                msg = deserialize_message(data, PoseWithCovarianceStamped)
                pose_obj = msg.pose.pose
            p = pose_obj.position
            q = pose_obj.orientation
            yaw = yaw_from_quat(q)
            header_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
            stamps.append(header_ns)
            poses.append((p.x, p.y, yaw, (ts_ns - ts_first) * 1e-9))
            poses_3d.append((header_ns, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
        elif topic == "/tf_static":
            tf_static_msgs.append(deserialize_message(data, TFMessage))

    return {
        "bag_duration_s": bag_duration_s,
        "stamps": stamps,
        "poses": poses,
        "poses_3d": poses_3d,
        "tf_static_msgs": tf_static_msgs,
    }


def lookup_ref_by_stamp(stamps, poses, target_ns):
    if not stamps:
        return (0.0, 0.0, 0.0, 0.0), float("inf")
    idx = bisect.bisect_left(stamps, target_ns)
    if idx == 0:
        cand = 0
    elif idx >= len(stamps):
        cand = len(stamps) - 1
    else:
        cand = idx if abs(stamps[idx] - target_ns) < abs(stamps[idx - 1] - target_ns) else idx - 1
    dt = abs(stamps[cand] - target_ns) * 1e-9
    return poses[cand], dt


def lookup_ref_by_offset(poses, offset_s):
    if not poses:
        return 0.0, 0.0, 0.0
    best = min(poses, key=lambda item: abs(item[3] - offset_s))
    return best[0], best[1], best[2]


def terminate_process_group(proc):
    """Terminates a subprocess group cleanly with SIGINT -> SIGKILL fallback."""
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGINT)
        proc.wait(timeout=2.0)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=2.0)
        except Exception:
            pass


def resolve_localization_paths():
    pkg_prefix = get_package_prefix("gn10_pointcloud_localization")
    share_dir = get_package_share_directory("gn10_pointcloud_localization")
    node_bin = os.path.join(
        pkg_prefix, "lib", "gn10_pointcloud_localization", "gn10_pointcloud_localization_node"
    )
    default_params = os.path.join(share_dir, "config", "common_params.yaml")
    return node_bin, share_dir, default_params


def build_static_tf_commands():
    """Returns static_transform_publisher commands matching localization.launch.py."""
    return [
        [
            "ros2", "run", "tf2_ros", "static_transform_publisher",
            "--x", "0.25", "--y", "-0.25", "--z", "1.09",
            "--yaw", "0.0", "--pitch", "-0.273", "--roll", "3.14159",
            "--frame-id", "base_link", "--child-frame-id", "livox_frame",
        ],
        [
            "ros2", "run", "tf2_ros", "static_transform_publisher",
            "--x", "-0.33", "--y", "0.0", "--z", "0.08",
            "--yaw", "3.13", "--pitch", "0.0", "--roll", "0.0",
            "--frame-id", "base_link", "--child-frame-id", "lakibeam_frame",
        ],
    ]


class EvalListener(Node):
    """Collects localized poses (2D + 3D TUM), telemetry, and simulation clock."""

    def __init__(self, tf_static_msgs):
        super().__init__("benchmark_eval_listener")
        self._lock = threading.Lock()
        self.poses = []
        self.poses_3d = []
        self.telems = []
        self.clock_start_ns = None
        self.clock_last_ns = None
        self.active = False

        qos_be = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_static = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_tf_static = self.create_publisher(TFMessage, "/tf_static", qos_static)
        merged_static = TFMessage()
        for msg in tf_static_msgs:
            merged_static.transforms.extend(msg.transforms)
        self._merged_static = merged_static

        self.create_subscription(Clock, "/clock", self._clock_cb, qos_be)
        self.create_subscription(
            PoseWithCovarianceStamped, "/platform_constraint", self._pose_cb, qos_be
        )
        self.create_subscription(LocalizationTelemetry, "/localization/telemetry", self._telem_cb, 200)
        self.create_timer(0.02, self.publish_static_tf)

    def reset(self):
        with self._lock:
            self.poses.clear()
            self.poses_3d.clear()
            self.telems.clear()
            self.clock_start_ns = None
            self.clock_last_ns = None
            self.active = True

    def stop_recording(self):
        with self._lock:
            self.active = False
            return list(self.poses), list(self.poses_3d), list(self.telems)

    def set_static_tf(self, tf_static_msgs):
        merged_static = TFMessage()
        for msg in tf_static_msgs:
            merged_static.transforms.extend(msg.transforms)
        with self._lock:
            self._merged_static = merged_static

    def publish_static_tf(self):
        with self._lock:
            msg = self._merged_static
        if msg.transforms:
            self.pub_tf_static.publish(msg)

    def _clock_cb(self, msg):
        with self._lock:
            if not self.active or not self.telems:
                return
            ns = int(msg.clock.sec) * 10**9 + int(msg.clock.nanosec)
            if self.clock_last_ns is not None and abs(ns - self.clock_last_ns) > 2 * 10**9:
                self.clock_start_ns = ns
            if self.clock_start_ns is None:
                self.clock_start_ns = ns
            self.clock_last_ns = ns

    def sim_elapsed(self):
        with self._lock:
            if self.clock_start_ns is None or self.clock_last_ns is None:
                return 0.0
            return (self.clock_last_ns - self.clock_start_ns) * 1e-9

    def _pose_cb(self, msg):
        with self._lock:
            if not self.active:
                return
            stamp_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            yaw = yaw_from_quat(q)
            self.poses.append((stamp_ns, p.x, p.y, yaw))
            self.poses_3d.append((stamp_ns, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    def _telem_cb(self, msg):
        with self._lock:
            if not self.active:
                return
            stamp_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
            self.telems.append({
                "stamp_ns": stamp_ns,
                "best_cost": msg.best_cost,
                "cost_3d": msg.cost_3d,
                "cost_2d": msg.cost_2d,
                "inlier_cost": msg.inlier_cost,
                "inlier_count": msg.inlier_count,
                "inlier_ratio": msg.inlier_ratio,
                "obs_3d": msg.obstacle_count_3d,
                "obs_2d": msg.obstacle_count_2d,
                "linear_speed": msg.linear_speed,
                "accepted": msg.match_accepted,
                "is_lost": msg.is_lost,
                "global_recovered": msg.global_recovered,
            })

#!/usr/bin/env python3
# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0
"""Accumulate high-rate Livox packets into 10 Hz (0.10 s) ROS 2 bags for fair SOTA baselines."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore


ALLOWED_PASSTHROUGH_TOPICS = {"/tf_static", "/livox/imu", "/lakibeam/scan"}


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _so3_exp(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        wx, wy, wz = rotvec
        return np.array(
            [[1.0, -wz, wy], [wz, 1.0, -wx], [-wy, wx, 1.0]],
            dtype=np.float64,
        )
    k = rotvec / angle
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rot_to_rotvec(R: np.ndarray) -> np.ndarray:
    trace_val = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(trace_val))
    if angle < 1e-12:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if np.pi - angle < 1e-6:
        # Handle near-180 deg safely
        diag = np.clip((np.diag(R) + 1.0) * 0.5, 0.0, 1.0)
        axis = np.sqrt(diag)
        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]
        n = np.linalg.norm(axis)
        return (axis / max(n, 1e-12)) * angle
    denom = 2.0 * np.sin(angle)
    return np.array(
        [(R[2, 1] - R[1, 2]) / denom, (R[0, 2] - R[2, 0]) / denom, (R[1, 0] - R[0, 1]) / denom],
        dtype=np.float64,
    ) * angle


def _extract_point_timestamps_ns(msg: Any, fallback_ns: int) -> np.ndarray:
    n_pts = int(msg.width) * int(msg.height)
    if n_pts == 0:
        return np.zeros(0, dtype=np.int64)
    ts_field = next((f for f in msg.fields if f.name in ("timestamp", "time", "t")), None)
    if ts_field is None or ts_field.offset + (8 if ts_field.datatype == 8 else 4) > msg.point_step:
        return np.full(n_pts, fallback_ns, dtype=np.int64)
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_pts, msg.point_step)
    hdr_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
    if ts_field.datatype == 8:
        vals = np.frombuffer(arr[:, ts_field.offset : ts_field.offset + 8].tobytes(), dtype=np.float64)
        vals = np.where(np.isfinite(vals), vals, float(fallback_ns))
        med = float(np.median(vals))
        if med > 1e12 or (hdr_ns < 1e12 and med > 1e6):
            return np.rint(vals).astype(np.int64)
        if med < 10.0 and hdr_ns > 0:
            return hdr_ns + np.rint(vals * 1e9).astype(np.int64)
        return np.rint(vals * 1e9).astype(np.int64)
    if ts_field.datatype == 7:
        vals = np.frombuffer(arr[:, ts_field.offset : ts_field.offset + 4].tobytes(), dtype=np.float32).astype(np.float64)
        return hdr_ns + np.rint(np.where(np.isfinite(vals), vals, 0.0) * 1e9).astype(np.int64)
    if ts_field.datatype == 6:
        vals = np.frombuffer(arr[:, ts_field.offset : ts_field.offset + 4].tobytes(), dtype=np.uint32).astype(np.int64)
        return hdr_ns + vals
    return np.full(n_pts, fallback_ns, dtype=np.int64)


class ImuOrientator:
    """Piecewise-integrated base-frame SO(3) orientation lookup from IMU angular velocity."""

    def __init__(self, stamps_ns: np.ndarray, rots: list[np.ndarray]) -> None:
        self.stamps_ns = stamps_ns
        self.rots = rots

    def rotation_at(self, t_ns: int) -> np.ndarray:
        if len(self.stamps_ns) == 0:
            return np.eye(3, dtype=np.float64)
        if t_ns <= self.stamps_ns[0]:
            return self.rots[0]
        if t_ns >= self.stamps_ns[-1]:
            return self.rots[-1]
        idx = bisect_right(self.stamps_ns, int(t_ns)) - 1
        t0 = int(self.stamps_ns[idx])
        t1 = int(self.stamps_ns[idx + 1])
        if t1 <= t0:
            return self.rots[idx]
        alpha = float(t_ns - t0) / float(t1 - t0)
        dR = self.rots[idx].T @ self.rots[idx + 1]
        return self.rots[idx] @ _so3_exp(_rot_to_rotvec(dR) * alpha)


def _load_static_tfs_and_imu(
    bag_path: Path, imu_topic: str, ts: Any
) -> tuple[dict[str, np.ndarray], ImuOrientator]:
    tfs: dict[str, np.ndarray] = {}
    imu_samples: list[tuple[int, np.ndarray, str]] = []
    with AnyReader([bag_path], default_typestore=ts) as r:
        conns = [c for c in r.connections if c.topic in ("/tf_static", imu_topic)]
        for c, _, raw in r.messages(connections=conns):
            msg = r.deserialize(raw, c.msgtype)
            if c.topic == "/tf_static":
                for tf in msg.transforms:
                    child = tf.child_frame_id.lstrip("/")
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = _quat_to_rot(
                        tf.transform.rotation.x,
                        tf.transform.rotation.y,
                        tf.transform.rotation.z,
                        tf.transform.rotation.w,
                    )
                    T[:3, 3] = [
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z,
                    ]
                    tfs[child] = T
            elif c.topic == imu_topic:
                st = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
                w = np.array(
                    [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                    dtype=np.float64,
                )
                imu_samples.append((st, w, msg.header.frame_id.lstrip("/")))

    imu_samples.sort(key=lambda item: item[0])
    if not imu_samples:
        return tfs, ImuOrientator(np.zeros(0, dtype=np.int64), [])

    stamps: list[int] = []
    omegas_base: list[np.ndarray] = []
    for st, w, frame in imu_samples:
        if frame != "base_link" and frame in tfs:
            w_b = tfs[frame][:3, :3] @ w
        else:
            w_b = w
        if stamps and st == stamps[-1]:
            omegas_base[-1] = 0.5 * (omegas_base[-1] + w_b)
        else:
            stamps.append(st)
            omegas_base.append(w_b)

    rots: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    for i in range(1, len(stamps)):
        dt = (stamps[i] - stamps[i - 1]) * 1e-9
        if 0.0 < dt < 0.5:
            w_avg = 0.5 * (omegas_base[i - 1] + omegas_base[i])
            rots.append(rots[-1] @ _so3_exp(w_avg * dt))
        else:
            rots.append(rots[-1].copy())
    return tfs, ImuOrientator(np.asarray(stamps, dtype=np.int64), rots)


def _build_concat_and_deskewed_msgs(
    packet_msgs: list[Any],
    tfs: dict[str, np.ndarray],
    imu_orient: ImuOrientator,
    ts: Any,
) -> tuple[Any, Any, int, int]:
    """Combine consecutive packets into raw (/livox/lidar) and IMU-deskewed (/livox/lidar_deskewed)."""
    PointCloud2 = ts.types["sensor_msgs/msg/PointCloud2"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]

    first_msg = packet_msgs[0]
    point_step = int(first_msg.point_step)
    raw_chunks: list[np.ndarray] = []
    pt_stamps: list[np.ndarray] = []
    for msg in packet_msgs:
        n_pts = int(msg.width) * int(msg.height)
        if n_pts == 0:
            continue
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_pts, point_step)
        hdr_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        raw_chunks.append(arr)
        pt_stamps.append(_extract_point_timestamps_ns(msg, hdr_ns))

    combined = np.vstack(raw_chunks)
    all_ts_ns = np.concatenate(pt_stamps)
    total_pts = combined.shape[0]
    t_start_ns = int(first_msg.header.stamp.sec) * 10**9 + int(first_msg.header.stamp.nanosec)
    t_end_ns = int(all_ts_ns.max()) if len(all_ts_ns) else t_start_ns

    # Raw 0.10 s scan keeps start timestamp to match Livox + lidar_localization_ros2 convention.
    raw_hdr = Header(
        stamp=Time(sec=t_start_ns // 10**9, nanosec=t_start_ns % 10**9),
        frame_id=first_msg.header.frame_id,
    )
    raw_cloud = PointCloud2(
        header=raw_hdr,
        height=1,
        width=total_pts,
        fields=first_msg.fields,
        is_bigendian=first_msg.is_bigendian,
        point_step=point_step,
        row_step=total_pts * point_step,
        data=np.ascontiguousarray(combined.reshape(-1)),
        is_dense=first_msg.is_dense,
    )

    # Deskewed 0.10 s scan applies per-point SO(3) IMU gyro motion removal to t_end_ns in base_link,
    # transforms back to the lidar frame so external pipelines apply their own T_base_lidar TF once,
    # and stamps at t_end_ns.
    deskew_arr = combined.copy()
    xyz = np.frombuffer(deskew_arr[:, :12].tobytes(), dtype=np.float32).reshape(total_pts, 3).astype(np.float64)
    finite_mask = np.isfinite(xyz).all(axis=1)
    frame_id = first_msg.header.frame_id.lstrip("/")
    T_bl = tfs.get(frame_id, np.eye(4, dtype=np.float64)) if frame_id != "base_link" else np.eye(4, dtype=np.float64)
    T_lb = np.linalg.inv(T_bl)

    if len(imu_orient.stamps_ns) >= 2 and finite_mask.any():
        pts_base = (T_bl[:3, :3] @ xyz[finite_mask].T).T + T_bl[:3, 3]
        R_ref_T = imu_orient.rotation_at(t_end_ns).T
        # Sample IMU rotation per unique packet or 1 ms bucket for fast, accurate interpolation.
        ts_valid = all_ts_ns[finite_mask]
        ts_buckets = (ts_valid // 500_000) * 500_000  # 0.5 ms resolution
        uniq_buckets, inv_idx = np.unique(ts_buckets, return_inverse=True)
        rel_rots = np.empty((len(uniq_buckets), 3, 3), dtype=np.float64)
        for bi, b_ns in enumerate(uniq_buckets):
            rel_rots[bi] = R_ref_T @ imu_orient.rotation_at(int(b_ns))
        rot_per_pt = rel_rots[inv_idx]
        pts_deskew_base = np.einsum("nij,nj->ni", rot_per_pt, pts_base)
        pts_deskew_lidar = (T_lb[:3, :3] @ pts_deskew_base.T).T + T_lb[:3, 3]
        deskew_arr[finite_mask, :12] = np.frombuffer(
            pts_deskew_lidar.astype(np.float32).tobytes(), dtype=np.uint8
        ).reshape(-1, 12)

    deskew_hdr = Header(
        stamp=Time(sec=t_end_ns // 10**9, nanosec=t_end_ns % 10**9),
        frame_id=first_msg.header.frame_id,
    )
    deskew_cloud = PointCloud2(
        header=deskew_hdr,
        height=1,
        width=total_pts,
        fields=first_msg.fields,
        is_bigendian=first_msg.is_bigendian,
        point_step=point_step,
        row_step=total_pts * point_step,
        data=np.ascontiguousarray(deskew_arr.reshape(-1)),
        is_dense=first_msg.is_dense,
    )
    return raw_cloud, deskew_cloud, t_start_ns, t_end_ns


def accumulate_bag(
    src_bag: Path,
    dst_bag: Path,
    cloud_topic: str,
    imu_topic: str,
    packets_per_scan: int,
    overwrite: bool,
) -> dict[str, int]:
    if dst_bag.exists():
        if not overwrite:
            raise FileExistsError(f"Destination bag already exists: {dst_bag}")
        shutil.rmtree(dst_bag)

    ts = get_typestore(Stores.ROS2_HUMBLE)
    tfs, imu_orient = _load_static_tfs_and_imu(src_bag, imu_topic, ts)

    counts = {"passthrough": 0, "scans_emitted": 0, "packets_consumed": 0}
    with AnyReader([src_bag], default_typestore=ts) as reader, Writer(
        dst_bag, version=8, storage_plugin=StoragePlugin.SQLITE3
    ) as writer:
        conn_map: dict[str, Any] = {}
        passthrough_topics = set(ALLOWED_PASSTHROUGH_TOPICS)
        if imu_topic:
            passthrough_topics.add(imu_topic)

        cloud_qos = ()
        for c in reader.connections:
            qos = getattr(c.ext, "offered_qos_profiles", ()) if getattr(c, "ext", None) else ()
            if c.topic == cloud_topic and qos:
                cloud_qos = qos
            if c.topic in passthrough_topics and c.topic not in conn_map:
                conn_map[c.topic] = writer.add_connection(
                    c.topic,
                    c.msgtype,
                    typestore=ts,
                    offered_qos_profiles=qos,
                )

        raw_out_conn = writer.add_connection(
            cloud_topic,
            "sensor_msgs/msg/PointCloud2",
            typestore=ts,
            offered_qos_profiles=cloud_qos,
        )
        deskew_out_conn = writer.add_connection(
            f"{cloud_topic}_deskewed",
            "sensor_msgs/msg/PointCloud2",
            typestore=ts,
            offered_qos_profiles=cloud_qos,
        )

        # Preserve QoS metadata for /tf_static if present
        packet_buf: list[tuple[int, Any]] = []
        active_topics = passthrough_topics | {cloud_topic}
        conns = [c for c in reader.connections if c.topic in active_topics]
        for c, bag_t_ns, raw in reader.messages(connections=conns):
            if c.topic in passthrough_topics:
                writer.write(conn_map[c.topic], bag_t_ns, raw)
                counts["passthrough"] += 1
                continue
            if c.topic == cloud_topic:
                msg = reader.deserialize(raw, c.msgtype)
                packet_buf.append((bag_t_ns, msg))
                counts["packets_consumed"] += 1
                if len(packet_buf) >= packets_per_scan:
                    msgs = [m for _, m in packet_buf]
                    last_bag_t = packet_buf[-1][0]
                    raw_cloud, deskew_cloud, _, _ = _build_concat_and_deskewed_msgs(
                        msgs, tfs, imu_orient, ts
                    )
                    writer.write(
                        raw_out_conn,
                        last_bag_t,
                        ts.serialize_cdr(raw_cloud, "sensor_msgs/msg/PointCloud2"),
                    )
                    writer.write(
                        deskew_out_conn,
                        last_bag_t,
                        ts.serialize_cdr(deskew_cloud, "sensor_msgs/msg/PointCloud2"),
                    )
                    counts["scans_emitted"] += 1
                    packet_buf.clear()

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source clean ROS 2 bag directory")
    parser.add_argument("--output", required=True, help="Output accumulated ROS 2 bag directory")
    parser.add_argument("--cloud-topic", default="/livox/lidar")
    parser.add_argument("--imu-topic", default="/livox/imu")
    parser.add_argument("--packets-per-scan", type=int, default=5, help="Packets per scan (5 -> 10 Hz for 50 Hz bags)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stats = accumulate_bag(
        Path(args.input).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        args.cloud_topic,
        args.imu_topic,
        args.packets_per_scan,
        args.overwrite,
    )
    print(f"[accumulate] done: {stats}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

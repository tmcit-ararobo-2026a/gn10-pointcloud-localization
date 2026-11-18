#!/usr/bin/env python3
# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0

"""
Pure Apache-2.0 Trajectory Exporter and ATE/RPE Evaluator.

Converts ROS 2 bag trajectories (/platform_constraint or /tf) to standard TUM format
(`timestamp tx ty tz qx qy qz qw`) and computes timestamp-matched SE(2)/SE(3) ATE
and RPE metrics without linking or importing any copyleft libraries.
The exported TUM files remain directly usable with external CLI tools if desired.
"""

import argparse
import bisect
import json
import math
import sys
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage

# Allow running both directly and via package import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))
from ros_eval_common import load_reference_from_bag, open_rosbag_reader, wrap_angle, yaw_from_quat


def load_tum_file(tum_path: Path):
    """Loads TUM trajectory file into list of (stamp_ns, x, y, z, qx, qy, qz, qw)."""
    rows = []
    with open(tum_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 8:
                continue
            t_ns = int(round(float(parts[0]) * 1e9))
            vals = [float(v) for v in parts[1:8]]
            rows.append((t_ns, *vals))
    rows.sort(key=lambda r: r[0])
    return rows


def save_tum_file(poses_3d, out_path: Path):
    """Writes list of (stamp_ns, x, y, z, qx, qy, qz, qw) to standard TUM text format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for stamp_ns, x, y, z, qx, qy, qz, qw in poses_3d:
            f.write(
                f"{stamp_ns * 1e-9:.9f} {x:.6f} {y:.6f} {z:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )


def load_tf_poses_from_bag(bag_path: Path, parent_frame: str = "map", child_frame: str = "base_link"):
    """Extracts 3D poses from /tf topic in a ROS 2 bag as TUM-compatible tuples."""
    reader = open_rosbag_reader(bag_path)
    poses_3d = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == "/tf":
            msg = deserialize_message(data, TFMessage)
            for tf in msg.transforms:
                if tf.header.frame_id.lstrip("/") == parent_frame and tf.child_frame_id.lstrip("/") == child_frame:
                    stamp_ns = int(tf.header.stamp.sec) * 10**9 + int(tf.header.stamp.nanosec)
                    tr = tf.transform.translation
                    rot = tf.transform.rotation
                    poses_3d.append((stamp_ns, tr.x, tr.y, tr.z, rot.x, rot.y, rot.z, rot.w))
    poses_3d.sort(key=lambda r: r[0])
    return poses_3d


def compute_ate_rpe_metrics(est_poses_3d, ref_poses_3d, max_dt_s: float = 0.05, rpe_delta_s: float = 1.0):
    """
    Computes timestamp-matched SE(2) XY/Yaw ATE and temporal RPE metrics.

    Note: Uses explicit timestamp matching in map frame (no trajectory alignment cheat)
    so global accuracy against surveyed/reference map frames is measured faithfully.
    """
    if not est_poses_3d or not ref_poses_3d:
        return {
            "matched_frames": 0,
            "xy_rmse_m": None,
            "xy_mean_m": None,
            "xy_std_m": None,
            "xy_median_m": None,
            "xy_p95_m": None,
            "xy_max_m": None,
            "yaw_rmse_rad": None,
            "rpe_trans_rmse_m": None,
            "frames_gt_005m": 0,
            "frames_gt_010m": 0,
            "frames_gt_015m": 0,
        }

    ref_stamps = [r[0] for r in ref_poses_3d]

    def _yaw_of(r):
        class Q:
            def __init__(self, x, y, z, w):
                self.x, self.y, self.z, self.w = x, y, z, w
        return yaw_from_quat(Q(r[4], r[5], r[6], r[7]))

    matched_pairs = []
    xy_errs = []
    yaw_errs = []

    for est in est_poses_3d:
        t_ns = est[0]
        idx = bisect.bisect_left(ref_stamps, t_ns)
        cand = None
        if idx == 0:
            cand = 0
        elif idx >= len(ref_stamps):
            cand = len(ref_stamps) - 1
        else:
            cand = idx if abs(ref_stamps[idx] - t_ns) < abs(ref_stamps[idx - 1] - t_ns) else idx - 1
        dt_s = abs(ref_stamps[cand] - t_ns) * 1e-9
        if dt_s <= max_dt_s:
            ref = ref_poses_3d[cand]
            ex = est[1] - ref[1]
            ey = est[2] - ref[2]
            xy_err = math.hypot(ex, ey)
            eyaw = abs(wrap_angle(_yaw_of(est) - _yaw_of(ref)))
            xy_errs.append(xy_err)
            yaw_errs.append(eyaw)
            matched_pairs.append((t_ns, est[1], est[2], _yaw_of(est), ref[1], ref[2], _yaw_of(ref)))

    if not xy_errs:
        return {
            "matched_frames": 0,
            "xy_rmse_m": None,
            "xy_mean_m": None,
            "xy_std_m": None,
            "xy_median_m": None,
            "xy_p95_m": None,
            "xy_max_m": None,
            "yaw_rmse_rad": None,
            "rpe_trans_rmse_m": None,
            "frames_gt_005m": 0,
            "frames_gt_010m": 0,
            "frames_gt_015m": 0,
        }

    arr_xy = np.array(xy_errs, dtype=np.float64)
    arr_yaw = np.array(yaw_errs, dtype=np.float64)

    # Temporal translational RPE over delta = rpe_delta_s
    rpe_errs = []
    j = 0
    delta_ns = int(rpe_delta_s * 1e9)
    n_m = len(matched_pairs)
    for i in range(n_m):
        while j < n_m and (matched_pairs[j][0] - matched_pairs[i][0]) < delta_ns:
            j += 1
        if j < n_m and abs((matched_pairs[j][0] - matched_pairs[i][0]) - delta_ns) <= int(0.2 * 1e9):
            dx_est = matched_pairs[j][1] - matched_pairs[i][1]
            dy_est = matched_pairs[j][2] - matched_pairs[i][2]
            dx_ref = matched_pairs[j][4] - matched_pairs[i][4]
            dy_ref = matched_pairs[j][5] - matched_pairs[i][5]
            rpe_errs.append(math.hypot(dx_est - dx_ref, dy_est - dy_ref))

    rpe_rmse = float(np.sqrt(np.mean(np.square(rpe_errs)))) if rpe_errs else None

    return {
        "matched_frames": int(len(arr_xy)),
        "xy_rmse_m": float(np.sqrt(np.mean(np.square(arr_xy)))),
        "xy_mean_m": float(np.mean(arr_xy)),
        "xy_std_m": float(np.std(arr_xy)),
        "xy_median_m": float(np.median(arr_xy)),
        "xy_p95_m": float(np.percentile(arr_xy, 95)),
        "xy_max_m": float(np.max(arr_xy)),
        "yaw_rmse_rad": float(np.sqrt(np.mean(np.square(arr_yaw)))),
        "rpe_trans_rmse_m": rpe_rmse,
        "frames_gt_005m": int(np.sum(arr_xy > 0.05)),
        "frames_gt_010m": int(np.sum(arr_xy > 0.10)),
        "frames_gt_015m": int(np.sum(arr_xy > 0.15)),
    }


def main():
    parser = argparse.ArgumentParser(description="Apache-2.0 Trajectory Exporter and ATE/RPE Evaluator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("export", help="Export ROS 2 bag trajectory to TUM .txt")
    exp.add_argument("--bag", required=True, help="ROS 2 bag path")
    exp.add_argument("--topic", default="/platform_constraint", help="/platform_constraint or /tf")
    exp.add_argument("--out-tum", required=True, help="Output TUM file path")

    ev = sub.add_parser("evaluate", help="Evaluate estimated TUM file vs reference TUM file")
    ev.add_argument("--est", required=True, help="Estimated TUM trajectory")
    ev.add_argument("--ref", required=True, help="Reference TUM trajectory")
    ev.add_argument("--max-dt", type=float, default=0.05, help="Max timestamp match delta [s]")
    ev.add_argument("--rpe-delta", type=float, default=1.0, help="RPE window [s]")
    ev.add_argument("--out-json", default="", help="Optional JSON metrics path")

    args = parser.parse_args()
    if args.cmd == "export":
        bag_p = Path(args.bag).expanduser().resolve()
        if args.topic == "/tf":
            poses_3d = load_tf_poses_from_bag(bag_p)
        else:
            poses_3d = load_reference_from_bag(bag_p, ref_topic=args.topic)["poses_3d"]
        save_tum_file(poses_3d, Path(args.out_tum))
        print(f"Exported {len(poses_3d)} poses to {args.out_tum}")
    elif args.cmd == "evaluate":
        est = load_tum_file(Path(args.est))
        ref = load_tum_file(Path(args.ref))
        metrics = compute_ate_rpe_metrics(est, ref, max_dt_s=args.max_dt, rpe_delta_s=args.rpe_delta)
        print(json.dumps(metrics, indent=2))
        if args.out_json:
            Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_json, "w") as f:
                json.dump(metrics, f, indent=2)
        if metrics["matched_frames"] == 0:
            sys.exit(2)


if __name__ == "__main__":
    main()

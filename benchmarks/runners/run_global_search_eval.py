#!/usr/bin/env python3
# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0

"""
Reproducible Global Search Recovery & Safety Rejection Evaluator.

Supports both uncompressed and zstd-compressed ROS 2 bags via ros_eval_common,
declarative scenario YAML manifests, isolated localhost DDS execution,
and structured pass/fail validation output.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import rclpy
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ros_eval_common import (
    EvalListener,
    load_reference_from_bag,
    lookup_ref_by_offset,
    lookup_ref_by_stamp,
    resolve_localization_paths,
    terminate_process_group,
    wrap_angle,
)


def run_scenario(sc, bag_path: Path, listener: EvalListener, ref_stamps, ref_poses, args):
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(args.domain_id)
    if not args.no_localhost_only:
        env["ROS_LOCALHOST_ONLY"] = "1"

    x0, y0, yaw0 = lookup_ref_by_offset(ref_poses, sc["offset"])
    init_x = sc.get("init_x", x0)
    init_y = sc.get("init_y", y0)
    init_yaw = wrap_angle(sc.get("init_yaw_override", yaw0 + sc.get("yaw_perturb", 0.0)))

    node_bin, _, default_params = resolve_localization_paths()
    params_path = args.params_file or default_params

    launch_cmd = [
        node_bin,
        "--ros-args",
        "-r", "__node:=gn10_pointcloud_localization_node",
        "--params-file", params_path,
        "-p", "use_sim_time:=true",
        "-p", f"lidar_2d.enable:={'true' if sc['lidar_2d'] else 'false'}",
        "-p", f"initial_pose.use_for_local_search:={'true' if sc['initial_local_search'] else 'false'}",
        "-p", f"initial_pose.x:={init_x:.4f}",
        "-p", f"initial_pose.y:={init_y:.4f}",
        "-p", f"initial_pose.yaw:={init_yaw:.4f}",
    ]
    if args.map_file:
        launch_cmd.extend(["-p", f"map_file_path:={args.map_file}"])

    log_f = tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False)
    node_proc = subprocess.Popen(
        launch_cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    bag_proc = None
    poses = []
    telems = []
    try:
        ready = False
        t_wait_start = time.time()
        while time.time() - t_wait_start < 12.0:
            log_f.flush()
            with open(log_f.name, "r") as rf:
                txt = rf.read()
            if "Publishing localization telemetry on:" in txt:
                ready = True
                break
            if node_proc.poll() is not None:
                break
            time.sleep(0.05)

        if not ready:
            return {"name": sc["name"], "group": sc["group"], "passed": False, "error": "node_init_failed"}

        listener.reset()
        listener.publish_static_tf()
        time.sleep(0.15)

        rate = float(sc.get("rate", args.rate))
        bag_cmd = [
            "ros2", "bag", "play", str(bag_path),
            "--clock",
            "-d", "0.5",
            "--rate", str(rate),
            "--start-offset", str(sc["offset"]),
            "--topics", "/livox/lidar", "/livox/imu", "/lakibeam/scan",
        ]
        bag_proc = subprocess.Popen(
            bag_cmd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        max_wall = sc["bag_duration"] / rate + 15.0
        t0 = time.time()
        while time.time() - t0 < max_wall:
            if listener.sim_elapsed() >= sc["bag_duration"]:
                break
            time.sleep(0.05)

        time.sleep(0.1)
        poses, _, telems = listener.stop_recording()
    finally:
        terminate_process_group(bag_proc)
        terminate_process_group(node_proc)
        log_f.close()

    with open(log_f.name, "r") as rf:
        log_text = rf.read()
    os.unlink(log_f.name)

    gs_attempts = len(re.findall(r"\[GlobalSearch\] Starting GPU-Batched", log_text))
    gs_rejected_uniqueness = len(re.findall(r"uniqueness|borderline", log_text, re.IGNORECASE))
    trial_verified = len(re.findall(r"Trial verified", log_text))
    trial_rejected = len(re.findall(r"Trial rejected", log_text))

    pose_errors = []
    for stamp_ns, x, y, yaw in poses:
        (rx, ry, ryaw, _), dt = lookup_ref_by_stamp(ref_stamps, ref_poses, stamp_ns)
        if dt < 0.15:
            err_xy = math.hypot(x - rx, y - ry)
            err_yaw = abs(wrap_angle(yaw - ryaw))
            pose_errors.append((stamp_ns, err_xy, err_yaw))

    gs_rec_telem = [t for t in telems if t["global_recovered"]]
    tracked_frames = [t for t in telems if (not t["is_lost"]) and t["accepted"]]

    final_err_xy = pose_errors[-1][1] if pose_errors else None
    final_err_yaw = pose_errors[-1][2] if pose_errors else None
    min_err_xy = min((e[1] for e in pose_errors), default=None)

    recovered_accurately = bool(
        pose_errors and pose_errors[-1][1] < 0.35 and pose_errors[-1][2] < 0.25
    )
    false_jumps = sum(1 for e in pose_errors if e[1] > 1.0)

    expected = sc.get("expected", "recover")
    if expected == "reject":
        passed = (len(telems) > 0) and (len(poses) == 0) and (false_jumps == 0)
    else:
        passed = (len(telems) > 0) and recovered_accurately and (false_jumps == 0)

    return {
        "name": sc["name"],
        "group": sc["group"],
        "expected": expected,
        "passed": passed,
        "offset": sc["offset"],
        "lidar_2d": sc["lidar_2d"],
        "telem_count": len(telems),
        "pose_count": len(poses),
        "gs_attempts": gs_attempts,
        "gs_recovered_telem": len(gs_rec_telem),
        "gs_rejected_uniqueness": gs_rejected_uniqueness,
        "trial_verified": trial_verified,
        "trial_rejected": trial_rejected,
        "recovered_accurately": recovered_accurately,
        "final_err_xy": round(final_err_xy, 4) if final_err_xy is not None else None,
        "final_err_yaw": round(final_err_yaw, 4) if final_err_yaw is not None else None,
        "min_err_xy": round(min_err_xy, 4) if min_err_xy is not None else None,
        "false_jump_poses": false_jumps,
        "tracked_frames": len(tracked_frames),
    }


def main():
    default_manifest = Path(__file__).resolve().parents[1] / "manifests" / "n21_scenarios.yaml"

    parser = argparse.ArgumentParser(
        description="Run reproducible Global Search & Out-of-Map Safety evaluation"
    )
    parser.add_argument("--bag", required=True, help="Path to ROS 2 bag directory")
    parser.add_argument("--manifest", default=str(default_manifest), help="Scenario YAML manifest")
    parser.add_argument("--out-json", required=True, help="Output JSON results file")
    parser.add_argument("--groups", nargs="*", default=None, help="Filter scenario groups")
    parser.add_argument("--limit", type=int, default=0, help="Run at most N scenarios")
    parser.add_argument("--rate", type=float, default=4.0, help="Default rosbag playback rate")
    parser.add_argument("--domain-id", default="77", help="Isolated ROS_DOMAIN_ID")
    parser.add_argument("--no-localhost-only", action="store_true", help="Disable ROS_LOCALHOST_ONLY=1")
    parser.add_argument("--params-file", default="", help="Optional parameter YAML override")
    parser.add_argument("--map-file", default="", help="Optional map path override")
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.exists():
        print(f"Error: bag not found: {bag_path}", file=sys.stderr)
        sys.exit(1)

    with open(args.manifest, "r") as mf:
        scenarios = yaml.safe_load(mf)["scenarios"]

    if args.groups:
        allowed = set(args.groups)
        scenarios = [s for s in scenarios if s["group"] in allowed]

    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    if not args.no_localhost_only:
        os.environ["ROS_LOCALHOST_ONLY"] = "1"

    bag_data = load_reference_from_bag(bag_path)
    bag_duration_s = bag_data["bag_duration_s"]
    ref_stamps = bag_data["stamps"]
    ref_poses = bag_data["poses"]
    tf_static_msgs = bag_data["tf_static_msgs"]

    executable_scenarios = []
    skipped = []
    for s in scenarios:
        if s["offset"] + s["bag_duration"] > bag_duration_s + 0.5:
            skipped.append({"name": s["name"], "reason": "exceeds_bag_duration"})
        else:
            executable_scenarios.append(s)

    if args.limit > 0:
        executable_scenarios = executable_scenarios[:args.limit]

    if not executable_scenarios:
        print("Error: no executable scenarios fit the selected bag duration/filters.", file=sys.stderr)
        sys.exit(1)

    rclpy.init()
    listener = EvalListener(tf_static_msgs)
    stop_spin = threading.Event()

    def spin_loop():
        while not stop_spin.is_set() and rclpy.ok():
            rclpy.spin_once(listener, timeout_sec=0.02)

    spin_thread = threading.Thread(target=spin_loop, daemon=True)
    spin_thread.start()

    results = []
    try:
        for idx, sc in enumerate(executable_scenarios, 1):
            print(f"[{idx}/{len(executable_scenarios)}] Running {sc['name']} ...", flush=True)
            res = run_scenario(sc, bag_path, listener, ref_stamps, ref_poses, args)
            if res.get("telem_count", 0) == 0:
                print("   -> 0 telems, retrying once...", flush=True)
                res = run_scenario(sc, bag_path, listener, ref_stamps, ref_poses, args)
            print("   ->", res, flush=True)
            results.append(res)
    finally:
        stop_spin.set()
        spin_thread.join(timeout=1.0)
        listener.destroy_node()
        rclpy.shutdown()

    passed_count = sum(1 for r in results if r.get("passed", False))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bag_path": str(bag_path),
        "bag_duration_s": round(bag_duration_s, 3),
        "summary": {
            "executed": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "skipped": skipped,
        },
        "scenarios": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results ({passed_count}/{len(results)} passed) to {out_path}")

    if passed_count != len(results):
        sys.exit(2)


if __name__ == "__main__":
    main()

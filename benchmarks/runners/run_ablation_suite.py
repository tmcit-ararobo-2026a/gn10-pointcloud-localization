#!/usr/bin/env python3
# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0

"""
Reproducible Multi-Dataset & Ablation Benchmark Runner.

Executes configured ablation suites across ROS 2 bags in isolated localhost DDS,
collects BenchmarkProfiler latency CSVs, SystemGPUMonitor (nvidia-smi / tegrastats)
resource telemetry, TUM trajectories, ATE/RPE metrics, and generates Markdown + LaTeX tables.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rclpy
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "analysis"))

from benchmark_harness import SystemGPUMonitor
from evaluate_trajectories import compute_ate_rpe_metrics, save_tum_file
from ros_eval_common import (
    EvalListener,
    build_static_tf_commands,
    load_reference_from_bag,
    lookup_ref_by_offset,
    resolve_localization_paths,
    terminate_process_group,
)


def query_platform_info():
    info = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "nvpmodel": None,
    }
    if shutil.which("nvpmodel"):
        try:
            out = subprocess.check_output(["nvpmodel", "-q"], stderr=subprocess.DEVNULL, timeout=1.0).decode()
            info["nvpmodel"] = " | ".join(line.strip() for line in out.splitlines() if line.strip())
        except Exception:
            pass
    return info


def ros_param_str(val):
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def run_single_ablation(
    suite_name: str,
    suite_cfg: dict,
    run_cfg: dict,
    dataset_cfg: dict,
    bag_path: Path,
    bag_data: dict,
    listener: EvalListener,
    out_dir: Path,
    args,
):
    label = f"{suite_name}__{run_cfg['label']}"
    pipeline_csv = out_dir / f"{label}_pipeline.csv"
    resource_csv = out_dir / f"{label}_resources.csv"
    est_tum_path = out_dir / f"{label}_est_tum.txt"
    ref_tum_path = out_dir / f"{label}_ref_tum.txt"

    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(args.domain_id)
    if not args.no_localhost_only:
        env["ROS_LOCALHOST_ONLY"] = "1"

    node_bin, _, default_params = resolve_localization_paths()
    params_path = args.params_file or default_params

    offset_s = float(suite_cfg.get("offset_s", 0.0))
    duration_s = float(args.duration_override if args.duration_override > 0 else suite_cfg.get("duration_s", 30.0))
    rate = float(args.rate if args.rate > 0 else suite_cfg.get("rate", 1.0))

    x0, y0, yaw0 = lookup_ref_by_offset(bag_data["poses"], offset_s)

    launch_cmd = [
        node_bin,
        "--ros-args",
        "-r", "__node:=gn10_pointcloud_localization_node",
        "--params-file", params_path,
        "-p", "use_sim_time:=true",
        "-p", "benchmark.enable:=true",
        "-p", f"benchmark.csv_output_path:={pipeline_csv.resolve()}",
        "-p", f"initial_pose.use_for_local_search:={'true' if suite_cfg.get('initial_local_search', True) else 'false'}",
        "-p", f"initial_pose.x:={x0:.4f}",
        "-p", f"initial_pose.y:={y0:.4f}",
        "-p", f"initial_pose.yaw:={yaw0:.4f}",
    ]
    map_file = run_cfg.get("map_file") or suite_cfg.get("map_file") or dataset_cfg.get("map_file")
    map_type = run_cfg.get("map_type") or suite_cfg.get("map_type") or dataset_cfg.get("map_type")
    if map_file:
        mpath = Path(str(map_file)).expanduser()
        if not mpath.is_absolute() and (REPO_ROOT / mpath).exists():
            mpath = (REPO_ROOT / mpath).resolve()
        launch_cmd.extend(["-p", f"map_file_path:={mpath}"])
    if map_type:
        launch_cmd.extend(["-p", f"map_source_type:={map_type}"])
    for src_topic, dst_topic in (dataset_cfg.get("remaps") or {}).items():
        launch_cmd.extend(["-r", f"{src_topic}:={dst_topic}"])
    for k, v in (dataset_cfg.get("params") or {}).items():
        launch_cmd.extend(["-p", f"{k}:={ros_param_str(v)}"])
    for k, v in (run_cfg.get("params") or {}).items():
        launch_cmd.extend(["-p", f"{k}:={ros_param_str(v)}"])

    monitor = SystemGPUMonitor(sample_interval_s=0.10)
    monitor.start()

    tf_procs = []
    listener.set_static_tf(bag_data["tf_static_msgs"])
    if not bag_data["tf_static_msgs"]:
        for cmd in build_static_tf_commands():
            tf_procs.append(
                subprocess.Popen(
                    cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
                )
            )

    log_f = tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False)
    node_proc = subprocess.Popen(
        launch_cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    bag_proc = None
    poses_3d = []
    telems = []
    try:
        ready = False
        t_wait = time.time()
        while time.time() - t_wait < 15.0:
            log_f.flush()
            with open(log_f.name, "r") as rf:
                if "Publishing localization telemetry on:" in rf.read():
                    ready = True
                    break
            if node_proc.poll() is not None:
                break
            time.sleep(0.05)

        if not ready:
            return {"suite": suite_name, "label": run_cfg["label"], "passed": False, "error": "node_init_failed"}

        listener.reset()
        listener.publish_static_tf()
        time.sleep(0.15)

        play_topics = dataset_cfg.get("play_topics") or ["/livox/lidar", "/livox/imu", "/lakibeam/scan"]
        bag_cmd = [
            "ros2", "bag", "play", str(bag_path),
            "--clock",
            "-d", "0.5",
            "--rate", str(rate),
            "--start-offset", str(offset_s),
            "--topics", *play_topics,
        ]
        bag_proc = subprocess.Popen(
            bag_cmd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        max_wall = duration_s / rate + 20.0
        t0 = time.time()
        while time.time() - t0 < max_wall:
            if listener.sim_elapsed() >= duration_s:
                break
            if bag_proc.poll() is not None and listener.sim_elapsed() > 0.5:
                break
            time.sleep(0.05)

        time.sleep(0.1)
        _, poses_3d, telems = listener.stop_recording()
    finally:
        monitor.stop()
        terminate_process_group(bag_proc)
        terminate_process_group(node_proc)
        for p in tf_procs:
            terminate_process_group(p)
        log_f.close()
        if os.path.exists(log_f.name):
            os.unlink(log_f.name)

    df_res = monitor.to_dataframe()
    df_res.to_csv(resource_csv, index=False)
    df_pipe = pd.read_csv(pipeline_csv) if pipeline_csv.exists() and pipeline_csv.stat().st_size > 0 else pd.DataFrame()

    save_tum_file(poses_3d, est_tum_path)
    if not ref_tum_path.exists():
        save_tum_file(bag_data["poses_3d"], ref_tum_path)

    traj_metrics = compute_ate_rpe_metrics(poses_3d, bag_data["poses_3d"])

    # Compute steady-state metrics after initial 2.0 s warmup regardless of LiDAR rate (10 Hz vs 50 Hz)
    if len(telems) >= 2 and (telems[-1]["stamp_ns"] - telems[0]["stamp_ns"]) > 5 * 10**9:
        t_warm_ns = telems[0]["stamp_ns"] + 2 * 10**9
        steady_telem = [t for t in telems if t["stamp_ns"] >= t_warm_ns]
    else:
        steady_telem = telems
    c2d_vals = [t["cost_2d"] for t in steady_telem if t.get("obs_2d", 0) >= 30 and np.isfinite(t["cost_2d"])]
    c3d_vals = [t["cost_3d"] for t in steady_telem if t.get("obs_3d", 0) >= 50 and np.isfinite(t["cost_3d"])]
    inlier_vals = [t["inlier_cost"] for t in steady_telem if t.get("inlier_count", 0) > 0 and np.isfinite(t["inlier_cost"])]
    obs3d_vals = [t["obs_3d"] for t in steady_telem]
    lost_after_warmup = sum(1 for t in steady_telem if t.get("is_lost", False))

    def _mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else None

    def _p95(arr):
        return float(np.percentile(arr, 95)) if len(arr) > 0 else None

    pipe_total = df_pipe["time_total_ms"].dropna().to_numpy() if "time_total_ms" in df_pipe else np.array([])
    pipe_match = df_pipe["time_matching_ms"].dropna().to_numpy() if "time_matching_ms" in df_pipe else np.array([])
    gpu_power = df_res["gpu_power_w"].dropna().to_numpy() if "gpu_power_w" in df_res else np.array([])
    gpu_util = df_res["gpu_util_pct"].dropna().to_numpy() if "gpu_util_pct" in df_res else np.array([])
    cpu_proc = df_res["process_cpu_pct"].dropna().to_numpy() if "process_cpu_pct" in df_res else np.array([])
    rss_mb = df_res["process_rss_mb"].dropna().to_numpy() if "process_rss_mb" in df_res else np.array([])

    mean_power_w = _mean(gpu_power) if len(gpu_power) > 0 else None
    mean_total_ms = float(np.mean(pipe_total)) if len(pipe_total) > 0 else None

    passed = (len(df_pipe) > 0) and (traj_metrics["matched_frames"] > 0)

    return {
        "suite": suite_name,
        "label": run_cfg["label"],
        "dataset": suite_cfg["dataset"],
        "bag_path": str(bag_path),
        "offset_s": offset_s,
        "duration_s": duration_s,
        "passed": passed,
        "frames_profiled": int(len(df_pipe)),
        "telem_count": int(len(telems)),
        "lost_after_warmup": int(lost_after_warmup),
        "gpu_backend": monitor.backend,
        "latency_total_mean_ms": mean_total_ms,
        "latency_total_p95_ms": float(np.percentile(pipe_total, 95)) if len(pipe_total) > 0 else None,
        "latency_match_mean_ms": float(np.mean(pipe_match)) if len(pipe_match) > 0 else None,
        "latency_match_p95_ms": float(np.percentile(pipe_match, 95)) if len(pipe_match) > 0 else None,
        "obs_3d_mean": _mean(obs3d_vals),
        "cost_3d_mean_m": _mean(c3d_vals),
        "cost_2d_mean_m": _mean(c2d_vals),
        "inlier_cost_mean_m": _mean(inlier_vals),
        "cpu_node_mean_pct": float(np.mean(cpu_proc)) if len(cpu_proc) > 0 else None,
        "ram_rss_max_mb": float(np.max(rss_mb)) if len(rss_mb) > 0 else None,
        "gpu_util_mean_pct": _mean(gpu_util) if len(gpu_util) > 0 else None,
        "board_power_mean_w": mean_power_w,
        "energy_per_frame_mj": (mean_power_w * mean_total_ms) if (mean_power_w and mean_total_ms) else None,
        **traj_metrics,
    }


def write_summary_tables(rows, out_dir: Path):
    def fmt(v, digits=3):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{v:.{digits}f}"

    md_lines = [
        "# Reproducible Ablation & Multi-Dataset Summary",
        "",
        "| Suite | Run | Frames | Match Mean [ms] | Total Mean / p95 [ms] | XY Mean / p95 / Max [m] | >0.10m | Cost 2D Mean [m] | Power [W] |",
        "| :--- | :--- | ---: | ---: | :---: | :---: | ---: | ---: | ---: |",
    ]
    tex_lines = [
        "% Auto-generated reproducible ablation summary table (Apache-2.0)",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Ablation and Multi-Dataset Localization Benchmark Summary}",
        "\\label{tab:ablation_summary}",
        "\\begin{tabular}{llrrrrrrr}",
        "\\hline",
        "Suite & Configuration & Frames & Match [ms] & Total Mean/p95 [ms] & XY Mean/p95/Max [m] & $>0.10$m & $c_{\\text{2D}}$ [m] & Power [W] \\\\",
        "\\hline",
    ]

    for r in rows:
        tot_str = f"{fmt(r['latency_total_mean_ms'], 2)} / {fmt(r['latency_total_p95_ms'], 2)}"
        xy_str = f"{fmt(r['xy_mean_m'], 4)} / {fmt(r['xy_p95_m'], 4)} / {fmt(r['xy_max_m'], 4)}"
        md_lines.append(
            f"| {r['suite']} | {r['label']} | {r['frames_profiled']} | {fmt(r['latency_match_mean_ms'], 3)} | "
            f"{tot_str} | {xy_str} | {r['frames_gt_010m']} | {fmt(r['cost_2d_mean_m'], 4)} | {fmt(r['board_power_mean_w'], 2)} |"
        )
        s_esc = r["suite"].replace("_", "\\_")
        l_esc = r["label"].replace("_", "\\_")
        tex_lines.append(
            f"{s_esc} & {l_esc} & {r['frames_profiled']} & {fmt(r['latency_match_mean_ms'], 3)} & "
            f"{tot_str} & {xy_str} & {r['frames_gt_010m']} & {fmt(r['cost_2d_mean_m'], 4)} & {fmt(r['board_power_mean_w'], 2)} \\\\"
        )

    tex_lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}"])
    (out_dir / "ablation_summary.md").write_text("\n".join(md_lines) + "\n")
    (out_dir / "ablation_summary.tex").write_text("\n".join(tex_lines) + "\n")


def main():
    manifests_dir = Path(__file__).resolve().parents[1] / "manifests"
    parser = argparse.ArgumentParser(description="Run reproducible ablation suites across datasets")
    parser.add_argument("--matrix", default=str(manifests_dir / "ablation_matrix.yaml"))
    parser.add_argument("--datasets", default=str(manifests_dir / "datasets.yaml"))
    parser.add_argument("--suites", nargs="*", default=None, help="Filter specific suite names")
    parser.add_argument("--outdir", required=True, help="Output directory for CSV/JSON/TUM/TeX")
    parser.add_argument("--bag-override", default="", help="Override bag path for all selected suites")
    parser.add_argument("--duration-override", type=float, default=0.0, help="Override run duration [s] for quick smoke checks")
    parser.add_argument("--rate", type=float, default=0.0, help="Override bag playback rate")
    parser.add_argument("--domain-id", default="77")
    parser.add_argument("--no-localhost-only", action="store_true")
    parser.add_argument("--params-file", default="")
    args = parser.parse_args()

    out_dir = Path(args.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.matrix, "r") as f:
        suites = yaml.safe_load(f)["suites"]
    with open(args.datasets, "r") as f:
        datasets = yaml.safe_load(f)["datasets"]

    if args.suites:
        suites = {k: v for k, v in suites.items() if k in set(args.suites)}
    if not suites:
        print("Error: no matching suites selected.", file=sys.stderr)
        sys.exit(1)

    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    if not args.no_localhost_only:
        os.environ["ROS_LOCALHOST_ONLY"] = "1"

    bag_cache = {}
    for sname, scfg in suites.items():
        dkey = scfg["dataset"]
        dcfg = datasets[dkey]
        bpath_str = args.bag_override or dcfg["default_bag_path"]
        ref_topic = dcfg.get("ref_topic", "/platform_constraint")
        bpath = Path(bpath_str).expanduser().resolve()
        if not bpath.exists():
            print(f"Error: bag not found for suite '{sname}': {bpath}", file=sys.stderr)
            sys.exit(1)
        cache_key = (str(bpath), ref_topic)
        if cache_key not in bag_cache:
            print(f"[*] Indexing reference trajectory ({ref_topic}) from {bpath} ...", flush=True)
            bag_cache[cache_key] = (bpath, load_reference_from_bag(bpath, ref_topic=ref_topic))

    # Merge any static TF transforms from loaded bags
    all_static = []
    for _, (_, bdata) in bag_cache.items():
        all_static.extend(bdata["tf_static_msgs"])

    rclpy.init()
    listener = EvalListener(all_static)
    stop_spin = threading.Event()

    def spin_loop():
        while not stop_spin.is_set() and rclpy.ok():
            rclpy.spin_once(listener, timeout_sec=0.02)

    spin_thread = threading.Thread(target=spin_loop, daemon=True)
    spin_thread.start()

    results = []
    try:
        for sname, scfg in suites.items():
            dcfg = datasets[scfg["dataset"]]
            bpath_str = args.bag_override or dcfg["default_bag_path"]
            ref_topic = dcfg.get("ref_topic", "/platform_constraint")
            cache_key = (str(Path(bpath_str).expanduser().resolve()), ref_topic)
            bpath, bdata = bag_cache[cache_key]
            for rcfg in scfg["runs"]:
                print(f"[*] Running {sname} :: {rcfg['label']} ...", flush=True)
                res = run_single_ablation(sname, scfg, rcfg, dcfg, bpath, bdata, listener, out_dir, args)
                print(f"    -> passed={res.get('passed')}, frames={res.get('frames_profiled')}, "
                      f"mean_ms={res.get('latency_total_mean_ms')}, xy_mean={res.get('xy_mean_m')}", flush=True)
                results.append(res)
    finally:
        stop_spin.set()
        spin_thread.join(timeout=1.0)
        listener.destroy_node()
        rclpy.shutdown()

    passed_all = all(r.get("passed", False) for r in results)
    summary_payload = {
        "platform": query_platform_info(),
        "validation": {
            "total_runs": len(results),
            "passed_runs": sum(1 for r in results if r.get("passed", False)),
            "all_passed": passed_all,
        },
        "runs": results,
    }
    (out_dir / "ablation_results.json").write_text(json.dumps(summary_payload, indent=2))
    write_summary_tables(results, out_dir)
    print(f"\n[+] Ablation complete ({summary_payload['validation']['passed_runs']}/{len(results)} passed) -> {out_dir}")

    if not passed_all:
        sys.exit(2)


if __name__ == "__main__":
    main()

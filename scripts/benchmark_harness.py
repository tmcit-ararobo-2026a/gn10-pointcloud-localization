#!/usr/bin/env python3
# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0

"""
Autonomous System & Pipeline Benchmark Harness for gn10-pointcloud-localization.
Evaluates:
  - Pipeline Latency & Stage Breakdown (μs/ms resolution)
  - CPU Utilization (Process % & System %) and Host RAM (RSS/VMS)
  - GPU Utilization (Compute %), VRAM (Allocated/Peak), Power (W), Temp (C)
  - Communication Bandwidth (PCIe D2H Transfer & ROS 2 Topic Throughput)
Outputs:
  - Raw Metrics CSV files
  - Publication-ready High-Resolution Visualizations (PNG 300 DPI)
  - Statistical Markdown & LaTeX Tables
"""

import os
import sys
import time
import json
import subprocess
import signal
import threading
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import re
import shutil


class SystemGPUMonitor:
    def __init__(
        self,
        target_process_name="gn10_pointcloud_localization_node",
        sample_interval_s=0.05,
        target_pid=None,
    ):
        self.target_name = target_process_name
        self.interval = sample_interval_s
        self.running = False
        self.thread = None
        self.records = []
        self._pid_lock = threading.Lock()
        self._explicit_pid = target_pid
        self.target_pid = target_pid
        self._target_proc = None
        self._first_cpu_times = None
        self._last_cpu_times = None
        self._first_cpu_wall = None
        self._last_cpu_wall = None

        # Check GPU monitoring backend (prefer tegrastats on Jetson because JetPack 6 ships nvidia-smi returning N/A)
        self.backend = "none"
        if shutil.which("tegrastats") or os.path.exists("/usr/bin/tegrastats"):
            self.backend = "tegrastats"
            self.tegrastats_bin = shutil.which("tegrastats") or "/usr/bin/tegrastats"
        elif shutil.which("nvidia-smi"):
            try:
                subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, timeout=1.0)
                self.backend = "nvidia-smi"
            except Exception:
                pass

        self._tegra_lock = threading.Lock()
        self._tegra_proc = None
        self._tegra_thread = None
        self._latest_tegra = {
            'gpu_util_pct': np.nan,
            'gpu_mem_used_mb': np.nan,
            'gpu_mem_total_mb': np.nan,
            'gpu_power_w': np.nan,
            'gpu_temp_c': np.nan,
            'cpu_gpu_power_w': np.nan,
            'emc_util_pct': np.nan,
        }
        self._tegra_warned = False

    def set_target_pid(self, pid):
        with self._pid_lock:
            self._explicit_pid = pid
            self.target_pid = pid
            self._target_proc = None
            self._first_cpu_times = None
            self._last_cpu_times = None
            self._first_cpu_wall = None
            self._last_cpu_wall = None

    def get_cumulative_cpu_pct(self):
        with self._pid_lock:
            if (
                self._first_cpu_times is None
                or self._last_cpu_times is None
                or self._first_cpu_wall is None
                or self._last_cpu_wall is None
            ):
                return np.nan
            wall_dt = self._last_cpu_wall - self._first_cpu_wall
            if wall_dt <= 1e-6:
                return np.nan
            cpu_dt = (
                (self._last_cpu_times.user + self._last_cpu_times.system)
                - (self._first_cpu_times.user + self._first_cpu_times.system)
            )
            return float(max(0.0, cpu_dt / wall_dt * 100.0))

    def find_target_pid(self):
        if self._explicit_pid is not None:
            return self._explicit_pid if psutil.pid_exists(self._explicit_pid) else None
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline') or []
                cmd_str = " ".join(cmdline)
                if self.target_name in cmd_str or proc.info.get('name') == self.target_name:
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return None

    def query_gpu_metrics(self):
        if self.backend == "nvidia-smi":
            return self._query_nvidia_smi()
        elif self.backend == "tegrastats":
            return self._query_tegrastats()
        return {
            'gpu_util_pct': np.nan,
            'gpu_mem_used_mb': np.nan,
            'gpu_mem_total_mb': np.nan,
            'gpu_power_w': np.nan,
            'gpu_temp_c': np.nan,
        }

    def _query_nvidia_smi(self):
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
                '--format=csv,noheader,nounits'
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=0.5).decode().strip()
            raw_parts = [p.strip() for p in out.split(',')]
            def parse_val(v):
                if v == '[N/A]' or v == 'N/A' or not v:
                    return np.nan
                try:
                    return float(v)
                except ValueError:
                    return np.nan
            parts = [parse_val(p) for p in raw_parts]
            return {
                'gpu_util_pct': parts[0] if len(parts) > 0 else np.nan,
                'gpu_mem_used_mb': parts[1] if len(parts) > 1 else np.nan,
                'gpu_mem_total_mb': parts[2] if len(parts) > 2 else np.nan,
                'gpu_power_w': parts[3] if len(parts) > 3 else np.nan,
                'gpu_temp_c': parts[4] if len(parts) > 4 else np.nan,
            }
        except Exception:
            return {
                'gpu_util_pct': np.nan,
                'gpu_mem_used_mb': np.nan,
                'gpu_mem_total_mb': np.nan,
                'gpu_power_w': np.nan,
                'gpu_temp_c': np.nan,
            }

    @staticmethod
    def _parse_tegrastats_line(line):
        gpu_util = np.nan
        mem_used = np.nan
        mem_total = np.nan
        power_w = np.nan
        cpu_gpu_power_w = np.nan
        temp_c = np.nan
        emc_util = np.nan

        m_gpu = re.search(r'GR3D_FREQ\s+(\d+)%', line)
        if m_gpu:
            gpu_util = float(m_gpu.group(1))

        m_emc = re.search(r'EMC_FREQ\s+(\d+)%', line)
        if m_emc:
            emc_util = float(m_emc.group(1))

        m_ram = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
        if m_ram:
            mem_used = float(m_ram.group(1))
            mem_total = float(m_ram.group(2))

        # Parse total board power (VDD_IN / POM_5V_IN) in mW
        m_pow = re.search(r'(?:VDD_IN|POM_5V_IN)\s+(\d+)mW', line)
        if m_pow:
            power_w = float(m_pow.group(1)) / 1000.0

        # Parse Orin compute rail (VDD_CPU_GPU_CV) in mW when reported
        m_cg = re.search(r'VDD_CPU_GPU_CV\s+(\d+)mW', line)
        if m_cg:
            cpu_gpu_power_w = float(m_cg.group(1)) / 1000.0

        m_temp = re.search(r'gpu@([\d\.]+)C', line, re.IGNORECASE)
        if not m_temp:
            m_temp = re.search(r'tj@([\d\.]+)C', line, re.IGNORECASE)
        if m_temp:
            temp_c = float(m_temp.group(1))

        return {
            'gpu_util_pct': gpu_util,
            'gpu_mem_used_mb': mem_used,
            'gpu_mem_total_mb': mem_total,
            'gpu_power_w': power_w,
            'gpu_temp_c': temp_c,
            'cpu_gpu_power_w': cpu_gpu_power_w,
            'emc_util_pct': emc_util,
        }

    def _tegra_reader_loop(self):
        try:
            self._tegra_proc = subprocess.Popen(
                [self.tegrastats_bin, "--interval", "100"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            while self.running and self._tegra_proc.stdout:
                line = self._tegra_proc.stdout.readline()
                if not line:
                    if self._tegra_proc.poll() is not None:
                        break
                    continue
                parsed = self._parse_tegrastats_line(line)
                with self._tegra_lock:
                    self._latest_tegra = parsed
        except Exception as exc:
            if not self._tegra_warned:
                print(f"[!] Warning: tegrastats reader failed ({exc}); GPU metrics may be NaN.", file=sys.stderr)
                self._tegra_warned = True

    def _query_tegrastats(self):
        with self._tegra_lock:
            return dict(self._latest_tegra)

    def _monitor_loop(self):
        start_time = time.time()
        while self.running:
            now = time.time() - start_time

            # Locate target process PID and reuse Process instance so cpu_percent(interval=None) accumulates delta
            with self._pid_lock:
                explicit = self._explicit_pid
                if explicit is not None and self.target_pid != explicit:
                    self.target_pid = explicit
                    self._target_proc = None
                if self.target_pid is None or not psutil.pid_exists(self.target_pid):
                    self.target_pid = self.find_target_pid()
                    self._target_proc = None
                if self.target_pid and self._target_proc is None:
                    try:
                        proc = psutil.Process(self.target_pid)
                        proc.cpu_percent(interval=None)
                        self._target_proc = proc
                        now_mono = time.monotonic()
                        ct = proc.cpu_times()
                        if self._first_cpu_times is None:
                            self._first_cpu_times = ct
                            self._first_cpu_wall = now_mono
                        self._last_cpu_times = ct
                        self._last_cpu_wall = now_mono
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        self.target_pid = None
                        self._target_proc = None
                active_proc = self._target_proc

            proc_cpu = np.nan
            proc_rss_mb = np.nan
            proc_vms_mb = np.nan
            proc_threads = np.nan

            if active_proc is not None:
                try:
                    now_mono = time.monotonic()
                    proc_cpu = active_proc.cpu_percent(interval=None)
                    ct = active_proc.cpu_times()
                    with self._pid_lock:
                        if self._first_cpu_times is None:
                            self._first_cpu_times = ct
                            self._first_cpu_wall = now_mono
                        self._last_cpu_times = ct
                        self._last_cpu_wall = now_mono
                    mem_info = active_proc.memory_info()
                    proc_rss_mb = mem_info.rss / (1024.0 * 1024.0)
                    proc_vms_mb = mem_info.vms / (1024.0 * 1024.0)
                    proc_threads = active_proc.num_threads()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    with self._pid_lock:
                        if self._explicit_pid is None:
                            self.target_pid = None
                        self._target_proc = None

            sys_cpu_pct = psutil.cpu_percent(interval=None)
            sys_mem = psutil.virtual_memory()
            sys_mem_used_mb = (sys_mem.total - sys_mem.available) / (1024.0 * 1024.0)

            gpu_metrics = self.query_gpu_metrics()

            record = {
                'elapsed_sec': now,
                'process_cpu_pct': proc_cpu,
                'process_rss_mb': proc_rss_mb,
                'process_vms_mb': proc_vms_mb,
                'process_threads': proc_threads,
                'system_cpu_pct': sys_cpu_pct,
                'system_mem_used_mb': sys_mem_used_mb,
                **gpu_metrics
            }
            self.records.append(record)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.records.clear()
        if self.backend == "tegrastats":
            self._tegra_thread = threading.Thread(target=self._tegra_reader_loop, daemon=True)
            self._tegra_thread.start()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self._tegra_proc is not None:
            try:
                self._tegra_proc.terminate()
                self._tegra_proc.wait(timeout=1.0)
            except Exception:
                try:
                    self._tegra_proc.kill()
                except Exception:
                    pass
            self._tegra_proc = None
        if self._tegra_thread and self._tegra_thread.is_alive():
            self._tegra_thread.join(timeout=1.5)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def to_dataframe(self):
        return pd.DataFrame(self.records)


def run_benchmark_trial(
    bag_path,
    trial_duration_s=25.0,
    output_dir=Path("./benchmark_results"),
    trial_label="trial",
    domain_id="77",
    localhost_only=True,
    lidar_2d=None,
    extra_launch_args=None,
):
    bag_p = Path(bag_path).expanduser()
    if not bag_p.exists():
        raise FileNotFoundError(f"ROS 2 bag path does not exist: {bag_p}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_csv = output_dir / f"{trial_label}_pipeline_metrics.csv"
    resource_csv = output_dir / f"{trial_label}_resource_metrics.csv"
    validation_json = output_dir / f"{trial_label}_validation_report.json"

    print(f"\n========================================================")
    print(f"[*] Starting Benchmark Trial: {trial_label}")
    print(f"[*] Bag: {bag_p}")
    print(f"[*] Pipeline CSV: {pipeline_csv}")
    print(f"[*] Resource CSV: {resource_csv}")
    print(f"========================================================")

    # 1. Clean up any existing ROS 2 localization node processes
    subprocess.run(["pkill", "-9", "-f", "gn10_pointcloud_localization_node"], stderr=subprocess.DEVNULL)
    time.sleep(1.0)

    # 2. Initialize system & GPU resource monitor
    monitor = SystemGPUMonitor(sample_interval_s=0.05)
    monitor.start()

    # 3. Launch localization node with isolated local DDS and benchmark profiling enabled
    launch_cmd = [
        "ros2", "launch", "gn10_pointcloud_localization", "localization.launch.py",
        "use_sim_time:=true",
        "benchmark:=true",
        f"benchmark_csv:={pipeline_csv.resolve()}"
    ]
    if lidar_2d is not None:
        launch_cmd.append(f"lidar_2d:={'true' if lidar_2d else 'false'}")
    if extra_launch_args:
        launch_cmd.extend(extra_launch_args)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["ROS_DOMAIN_ID"] = str(domain_id)
    if localhost_only:
        env["ROS_LOCALHOST_ONLY"] = "1"

    launch_proc = subprocess.Popen(
        launch_cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(3.5)  # Wait for ESDF map initialization

    # 4. Play rosbag locally inside own process group with SIGKILL fallback so shutdown never hangs
    bag_cmd = [
        "ros2", "bag", "play", str(bag_p), "--clock", "--disable-keyboard-controls"
    ]
    print(f"[*] Playing rosbag for {trial_duration_s}s (ROS_LOCALHOST_ONLY={env.get('ROS_LOCALHOST_ONLY', '0')}, DOMAIN={domain_id})...")
    bag_proc = subprocess.Popen(
        bag_cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        bag_proc.wait(timeout=float(trial_duration_s))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(bag_proc.pid), signal.SIGINT)
            bag_proc.wait(timeout=2.0)
        except Exception:
            try:
                os.killpg(os.getpgid(bag_proc.pid), signal.SIGKILL)
            except Exception:
                pass

    # 5. Stop monitoring and terminate node processes cleanly
    print("[*] Benchmark run completed. Shutting down nodes...")
    monitor.stop()

    try:
        os.killpg(os.getpgid(launch_proc.pid), signal.SIGINT)
        launch_proc.wait(timeout=3.0)
    except Exception:
        try:
            os.killpg(os.getpgid(launch_proc.pid), signal.SIGKILL)
        except Exception:
            pass
    subprocess.run(["pkill", "-9", "-f", "gn10_pointcloud_localization_node"], stderr=subprocess.DEVNULL)

    # 6. Save resource metrics CSV
    df_res = monitor.to_dataframe()
    df_res.to_csv(resource_csv, index=False)
    print(f"[+] Saved system & GPU resource metrics: {len(df_res)} samples -> {resource_csv}")

    # 7. Verify pipeline profiling CSV & write validation report
    if pipeline_csv.exists() and pipeline_csv.stat().st_size > 0:
        df_pipe = pd.read_csv(pipeline_csv)
        print(f"[+] Recorded pipeline profiling frames: {len(df_pipe)} frames -> {pipeline_csv}")
    else:
        print(f"[!] Error: Pipeline CSV was empty or not generated.", file=sys.stderr)
        df_pipe = pd.DataFrame()

    valid_gpu = int(df_res['gpu_util_pct'].notna().sum()) if not df_res.empty and 'gpu_util_pct' in df_res else 0
    lost_frames = int((df_pipe['is_lost'] != 0).sum()) if not df_pipe.empty and 'is_lost' in df_pipe else 0
    validation = {
        "trial_label": trial_label,
        "bag_path": str(bag_p.resolve()),
        "duration_s": float(trial_duration_s),
        "gpu_monitor_backend": monitor.backend,
        "pipeline_frames": int(len(df_pipe)),
        "lost_frames": lost_frames,
        "resource_samples": int(len(df_res)),
        "valid_gpu_samples": valid_gpu,
        "passed": bool(len(df_pipe) > 0),
    }
    if valid_gpu == 0 and len(df_res) > 0:
        print(f"[!] Warning: 0 valid GPU telemetry samples collected (backend={monitor.backend}).", file=sys.stderr)
    with open(validation_json, "w") as vf:
        json.dump(validation, vf, indent=2)
    print(f"[+] Saved validation report -> {validation_json}")

    return df_pipe, df_res


def compute_statistics(series, name=""):
    s = series.dropna()
    if len(s) == 0:
        return {
            'name': name,
            'count': 0,
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'q25': np.nan,
            'median': np.nan,
            'q75': np.nan,
            'p95': np.nan,
            'p99': np.nan,
            'max': np.nan
        }
    return {
        'name': name,
        'count': int(len(s)),
        'mean': float(s.mean()),
        'std': float(s.std()) if len(s) > 1 else 0.0,
        'min': float(s.min()),
        'q25': float(s.quantile(0.25)),
        'median': float(s.median()),
        'q75': float(s.quantile(0.75)),
        'p95': float(s.quantile(0.95)),
        'p99': float(s.quantile(0.99)),
        'max': float(s.max())
    }


def generate_publication_visualizations(df_pipe, df_res, output_dir=Path("./benchmark_results"), prefix="eval"):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 1.8,
        'grid.alpha': 0.4,
    })

    # -------------------------------------------------------------
    # Fig 1: Pipeline Latency Breakdown & Target 60Hz Horizon
    # -------------------------------------------------------------
    if not df_pipe.empty and 'time_total_ms' in df_pipe.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [2.2, 1]})

        t = (df_pipe['timestamp_sec'] - df_pipe['timestamp_sec'].iloc[0]).to_numpy()
        ax1.plot(t, df_pipe['time_total_ms'].to_numpy(), label='Total Pipeline Latency', color='#1f77b4', lw=2.0)
        ax1.plot(t, df_pipe['time_matching_ms'].to_numpy(), label='3D ESDF Matching (GPU)', color='#ff7f0e', lw=1.5, alpha=0.85)
        ax1.plot(t, df_pipe['time_ground_filter_ms'].to_numpy(), label='Ground Filter (GPU)', color='#2ca02c', lw=1.2, alpha=0.85)
        ax1.plot(t, df_pipe['time_deskew_ms'].to_numpy(), label='IMU Deskew (CPU)', color='#9467bd', lw=1.2, alpha=0.8)

        # 60 Hz deadline (16.67 ms) and 30 Hz deadline (33.33 ms)
        ax1.axhline(16.67, color='red', linestyle='--', lw=1.5, label='60 Hz Threshold (16.6 ms)')
        ax1.set_xlabel('Elapsed Time [s]')
        ax1.set_ylabel('Execution Time [ms]')
        ax1.set_title('Pipeline Stage Latency Time-Series')
        ax1.grid(True)
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.set_ylim(0, max(25.0, float(df_pipe['time_total_ms'].quantile(0.99)) * 1.3))

        # Box plot distribution
        box_data = [
            df_pipe['time_total_ms'].dropna().to_numpy(),
            df_pipe['time_matching_ms'].dropna().to_numpy(),
            df_pipe['time_ground_filter_ms'].dropna().to_numpy(),
            df_pipe['time_deskew_ms'].dropna().to_numpy(),
            df_pipe['time_publish_ms'].dropna().to_numpy()
        ]
        box_labels = ['Total', 'Matching', 'GroundFilter', 'Deskew', 'Publish']
        bplot = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, showmeans=True,
                            meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black'})
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.axhline(16.67, color='red', linestyle='--', lw=1.5)
        ax2.set_ylabel('Execution Time [ms]')
        ax2.set_title('Stage Distribution')
        ax2.grid(True)
        plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')

        plt.tight_layout()
        fig_path = output_dir / f"{prefix}_latency_breakdown.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[+] Generated figure: {fig_path}")

    # -------------------------------------------------------------
    # Fig 2: Compute Hardware Resource Utilization (CPU & GPU)
    # -------------------------------------------------------------
    if not df_res.empty:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))

        t = df_res['elapsed_sec'].to_numpy()
        # (a) CPU Utilization
        axes[0, 0].plot(t, df_res['process_cpu_pct'].to_numpy(), label='Node Process CPU %', color='#1f77b4', lw=2)
        axes[0, 0].plot(t, df_res['system_cpu_pct'].to_numpy(), label='Overall System CPU %', color='#7f7f7f', lw=1.2, linestyle=':')
        axes[0, 0].set_ylabel('CPU Usage [%]')
        axes[0, 0].set_title('(a) CPU Utilization')
        axes[0, 0].grid(True)
        axes[0, 0].legend(loc='upper right')

        # (b) GPU Utilization
        axes[0, 1].plot(t, df_res['gpu_util_pct'].to_numpy(), label='GPU Core %', color='#2ca02c', lw=2)
        axes[0, 1].set_ylabel('GPU Compute Utilization [%]')
        axes[0, 1].set_title('(b) GPU Core Activity')
        axes[0, 1].grid(True)
        axes[0, 1].legend(loc='upper right')

        # (c) Host RAM vs GPU VRAM
        axes[1, 0].plot(t, df_res['process_rss_mb'].to_numpy(), label='Host RSS [MB]', color='#9467bd', lw=2)
        axes[1, 0].plot(t, df_res['gpu_mem_used_mb'].to_numpy(), label='GPU VRAM Used [MB]', color='#d62728', lw=2)
        axes[1, 0].set_xlabel('Elapsed Time [s]')
        axes[1, 0].set_ylabel('Memory [MB]')
        axes[1, 0].set_title('(c) Memory Footprint')
        axes[1, 0].grid(True)
        axes[1, 0].legend(loc='upper right')

        # (d) GPU Power Consumption
        axes[1, 1].plot(t, df_res['gpu_power_w'].to_numpy(), label='GPU Power [W]', color='#ff7f0e', lw=2)
        axes[1, 1].set_xlabel('Elapsed Time [s]')
        axes[1, 1].set_ylabel('Power [W]')
        axes[1, 1].set_title('(d) GPU Power Draw')
        axes[1, 1].grid(True)
        axes[1, 1].legend(loc='upper right')

        plt.tight_layout()
        fig_path = output_dir / f"{prefix}_hardware_resources.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[+] Generated figure: {fig_path}")

    # -------------------------------------------------------------
    # Fig 3: Latency Cumulative Distribution Function (CDF)
    # -------------------------------------------------------------
    if not df_pipe.empty and 'time_total_ms' in df_pipe.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sorted_data = np.sort(df_pipe['time_total_ms'].dropna())
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

        ax.plot(sorted_data, yvals * 100.0, lw=2.5, color='#1f77b4', label='Total Latency Empirical CDF')
        p50 = np.percentile(sorted_data, 50)
        p95 = np.percentile(sorted_data, 95)
        p99 = np.percentile(sorted_data, 99)

        ax.axvline(p50, color='#2ca02c', linestyle='--', label=f'Median (p50): {p50:.2f} ms')
        ax.axvline(p95, color='#ff7f0e', linestyle='--', label=f'95th Percentile: {p95:.2f} ms')
        ax.axvline(16.67, color='red', linestyle='-', lw=1.5, label='60 Hz Deadline (16.67 ms)')

        ax.set_xlabel('Pipeline Total Latency [ms]')
        ax.set_ylabel('Cumulative Probability [%]')
        ax.set_title('Cumulative Distribution Function (CDF) of Localization Latency')
        ax.grid(True)
        ax.legend(loc='lower right')
        ax.set_xlim(0, max(25.0, p99 * 1.25))
        ax.set_ylim(0, 105)

        plt.tight_layout()
        fig_path = output_dir / f"{prefix}_latency_cdf.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[+] Generated figure: {fig_path}")


def generate_markdown_and_latex_reports(df_pipe, df_res, output_dir=Path("./benchmark_results"), prefix="eval"):
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = []
    if not df_pipe.empty:
        metrics_list.append(compute_statistics(df_pipe['time_total_ms'], 'Total Latency [ms]'))
        metrics_list.append(compute_statistics(df_pipe['time_deskew_ms'], 'Deskew Latency [ms]'))
        metrics_list.append(compute_statistics(df_pipe['time_ground_filter_ms'], 'Ground Filter [ms]'))
        metrics_list.append(compute_statistics(df_pipe['time_matching_ms'], 'ESDF Matching [ms]'))
        metrics_list.append(compute_statistics(df_pipe['time_publish_ms'], 'Publish & TF [ms]'))
        if 'raw_points' in df_pipe:
            metrics_list.append(compute_statistics(df_pipe['raw_points'], 'Input Points / Frame'))
        if 'obstacle_points' in df_pipe:
            metrics_list.append(compute_statistics(df_pipe['obstacle_points'], 'Obstacle Points / Frame'))
        if 'dynamic_points' in df_pipe:
            metrics_list.append(compute_statistics(df_pipe['dynamic_points'], 'Dynamic Points / Frame'))
        if 'd2h_transfer_bytes' in df_pipe:
            metrics_list.append(compute_statistics(df_pipe['d2h_transfer_bytes'] / 1024.0, 'PCIe D2H Transfer [KB/frame]'))

    if not df_res.empty:
        metrics_list.append(compute_statistics(df_res['process_cpu_pct'], 'Node CPU [%]'))
        metrics_list.append(compute_statistics(df_res['system_cpu_pct'], 'System CPU [%]'))
        metrics_list.append(compute_statistics(df_res['process_rss_mb'], 'Host RAM RSS [MB]'))
        metrics_list.append(compute_statistics(df_res['gpu_util_pct'], 'GPU Core Util [%]'))
        metrics_list.append(compute_statistics(df_res['gpu_mem_used_mb'], 'GPU VRAM Used [MB]'))
        metrics_list.append(compute_statistics(df_res['gpu_power_w'], 'GPU Power [W]'))

    df_summary = pd.DataFrame(metrics_list)

    # 1. Markdown Table (custom formatter without external tabulate dependency)
    headers = ["Metric", "Mean", "Std", "Min", "Median", "p95", "p99", "Max"]
    col_w = [30, 10, 10, 10, 10, 10, 10, 10]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_w)) + " |"
    sep_line = "|-" + "-|-".join("-" * w for w in col_w) + "-|"
    rows = [header_line, sep_line]
    def fmt(v):
        if not np.isfinite(v):
            return "N/A"
        return f"{v:.2f}"

    for m in metrics_list:
        row = [
            m['name'].ljust(col_w[0]),
            fmt(m['mean']).rjust(col_w[1]),
            fmt(m['std']).rjust(col_w[2]),
            fmt(m['min']).rjust(col_w[3]),
            fmt(m['median']).rjust(col_w[4]),
            fmt(m['p95']).rjust(col_w[5]),
            fmt(m['p99']).rjust(col_w[6]),
            fmt(m['max']).rjust(col_w[7]),
        ]
        rows.append("| " + " | ".join(row) + " |")
    md_table = "\n".join(rows)

    md_path = output_dir / f"{prefix}_summary_table.md"
    with open(md_path, 'w') as f:
        f.write("# Quantitative Performance Evaluation\n\n")
        f.write(md_table)
        f.write("\n")
    print(f"[+] Saved Markdown summary table: {md_path}")

    # 2. LaTeX Table
    latex_path = output_dir / f"{prefix}_summary_table.tex"
    with open(latex_path, 'w') as f:
        f.write("% Auto-generated performance evaluation table for paper publication\n")
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{Quantitative Benchmark Results of Proposed 3D ESDF Localization System}\n")
        f.write("\\label{tab:system_performance}\n")
        f.write("\\begin{tabular}{lrrrrrrrr}\n\\hline\n")
        f.write("Metric & Mean & Std & Min & Median & p95 & p99 & Max \\\\\n\\hline\n")
        for m in metrics_list:
            f.write(f"{m['name']} & {fmt(m['mean'])} & {fmt(m['std'])} & {fmt(m['min'])} & {fmt(m['median'])} & {fmt(m['p95'])} & {fmt(m['p99'])} & {fmt(m['max'])} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table*}\n")
    print(f"[+] Saved LaTeX summary table: {latex_path}")

    return df_summary


def compare_trials(baseline_dir, proposed_dir, baseline_label="baseline_main",
                   proposed_label="proposed_pr3", output_dir=Path("./benchmark_results")):
    """
    Compare two benchmark trial results with statistical significance testing.
    Generates:
      - Side-by-side box plots for pipeline latency stages
      - Overlaid CDF curves
      - Δ% summary tables (Markdown + LaTeX)
      - Statistical significance via Welch's t-test and Mann-Whitney U
    """
    from scipy import stats as scipy_stats

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 1.8,
        'grid.alpha': 0.4,
    })

    # Load CSVs
    b_pipe_path = Path(baseline_dir) / f"{baseline_label}_pipeline_metrics.csv"
    b_res_path = Path(baseline_dir) / f"{baseline_label}_resource_metrics.csv"
    p_pipe_path = Path(proposed_dir) / f"{proposed_label}_pipeline_metrics.csv"
    p_res_path = Path(proposed_dir) / f"{proposed_label}_resource_metrics.csv"

    df_b_pipe = pd.read_csv(b_pipe_path) if b_pipe_path.exists() else pd.DataFrame()
    df_b_res = pd.read_csv(b_res_path) if b_res_path.exists() else pd.DataFrame()
    df_p_pipe = pd.read_csv(p_pipe_path) if p_pipe_path.exists() else pd.DataFrame()
    df_p_res = pd.read_csv(p_res_path) if p_res_path.exists() else pd.DataFrame()

    print(f"[*] Baseline pipeline frames: {len(df_b_pipe)}, resource samples: {len(df_b_res)}")
    print(f"[*] Proposed pipeline frames: {len(df_p_pipe)}, resource samples: {len(df_p_res)}")

    # ---- Metric pairs for statistical comparison ----
    pipeline_metrics = [
        ('time_total_ms', 'Total Latency [ms]'),
        ('time_deskew_ms', 'Deskew [ms]'),
        ('time_ground_filter_ms', 'Ground Filter [ms]'),
        ('time_matching_ms', 'ESDF Matching [ms]'),
        ('time_publish_ms', 'Publish & TF [ms]'),
    ]
    resource_metrics = [
        ('process_cpu_pct', 'Node CPU [%]'),
        ('system_cpu_pct', 'System CPU [%]'),
        ('process_rss_mb', 'Host RAM RSS [MB]'),
        ('gpu_util_pct', 'GPU Core Util [%]'),
        ('gpu_mem_used_mb', 'GPU VRAM Used [MB]'),
    ]

    def stat_test(a, b, name):
        a = a.dropna()
        b = b.dropna()
        if len(a) < 3 or len(b) < 3:
            return {'metric': name, 'baseline_mean': np.nan, 'proposed_mean': np.nan,
                    'delta_pct': np.nan, 'welch_t': np.nan, 'welch_p': np.nan,
                    'mann_whitney_U': np.nan, 'mann_whitney_p': np.nan, 'significant': 'N/A'}
        t_stat, t_p = scipy_stats.ttest_ind(a, b, equal_var=False)
        u_stat, u_p = scipy_stats.mannwhitneyu(a, b, alternative='two-sided')
        b_mean = a.mean()
        p_mean = b.mean()
        delta = ((p_mean - b_mean) / b_mean * 100) if b_mean != 0 else np.nan
        sig = '***' if t_p < 0.001 else ('**' if t_p < 0.01 else ('*' if t_p < 0.05 else 'n.s.'))
        return {
            'metric': name,
            'baseline_mean': float(b_mean),
            'baseline_std': float(a.std()),
            'proposed_mean': float(p_mean),
            'proposed_std': float(b.std()),
            'delta_pct': float(delta),
            'welch_t': float(t_stat),
            'welch_p': float(t_p),
            'mann_whitney_U': float(u_stat),
            'mann_whitney_p': float(u_p),
            'significant': sig,
        }

    results = []
    for col, label in pipeline_metrics:
        if col in df_b_pipe.columns and col in df_p_pipe.columns:
            results.append(stat_test(df_b_pipe[col], df_p_pipe[col], label))
    for col, label in resource_metrics:
        if col in df_b_res.columns and col in df_p_res.columns:
            results.append(stat_test(df_b_res[col], df_p_res[col], label))

    # ---- Fig: Side-by-Side Box Plots ----
    if not df_b_pipe.empty and not df_p_pipe.empty:
        stage_cols = ['time_total_ms', 'time_deskew_ms', 'time_ground_filter_ms',
                      'time_matching_ms', 'time_publish_ms']
        stage_labels = ['Total', 'Deskew', 'Ground\nFilter', 'ESDF\nMatching', 'Publish']
        available = [(c, l) for c, l in zip(stage_cols, stage_labels) if c in df_b_pipe.columns and c in df_p_pipe.columns]

        if available:
            fig, ax = plt.subplots(figsize=(12, 5.5))
            n = len(available)
            positions_b = np.arange(n) * 3
            positions_p = positions_b + 1

            bp1 = ax.boxplot([df_b_pipe[c].dropna().to_numpy() for c, _ in available],
                             positions=positions_b, widths=0.8, patch_artist=True, showmeans=True,
                             meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black'})
            bp2 = ax.boxplot([df_p_pipe[c].dropna().to_numpy() for c, _ in available],
                             positions=positions_p, widths=0.8, patch_artist=True, showmeans=True,
                             meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': 'black'})

            for patch in bp1['boxes']:
                patch.set_facecolor('#1f77b4')
                patch.set_alpha(0.7)
            for patch in bp2['boxes']:
                patch.set_facecolor('#ff7f0e')
                patch.set_alpha(0.7)

            ax.set_xticks(positions_b + 0.5)
            ax.set_xticklabels([l for _, l in available])
            ax.set_ylabel('Execution Time [ms]')
            ax.set_title('Pipeline Stage Latency Comparison: Baseline (main) vs. Proposed (PR3)')
            ax.axhline(16.67, color='red', linestyle='--', lw=1.5, label='60 Hz Deadline')
            ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Baseline (main)', 'Proposed (PR3)'],
                      loc='upper right')
            ax.grid(True)
            plt.tight_layout()
            fig_path = output_dir / "comparison_latency_boxplot.png"
            plt.savefig(fig_path, dpi=300)
            plt.close()
            print(f"[+] Generated: {fig_path}")

    # ---- Fig: Overlaid CDF ----
    if not df_b_pipe.empty and not df_p_pipe.empty and 'time_total_ms' in df_b_pipe.columns:
        fig, ax = plt.subplots(figsize=(8, 5))

        for df, color, label_name in [(df_b_pipe, '#1f77b4', 'Baseline (main)'),
                                       (df_p_pipe, '#ff7f0e', 'Proposed (PR3)')]:
            sorted_data = np.sort(df['time_total_ms'].dropna().to_numpy())
            if len(sorted_data) > 1:
                yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
                ax.plot(sorted_data, yvals * 100, lw=2.5, color=color, label=label_name)

        ax.axvline(16.67, color='red', linestyle='-', lw=1.5, label='60 Hz Deadline')
        ax.set_xlabel('Pipeline Total Latency [ms]')
        ax.set_ylabel('Cumulative Probability [%]')
        ax.set_title('CDF Comparison: Total Pipeline Latency')
        ax.grid(True)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)
        plt.tight_layout()
        fig_path = output_dir / "comparison_latency_cdf.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[+] Generated: {fig_path}")

    # ---- Fig: Resource Comparison (2x2) ----
    if not df_b_res.empty and not df_p_res.empty:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
        pairs = [
            ('process_cpu_pct', 'Node CPU Usage [%]', '(a) CPU Utilization'),
            ('gpu_util_pct', 'GPU Compute [%]', '(b) GPU Utilization'),
            ('process_rss_mb', 'Memory [MB]', '(c) Host RAM (RSS)'),
            ('gpu_mem_used_mb', 'VRAM [MB]', '(d) GPU VRAM Used'),
        ]
        for idx, (col, ylabel, title) in enumerate(pairs):
            ax = axes[idx // 2, idx % 2]
            for df, t_label, color, ls in [
                (df_b_res, 'Baseline', '#1f77b4', '-'),
                (df_p_res, 'Proposed', '#ff7f0e', '-'),
            ]:
                if col in df.columns:
                    t = df['elapsed_sec'].to_numpy()
                    vals = df[col].to_numpy()
                    ax.plot(t, vals, label=t_label, color=color, lw=1.5, linestyle=ls)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True)
            ax.legend(loc='upper right', fontsize=9)
            if idx >= 2:
                ax.set_xlabel('Elapsed Time [s]')

        plt.tight_layout()
        fig_path = output_dir / "comparison_hardware_resources.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[+] Generated: {fig_path}")

    # ---- Markdown & LaTeX Comparison Tables ----
    if results:
        # Markdown
        headers = ["Metric", "Baseline μ±σ", "Proposed μ±σ", "Δ%", "Welch p", "Sig."]
        col_w = [26, 18, 18, 10, 12, 6]
        header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_w)) + " |"
        sep_line = "|-" + "-|-".join("-" * w for w in col_w) + "-|"
        rows_md = [header_line, sep_line]
        for r in results:
            b_str = f"{r['baseline_mean']:.3f}±{r.get('baseline_std', 0):.3f}"
            p_str = f"{r['proposed_mean']:.3f}±{r.get('proposed_std', 0):.3f}"
            delta_str = f"{r['delta_pct']:+.1f}%" if np.isfinite(r['delta_pct']) else "N/A"
            p_val_str = f"{r['welch_p']:.4f}" if np.isfinite(r['welch_p']) else "N/A"
            row = [
                r['metric'].ljust(col_w[0]),
                b_str.rjust(col_w[1]),
                p_str.rjust(col_w[2]),
                delta_str.rjust(col_w[3]),
                p_val_str.rjust(col_w[4]),
                r['significant'].center(col_w[5]),
            ]
            rows_md.append("| " + " | ".join(row) + " |")
        md_table = "\n".join(rows_md)

        md_path = output_dir / "comparison_summary.md"
        with open(md_path, 'w') as f:
            f.write("# Comparative Performance Evaluation: Baseline (main) vs. Proposed (PR3)\n\n")
            f.write("## Statistical Significance\n")
            f.write("- \\*\\*\\* p < 0.001, \\*\\* p < 0.01, \\* p < 0.05, n.s. = not significant\n")
            f.write("- Tests: Welch's t-test (parametric) and Mann-Whitney U (non-parametric)\n\n")
            f.write(md_table)
            f.write("\n")
        print(f"[+] Saved comparison Markdown: {md_path}")

        # LaTeX
        tex_path = output_dir / "comparison_summary.tex"
        with open(tex_path, 'w') as f:
            f.write("% Auto-generated comparative performance evaluation table\n")
            f.write("\\begin{table*}[t]\n\\centering\n")
            f.write("\\caption{Comparative Benchmark: Baseline (main) vs.\\ Proposed System (PR3)}\n")
            f.write("\\label{tab:comparison}\n")
            f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
            f.write("Metric & Baseline ($\\mu \\pm \\sigma$) & Proposed ($\\mu \\pm \\sigma$) & $\\Delta$\\% & Welch $p$ & Sig. \\\\\n\\hline\n")
            for r in results:
                b_str = f"${r['baseline_mean']:.3f} \\pm {r.get('baseline_std', 0):.3f}$"
                p_str = f"${r['proposed_mean']:.3f} \\pm {r.get('proposed_std', 0):.3f}$"
                delta_str = f"{r['delta_pct']:+.1f}\\%" if np.isfinite(r['delta_pct']) else "N/A"
                p_val_str = f"{r['welch_p']:.4f}" if np.isfinite(r['welch_p']) else "N/A"
                metric_escaped = r['metric'].replace('%', '\\%').replace('&', '\\&')
                f.write(f"{metric_escaped} & {b_str} & {p_str} & {delta_str} & {p_val_str} & {r['significant']} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n\\end{table*}\n")
        print(f"[+] Saved comparison LaTeX: {tex_path}")

    df_results = pd.DataFrame(results)
    return df_results


def main():
    parser = argparse.ArgumentParser(description="Autonomous Benchmark Harness for gn10-pointcloud-localization")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # --- run: single trial ---
    run_parser = subparsers.add_parser("run", help="Run a single benchmark trial")
    run_parser.add_argument("--bag", type=str, required=True, help="Path to ROS 2 bag directory")
    run_parser.add_argument("--duration", type=float, default=25.0, help="Trial duration in seconds")
    run_parser.add_argument("--outdir", type=str, default="./benchmark_results", help="Output directory")
    run_parser.add_argument("--label", type=str, default="baseline", help="Trial label")
    run_parser.add_argument("--domain-id", type=str, default="77", help="Isolated ROS_DOMAIN_ID")
    run_parser.add_argument("--no-localhost-only", action="store_true", help="Disable ROS_LOCALHOST_ONLY=1")
    run_parser.add_argument("--lidar-2d", type=str, choices=["true", "false"], default=None, help="Override lidar_2d.enable")

    # --- compare: two trial results ---
    cmp_parser = subparsers.add_parser("compare", help="Compare two trial results")
    cmp_parser.add_argument("--baseline-dir", type=str, default="./benchmark_results", help="Directory of baseline CSVs")
    cmp_parser.add_argument("--proposed-dir", type=str, default="./benchmark_results", help="Directory of proposed CSVs")
    cmp_parser.add_argument("--baseline-label", type=str, default="baseline_main", help="Baseline label prefix")
    cmp_parser.add_argument("--proposed-label", type=str, default="proposed_pr3", help="Proposed label prefix")
    cmp_parser.add_argument("--outdir", type=str, default="./benchmark_results", help="Output directory")

    args = parser.parse_args()

    # Backward compat: if no subcommand, act as 'run' with legacy flags
    if args.command is None:
        parser2 = argparse.ArgumentParser()
        parser2.add_argument("--bag", type=str, required=True)
        parser2.add_argument("--duration", type=float, default=25.0)
        parser2.add_argument("--outdir", type=str, default="./benchmark_results")
        parser2.add_argument("--label", type=str, default="baseline")
        parser2.add_argument("--domain-id", type=str, default="77")
        parser2.add_argument("--no-localhost-only", action="store_true")
        parser2.add_argument("--lidar-2d", type=str, choices=["true", "false"], default=None)
        args = parser2.parse_args()
        args.command = "run"

    if args.command == "run":
        out_dir = Path(args.outdir)
        lidar_2d_val = None if args.lidar_2d is None else (args.lidar_2d == "true")
        df_pipe, df_res = run_benchmark_trial(
            bag_path=args.bag,
            trial_duration_s=args.duration,
            output_dir=out_dir,
            trial_label=args.label,
            domain_id=args.domain_id,
            localhost_only=not args.no_localhost_only,
            lidar_2d=lidar_2d_val,
        )
        if df_pipe.empty:
            sys.exit(2)
        generate_publication_visualizations(df_pipe, df_res, output_dir=out_dir, prefix=args.label)
        df_summary = generate_markdown_and_latex_reports(df_pipe, df_res, output_dir=out_dir, prefix=args.label)
        print("\n[+] Benchmark Trial Completed Successfully!")
        print(df_summary.to_string(index=False))

    elif args.command == "compare":
        out_dir = Path(args.outdir)
        df_comp = compare_trials(
            baseline_dir=args.baseline_dir,
            proposed_dir=args.proposed_dir,
            baseline_label=args.baseline_label,
            proposed_label=args.proposed_label,
            output_dir=out_dir
        )
        print("\n[+] Comparison Completed!")
        print(df_comp.to_string(index=False))


if __name__ == "__main__":
    main()

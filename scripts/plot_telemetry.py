#!/usr/bin/env python3
# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0

"""
Telemetry Plotting and Parameter Tuning Tool for gn10_pointcloud_localization.

Provides both real-time dynamic animated plotting from the /localization/telemetry topic
and offline plotting from a recorded CSV file.

Usage:
    # Live real-time animated visualization (updates live at 10Hz from ~50Hz telemetry):
    ros2 run gn10_pointcloud_localization plot_telemetry.py

    # Offline plotting from CSV:
    python3 scripts/plot_telemetry.py --csv telemetry.csv --save telemetry_plot.png
"""

import argparse
import csv
import os
import sys
import threading
import time
from collections import deque
import numpy as np

# Re-exec with -O (PYTHONOPTIMIZE=1) to prevent rosidl generator assertion crashes
# on float32 bounds (e.g. std::numeric_limits<float>::max() > 3.402823466e+38) throwing:
# "RuntimeError: Unable to convert call argument to Python object"
if __debug__:
    os.environ["PYTHONOPTIMIZE"] = "1"
    os.execv(sys.executable, [sys.executable, "-O"] + sys.argv)

# Prevent OpenGL crashes on systems without dedicated GPU or DRI driver
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

# Use non-interactive Agg backend if running in a headless / no-display environment
IS_HEADLESS = ("DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ)
if IS_HEADLESS:
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError:
    print("Error: matplotlib is required. Install with: pip3 install matplotlib", file=sys.stderr)
    sys.exit(1)


def sanitize_metric(values, default=0.0, max_limit=None, min_limit=0.0):
    """Sanitizes non-finite or extreme values to prevent Matplotlib GUI crashes."""
    arr = np.array(values, dtype=float)
    non_finite = ~np.isfinite(arr)
    arr[non_finite] = default
    if min_limit is not None or max_limit is not None:
        arr = np.clip(arr, min_limit, max_limit)
    return arr


def plot_static(records, save_path=None, title="Localization Telemetry Analysis"):
    if not records or len(records["timestamp"]) == 0:
        print("No telemetry data to plot.")
        return

    t = np.array(records["timestamp"], dtype=float)
    t = t - t[0]  # Relative seconds

    # Sanitize metrics to guarantee finite bounds for plotting
    b_cost = sanitize_metric(records["best_cost"], default=np.nan, max_limit=5.0)
    c_3d = sanitize_metric(records["cost_3d"], default=np.nan, max_limit=5.0)
    c_2d = sanitize_metric(records["cost_2d"], default=np.nan, max_limit=5.0)
    i_ratio = sanitize_metric(records["inlier_ratio"], default=0.0, min_limit=0.0, max_limit=1.0) * 100.0
    i_cost = sanitize_metric(records["inlier_cost"], default=np.nan, max_limit=2.0)
    spd = sanitize_metric(records["linear_speed"], default=0.0, min_limit=0.0, max_limit=20.0)
    r_xy = sanitize_metric(records["search_range_xy"], default=0.0, min_limit=0.0, max_limit=10.0)
    r_yaw = sanitize_metric(records["search_range_yaw"], default=0.0, min_limit=0.0, max_limit=10.0)
    o_tot = sanitize_metric(records["obstacle_count_total"], default=0.0, min_limit=0.0)
    o_3d = sanitize_metric(records["obstacle_count_3d"], default=0.0, min_limit=0.0)
    o_2d = sanitize_metric(records["obstacle_count_2d"], default=0.0, min_limit=0.0)
    i_cnt = sanitize_metric(records["inlier_count"], default=0.0, min_limit=0.0)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Matching Costs
    ax0 = axes[0]
    ax0.plot(t, b_cost, label="Total Cost (best_cost)", color="tab:blue", linewidth=1.5)
    ax0.plot(t, c_3d, label="3D Cost", color="tab:cyan", linestyle="--", alpha=0.8)
    ax0.plot(t, c_2d, label="2D Cost", color="tab:orange", linestyle="--", alpha=0.8)
    ax0.axhline(0.165, color="tab:red", linestyle=":", label="cost_threshold (0.165)")
    ax0.axhline(0.220, color="tab:purple", linestyle=":", label="max_inlier_accept_cost (0.220)")
    ax0.set_ylabel("Cost [m]")
    ax0.set_title("Matching Cost Components & Thresholds")
    ax0.grid(True, linestyle="--", alpha=0.6)
    ax0.legend(loc="upper right", ncol=3, fontsize=9)
    valid_costs = b_cost[np.isfinite(b_cost)]
    max_c = max(0.40, min(5.0, float(np.max(valid_costs)) * 1.15)) if len(valid_costs) > 0 else 0.40
    ax0.set_ylim(0.0, max_c)

    # 2. Inlier Metrics
    ax1 = axes[1]
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(t, i_ratio, label="Inlier Ratio (%)", color="tab:green", linewidth=1.5)
    ax1.axhline(35.0, color="darkgreen", linestyle=":", label="min_inlier_ratio (35%)")
    l2 = ax1_twin.plot(t, i_cost, label="Inlier Cost [m]", color="tab:olive", linestyle="-.", alpha=0.9)
    ax1_twin.axhline(0.010, color="gray", linestyle=":", label="inlier_cost_thresh (0.01m)")
    ax1.set_ylabel("Inlier Ratio [%]")
    ax1_twin.set_ylabel("Inlier Cost [m]")
    ax1.set_title("Inlier Ratio & Inlier Cost")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_ylim(0.0, 100.0)
    valid_ic = i_cost[np.isfinite(i_cost)]
    max_ic = max(0.08, min(1.0, float(np.max(valid_ic)) * 1.15)) if len(valid_ic) > 0 else 0.08
    ax1_twin.set_ylim(0.0, max_ic)
    lines = l1 + l2
    labels = [l.get_label() for l in lines] + ["min_inlier_ratio (35%)", "inlier_cost_thresh (0.01m)"]
    ax1.legend(lines + [ax1.get_lines()[-1], ax1_twin.get_lines()[-1]], labels, loc="upper right", ncol=2, fontsize=9)

    # 3. Kinematics & Search Windows
    ax2 = axes[2]
    ax2.plot(t, spd, label="Speed (m/s)", color="tab:red", linewidth=1.5)
    ax2.plot(t, r_xy, label="Search Range XY [m]", color="tab:blue", linestyle="-.")
    ax2.plot(t, r_yaw, label="Search Range Yaw [rad]", color="tab:orange", linestyle="--")
    ax2.set_ylabel("Kinematics / Bounds")
    ax2.set_title("Robot Linear Speed & Adaptive Search Windows")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", ncol=3, fontsize=9)

    # 4. Point Counts
    ax3 = axes[3]
    ax3.plot(t, o_tot, label="Total Obstacles", color="black", linewidth=1.2)
    ax3.plot(t, o_3d, label="3D Obstacles", color="tab:cyan", linestyle="--", alpha=0.7)
    ax3.plot(t, o_2d, label="2D Obstacles", color="tab:orange", linestyle="--", alpha=0.7)
    ax3.plot(t, i_cnt, label="Inliers", color="tab:green", linewidth=1.2)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Points")
    ax3.set_title("Obstacle & Inlier Point Counts")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right", ncol=4, fontsize=9)
    max_pts = max(50.0, float(np.max(o_tot)) * 1.15) if len(o_tot) > 0 else 100.0
    ax3.set_ylim(0.0, max_pts)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved successfully to: {save_path}")
    elif IS_HEADLESS:
        print("Warning: Running in a headless environment without GUI. Specify --save <file.png> to save the plot.")
    else:
        plt.show()


def run_csv_mode(csv_path, save_path=None):
    data = {
        "timestamp": [],
        "best_cost": [],
        "cost_3d": [],
        "cost_2d": [],
        "weight_2d": [],
        "inlier_cost": [],
        "inlier_count": [],
        "inlier_ratio": [],
        "obstacle_count_3d": [],
        "obstacle_count_2d": [],
        "obstacle_count_total": [],
        "linear_speed": [],
        "search_range_xy": [],
        "search_range_yaw": [],
    }
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["timestamp"].append(float(row.get("header.stamp.sec", 0)) + float(row.get("header.stamp.nanosec", 0)) * 1e-9)
            data["best_cost"].append(float(row.get("best_cost", 0.0)))
            data["cost_3d"].append(float(row.get("cost_3d", 0.0)))
            data["cost_2d"].append(float(row.get("cost_2d", 0.0)))
            data["weight_2d"].append(float(row.get("weight_2d", 0.0)))
            data["inlier_cost"].append(float(row.get("inlier_cost", 0.0)))
            data["inlier_count"].append(int(float(row.get("inlier_count", 0))))
            data["inlier_ratio"].append(float(row.get("inlier_ratio", 0.0)))
            data["obstacle_count_3d"].append(int(float(row.get("obstacle_count_3d", 0))))
            data["obstacle_count_2d"].append(int(float(row.get("obstacle_count_2d", 0))))
            data["obstacle_count_total"].append(int(float(row.get("obstacle_count_total", 0))))
            data["linear_speed"].append(float(row.get("linear_speed", 0.0)))
            data["search_range_xy"].append(float(row.get("search_range_xy", 0.0)))
            data["search_range_yaw"].append(float(row.get("search_range_yaw", 0.0)))

    plot_static(data, save_path=save_path, title=f"Localization Telemetry Analysis ({csv_path})")


def run_live_node(topic, buffer_len=500, save_path=None, window_sec=15.0):
    """
    Subscribes to the telemetry topic in a background ROS 2 thread
    and renders real-time dynamic animated curves via matplotlib FuncAnimation.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from gn10_pointcloud_localization.msg import LocalizationTelemetry
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("ROS 2 Python packages not available. In GPU-less environments, build the CPU monitor with:", file=sys.stderr)
        print("    colcon build --packages-select gn10_pointcloud_localization", file=sys.stderr)
        print("    source install/setup.bash", file=sys.stderr)
        sys.exit(1)

    rclpy.init()

    lock = threading.Lock()
    buffer = {
        "timestamp": deque(maxlen=buffer_len),
        "best_cost": deque(maxlen=buffer_len),
        "cost_3d": deque(maxlen=buffer_len),
        "cost_2d": deque(maxlen=buffer_len),
        "weight_2d": deque(maxlen=buffer_len),
        "inlier_cost": deque(maxlen=buffer_len),
        "inlier_count": deque(maxlen=buffer_len),
        "inlier_ratio": deque(maxlen=buffer_len),
        "obstacle_count_3d": deque(maxlen=buffer_len),
        "obstacle_count_2d": deque(maxlen=buffer_len),
        "obstacle_count_total": deque(maxlen=buffer_len),
        "linear_speed": deque(maxlen=buffer_len),
        "search_range_xy": deque(maxlen=buffer_len),
        "search_range_yaw": deque(maxlen=buffer_len),
    }

    class TelemetrySubscriber(Node):
        def __init__(self):
            super().__init__("telemetry_plotter")
            self.sub = self.create_subscription(
                LocalizationTelemetry, topic, self.callback, 10
            )
            self.get_logger().info(f"Subscribed to {topic}. Streaming real-time telemetry animation...")

        def callback(self, msg: LocalizationTelemetry):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with lock:
                buffer["timestamp"].append(t)
                buffer["best_cost"].append(msg.best_cost)
                buffer["cost_3d"].append(msg.cost_3d)
                buffer["cost_2d"].append(msg.cost_2d)
                buffer["weight_2d"].append(msg.weight_2d)
                buffer["inlier_cost"].append(msg.inlier_cost)
                buffer["inlier_count"].append(msg.inlier_count)
                buffer["inlier_ratio"].append(msg.inlier_ratio)
                buffer["obstacle_count_3d"].append(msg.obstacle_count_3d)
                buffer["obstacle_count_2d"].append(msg.obstacle_count_2d)
                buffer["obstacle_count_total"].append(msg.obstacle_count_total)
                buffer["linear_speed"].append(msg.linear_speed)
                buffer["search_range_xy"].append(msg.search_range_xy)
                buffer["search_range_yaw"].append(msg.search_range_yaw)

    node = TelemetrySubscriber()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    def spin_worker():
        try:
            executor.spin()
        except Exception:
            pass

    # Run ROS 2 spin in background daemon thread
    spin_thread = threading.Thread(target=spin_worker, daemon=True)
    spin_thread.start()

    # Set up interactive matplotlib figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Real-Time Localization Telemetry ({topic})", fontsize=13, fontweight="bold")

    # 1. Costs
    ax0 = axes[0]
    line_best, = ax0.plot([], [], label="Total Cost", color="tab:blue", linewidth=1.5)
    line_3d, = ax0.plot([], [], label="3D Cost", color="tab:cyan", linestyle="--", alpha=0.8)
    line_2d, = ax0.plot([], [], label="2D Cost", color="tab:orange", linestyle="--", alpha=0.8)
    ax0.axhline(0.165, color="tab:red", linestyle=":", label="cost_threshold (0.165)")
    ax0.axhline(0.220, color="tab:purple", linestyle=":", label="max_inlier_accept_cost (0.220)")
    ax0.set_ylabel("Cost [m]")
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="upper right", ncol=3, fontsize=8)
    ax0.set_ylim(0.0, 0.40)

    # 2. Inlier Metrics
    ax1 = axes[1]
    ax1_twin = ax1.twinx()
    line_inlier_ratio, = ax1.plot([], [], label="Inlier Ratio (%)", color="tab:green", linewidth=1.5)
    ax1.axhline(35.0, color="darkgreen", linestyle=":", label="min_inlier_ratio (35%)")
    line_inlier_cost, = ax1_twin.plot([], [], label="Inlier Cost [m]", color="tab:olive", linestyle="-.", alpha=0.9)
    ax1_twin.axhline(0.010, color="gray", linestyle=":", label="inlier_cost_thresh (0.01m)")
    ax1.set_ylabel("Inlier Ratio [%]")
    ax1_twin.set_ylabel("Inlier Cost [m]")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_ylim(0.0, 100.0)
    ax1_twin.set_ylim(0.0, 0.08)

    # 3. Kinematics & Search Windows
    ax2 = axes[2]
    line_speed, = ax2.plot([], [], label="Speed [m/s]", color="tab:red", linewidth=1.5)
    line_range_xy, = ax2.plot([], [], label="Search Range XY [m]", color="tab:blue", linestyle="-.")
    line_range_yaw, = ax2.plot([], [], label="Search Range Yaw [rad]", color="tab:orange", linestyle="--")
    ax2.set_ylabel("Kinematics / Bounds")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", ncol=3, fontsize=8)
    ax2.set_ylim(0.0, 4.5)

    # 4. Point Counts
    ax3 = axes[3]
    line_obs_tot, = ax3.plot([], [], label="Total Obstacles", color="black", linewidth=1.2)
    line_obs_3d, = ax3.plot([], [], label="3D Obstacles", color="tab:cyan", linestyle="--", alpha=0.7)
    line_obs_2d, = ax3.plot([], [], label="2D Obstacles", color="tab:orange", linestyle="--", alpha=0.7)
    line_inliers, = ax3.plot([], [], label="Inliers", color="tab:green", linewidth=1.2)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Points")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", ncol=4, fontsize=8)

    plt.tight_layout()

    def update_plot(_frame):
        artists = (
            line_best, line_3d, line_2d, line_inlier_ratio, line_inlier_cost,
            line_speed, line_range_xy, line_range_yaw, line_obs_tot,
            line_obs_3d, line_obs_2d, line_inliers
        )
        try:
            with lock:
                if len(buffer["timestamp"]) < 2:
                    return artists
                t_arr = np.array(buffer["timestamp"], dtype=float)
                b_cost = np.array(buffer["best_cost"], dtype=float)
                c_3d = np.array(buffer["cost_3d"], dtype=float)
                c_2d = np.array(buffer["cost_2d"], dtype=float)
                i_cost = np.array(buffer["inlier_cost"], dtype=float)
                i_cnt = np.array(buffer["inlier_count"], dtype=float)
                i_ratio = np.array(buffer["inlier_ratio"], dtype=float) * 100.0
                o_3d = np.array(buffer["obstacle_count_3d"], dtype=float)
                o_2d = np.array(buffer["obstacle_count_2d"], dtype=float)
                o_tot = np.array(buffer["obstacle_count_total"], dtype=float)
                spd = np.array(buffer["linear_speed"], dtype=float)
                r_xy = np.array(buffer["search_range_xy"], dtype=float)
                r_yaw = np.array(buffer["search_range_yaw"], dtype=float)

            # Scrolling relative time: display up to window_sec
            t_latest = t_arr[-1]
            mask = (t_arr >= (t_latest - window_sec))
            if not np.any(mask):
                return artists

            t_sub = t_arr[mask] - t_latest  # Negative to 0s

            # Sanitize metrics to prevent Tkinter/GUI backend overflow crashes on spikes
            b_sub = np.clip(np.where(np.isfinite(b_cost[mask]), b_cost[mask], np.nan), 0.0, 5.0)
            c3_sub = np.clip(np.where(np.isfinite(c_3d[mask]), c_3d[mask], np.nan), 0.0, 5.0)
            c2_sub = np.clip(np.where(np.isfinite(c_2d[mask]), c_2d[mask], np.nan), 0.0, 5.0)
            ic_sub = np.clip(np.where(np.isfinite(i_cost[mask]), i_cost[mask], np.nan), 0.0, 2.0)
            ir_sub = np.clip(np.where(np.isfinite(i_ratio[mask]), i_ratio[mask], 0.0), 0.0, 100.0)
            ot_sub = np.clip(np.where(np.isfinite(o_tot[mask]), o_tot[mask], 0.0), 0.0, 1e6)
            o3_sub = np.clip(np.where(np.isfinite(o_3d[mask]), o_3d[mask], 0.0), 0.0, 1e6)
            o2_sub = np.clip(np.where(np.isfinite(o_2d[mask]), o_2d[mask], 0.0), 0.0, 1e6)
            icnt_sub = np.clip(np.where(np.isfinite(i_cnt[mask]), i_cnt[mask], 0.0), 0.0, 1e6)
            spd_sub = np.clip(np.where(np.isfinite(spd[mask]), spd[mask], 0.0), 0.0, 20.0)
            rxy_sub = np.clip(np.where(np.isfinite(r_xy[mask]), r_xy[mask], 0.0), 0.0, 10.0)
            ryaw_sub = np.clip(np.where(np.isfinite(r_yaw[mask]), r_yaw[mask], 0.0), 0.0, 10.0)

            line_best.set_data(t_sub, b_sub)
            line_3d.set_data(t_sub, c3_sub)
            line_2d.set_data(t_sub, c2_sub)

            line_inlier_ratio.set_data(t_sub, ir_sub)
            line_inlier_cost.set_data(t_sub, ic_sub)

            line_speed.set_data(t_sub, spd_sub)
            line_range_xy.set_data(t_sub, rxy_sub)
            line_range_yaw.set_data(t_sub, ryaw_sub)

            line_obs_tot.set_data(t_sub, ot_sub)
            line_obs_3d.set_data(t_sub, o3_sub)
            line_obs_2d.set_data(t_sub, o2_sub)
            line_inliers.set_data(t_sub, icnt_sub)

            # Adjust X bounds to scrolling window [-window_sec, 0]
            x_min = min(-window_sec, float(t_sub[0]))
            ax3.set_xlim(x_min, 0.5)

            # Adaptive autoscale Y for cost (ax0) with a safe ceiling so spikes are visible without crashing
            finite_b = b_sub[np.isfinite(b_sub)]
            max_cost = max(0.40, min(5.0, float(np.max(finite_b)) * 1.15)) if len(finite_b) > 0 else 0.40
            ax0.set_ylim(0.0, max_cost)

            # Adaptive autoscale Y for inlier cost (ax1_twin)
            finite_ic = ic_sub[np.isfinite(ic_sub)]
            max_ic = max(0.08, min(1.0, float(np.max(finite_ic)) * 1.15)) if len(finite_ic) > 0 else 0.08
            ax1_twin.set_ylim(0.0, max_ic)

            # Autoscale Y for obstacle counts
            max_pts = max(50.0, float(np.max(ot_sub)) * 1.15) if len(ot_sub) > 0 else 100.0
            ax3.set_ylim(0.0, max_pts)
        except Exception as e:
            # Prevent unhandled exceptions from terminating FuncAnimation
            print(f"[TelemetryViewer Warning] update_plot exception caught: {e}", file=sys.stderr)

        return artists

    # Animate at 10 Hz (interval=100ms)
    ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)

    import signal

    def sigint_handler(_sig, _frame):
        plt.close("all")

    try:
        signal.signal(signal.SIGINT, sigint_handler)
    except Exception:
        pass

    try:
        backend = matplotlib.get_backend().lower()
        if "agg" in backend and not ("tkagg" in backend or "qtagg" in backend):
            print(f"Running in headless/no-GUI mode (backend: {backend}).")
            print(f"Streaming telemetry from {topic}. Press Ctrl+C to stop and process plot...")
            while rclpy.ok():
                time.sleep(0.2)
        else:
            plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close("all")
        if save_path:
            with lock:
                records = {k: list(v) for k, v in buffer.items()}
            plot_static(records, save_path=save_path)
        executor.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="GN10 Localization Telemetry Plotter")
    parser.add_argument("--csv", type=str, default=None, help="Path to telemetry CSV file")
    parser.add_argument("--topic", type=str, default="/localization/telemetry", help="ROS 2 telemetry topic")
    parser.add_argument("--save", type=str, default=None, help="Path to save output plot image")
    parser.add_argument("--buffer", type=int, default=1500, help="Live buffer capacity")
    parser.add_argument("--window", type=float, default=15.0, help="Real-time scrolling window duration [s]")
    args = parser.parse_args()

    if args.csv:
        run_csv_mode(args.csv, save_path=args.save)
    else:
        run_live_node(args.topic, buffer_len=args.buffer, save_path=args.save, window_sec=args.window)


if __name__ == "__main__":
    main()

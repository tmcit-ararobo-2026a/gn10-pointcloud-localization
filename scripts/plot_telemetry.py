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


def plot_static(records, save_path=None, title="Localization Telemetry Analysis"):
    if not records or len(records["timestamp"]) == 0:
        print("No telemetry data to plot.")
        return

    t = np.array(records["timestamp"])
    t = t - t[0]  # Relative seconds

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Matching Costs
    ax0 = axes[0]
    ax0.plot(t, records["best_cost"], label="Total Cost (best_cost)", color="tab:blue", linewidth=1.5)
    ax0.plot(t, records["cost_3d"], label="3D Cost", color="tab:cyan", linestyle="--", alpha=0.8)
    ax0.plot(t, records["cost_2d"], label="2D Cost", color="tab:orange", linestyle="--", alpha=0.8)
    ax0.axhline(0.165, color="tab:red", linestyle=":", label="cost_threshold (0.165)")
    ax0.axhline(0.220, color="tab:purple", linestyle=":", label="max_inlier_accept_cost (0.220)")
    ax0.set_ylabel("Cost [m]")
    ax0.set_title("Matching Cost Components & Thresholds")
    ax0.grid(True, linestyle="--", alpha=0.6)
    ax0.legend(loc="upper right", ncol=3, fontsize=9)

    # 2. Inlier Metrics
    ax1 = axes[1]
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(t, np.array(records["inlier_ratio"]) * 100.0, label="Inlier Ratio (%)", color="tab:green", linewidth=1.5)
    ax1.axhline(35.0, color="darkgreen", linestyle=":", label="min_inlier_ratio (35%)")
    l2 = ax1_twin.plot(t, records["inlier_cost"], label="Inlier Cost [m]", color="tab:olive", linestyle="-.", alpha=0.9)
    ax1_twin.axhline(0.010, color="gray", linestyle=":", label="inlier_cost_thresh (0.01m)")
    ax1.set_ylabel("Inlier Ratio [%]")
    ax1_twin.set_ylabel("Inlier Cost [m]")
    ax1.set_title("Inlier Ratio & Inlier Cost")
    ax1.grid(True, linestyle="--", alpha=0.6)
    lines = l1 + l2
    labels = [l.get_label() for l in lines] + ["min_inlier_ratio (35%)", "inlier_cost_thresh (0.01m)"]
    ax1.legend(lines + [ax1.get_lines()[-1], ax1_twin.get_lines()[-1]], labels, loc="upper right", ncol=2, fontsize=9)

    # 3. Kinematics & Search Windows
    ax2 = axes[2]
    ax2.plot(t, records["linear_speed"], label="Speed (m/s)", color="tab:red", linewidth=1.5)
    ax2.plot(t, records["search_range_xy"], label="Search Range XY [m]", color="tab:blue", linestyle="-.")
    ax2.plot(t, records["search_range_yaw"], label="Search Range Yaw [rad]", color="tab:orange", linestyle="--")
    ax2.set_ylabel("Kinematics / Bounds")
    ax2.set_title("Robot Linear Speed & Adaptive Search Windows")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", ncol=3, fontsize=9)

    # 4. Point Counts
    ax3 = axes[3]
    ax3.plot(t, records["obstacle_count_total"], label="Total Obstacles", color="black", linewidth=1.2)
    ax3.plot(t, records["obstacle_count_3d"], label="3D Obstacles", color="tab:cyan", linestyle="--", alpha=0.7)
    ax3.plot(t, records["obstacle_count_2d"], label="2D Obstacles", color="tab:orange", linestyle="--", alpha=0.7)
    ax3.plot(t, records["inlier_count"], label="Inliers", color="tab:green", linewidth=1.2)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Points")
    ax3.set_title("Obstacle & Inlier Point Counts")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right", ncol=4, fontsize=9)

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

    # Run ROS 2 spin in background daemon thread
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
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
        with lock:
            if len(buffer["timestamp"]) < 2:
                return line_best,
            t_arr = np.array(buffer["timestamp"])
            b_cost = np.array(buffer["best_cost"])
            c_3d = np.array(buffer["cost_3d"])
            c_2d = np.array(buffer["cost_2d"])
            i_cost = np.array(buffer["inlier_cost"])
            i_cnt = np.array(buffer["inlier_count"])
            i_ratio = np.array(buffer["inlier_ratio"]) * 100.0
            o_3d = np.array(buffer["obstacle_count_3d"])
            o_2d = np.array(buffer["obstacle_count_2d"])
            o_tot = np.array(buffer["obstacle_count_total"])
            spd = np.array(buffer["linear_speed"])
            r_xy = np.array(buffer["search_range_xy"])
            r_yaw = np.array(buffer["search_range_yaw"])

        # Scrolling relative time: display up to window_sec
        t_latest = t_arr[-1]
        mask = (t_arr >= (t_latest - window_sec))
        if not np.any(mask):
            return line_best,

        t_sub = t_arr[mask] - t_latest  # Negative to 0s

        line_best.set_data(t_sub, b_cost[mask])
        line_3d.set_data(t_sub, c_3d[mask])
        line_2d.set_data(t_sub, c_2d[mask])

        line_inlier_ratio.set_data(t_sub, i_ratio[mask])
        line_inlier_cost.set_data(t_sub, i_cost[mask])

        line_speed.set_data(t_sub, spd[mask])
        line_range_xy.set_data(t_sub, r_xy[mask])
        line_range_yaw.set_data(t_sub, r_yaw[mask])

        line_obs_tot.set_data(t_sub, o_tot[mask])
        line_obs_3d.set_data(t_sub, o_3d[mask])
        line_obs_2d.set_data(t_sub, o_2d[mask])
        line_inliers.set_data(t_sub, i_cnt[mask])

        # Adjust X bounds to scrolling window [-window_sec, 0]
        x_min = min(-window_sec, t_sub[0])
        ax3.set_xlim(x_min, 0.5)

        # Autoscale Y for obstacle counts
        max_pts = max(50, np.max(o_tot[mask]) * 1.15) if len(o_tot[mask]) > 0 else 100
        ax3.set_ylim(0, max_pts)

        return line_best, line_3d, line_2d, line_inlier_ratio, line_inlier_cost, line_speed, line_range_xy, line_range_yaw, line_obs_tot, line_obs_3d, line_obs_2d, line_inliers

    # Animate at 10 Hz (interval=100ms)
    ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)

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
        if save_path:
            with lock:
                records = {k: list(v) for k, v in buffer.items()}
            plot_static(records, save_path=save_path)
        node.destroy_node()
        rclpy.shutdown()


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

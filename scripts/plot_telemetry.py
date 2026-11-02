#!/usr/bin/env python3
# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0

"""
Telemetry Plotting and Parameter Tuning Tool for gn10_pointcloud_localization.

Plots matching costs, inlier metrics, speed, and adaptive search windows
either in real time from the /localization/telemetry topic or offline from a CSV file.

Usage:
    # Live visualization via ROS 2 topic:
    ros2 run gn10_pointcloud_localization plot_telemetry.py

    # Offline plotting from CSV:
    python3 scripts/plot_telemetry.py --csv telemetry.csv --save telemetry_plot.png
"""

import argparse
import sys
from collections import deque
import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required. Install with: pip3 install matplotlib", file=sys.stderr)
    sys.exit(1)


def plot_data(records, save_path=None, title="Localization Telemetry Analysis"):
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
    else:
        plt.show()


def run_csv_mode(csv_path, save_path=None):
    import csv
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

    plot_data(data, save_path=save_path, title=f"Localization Telemetry Analysis ({csv_path})")


def run_live_node(topic, buffer_len=500, save_path=None):
    try:
        import rclpy
        from rclpy.node import Node
        from gn10_pointcloud_localization.msg import LocalizationTelemetry
    except ImportError:
        print("ROS 2 Python packages not available. Source your ROS 2 environment first.", file=sys.stderr)
        sys.exit(1)

    rclpy.init()

    class TelemetrySubscriber(Node):
        def __init__(self):
            super().__init__("telemetry_plotter")
            self.buffer = {
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
            self.sub = self.create_subscription(
                LocalizationTelemetry, topic, self.callback, 10
            )
            self.get_logger().info(f"Subscribed to {topic}. Waiting for telemetry messages...")

        def callback(self, msg: LocalizationTelemetry):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.buffer["timestamp"].append(t)
            self.buffer["best_cost"].append(msg.best_cost)
            self.buffer["cost_3d"].append(msg.cost_3d)
            self.buffer["cost_2d"].append(msg.cost_2d)
            self.buffer["weight_2d"].append(msg.weight_2d)
            self.buffer["inlier_cost"].append(msg.inlier_cost)
            self.buffer["inlier_count"].append(msg.inlier_count)
            self.buffer["inlier_ratio"].append(msg.inlier_ratio)
            self.buffer["obstacle_count_3d"].append(msg.obstacle_count_3d)
            self.buffer["obstacle_count_2d"].append(msg.obstacle_count_2d)
            self.buffer["obstacle_count_total"].append(msg.obstacle_count_total)
            self.buffer["linear_speed"].append(msg.linear_speed)
            self.buffer["search_range_xy"].append(msg.search_range_xy)
            self.buffer["search_range_yaw"].append(msg.search_range_yaw)

    node = TelemetrySubscriber()

    try:
        # Spin until interrupted
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        records = {k: list(v) for k, v in node.buffer.items()}
        plot_data(records, save_path=save_path)
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="GN10 Localization Telemetry Plotter")
    parser.add_argument("--csv", type=str, default=None, help="Path to telemetry CSV file")
    parser.add_argument("--topic", type=str, default="/localization/telemetry", help="ROS 2 telemetry topic")
    parser.add_argument("--save", type=str, default=None, help="Path to save output plot image")
    parser.add_argument("--buffer", type=int, default=1000, help="Live buffer capacity")
    args = parser.parse_args()

    if args.csv:
        run_csv_mode(args.csv, save_path=args.save)
    else:
        run_live_node(args.topic, buffer_len=args.buffer, save_path=args.save)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0

"""
Telemetry Plotting and Parameter Tuning Tool for gn10_pointcloud_localization.

Provides both real-time dynamic animated plotting from the /localization/telemetry topic
and offline plotting from a recorded CSV file. Threshold reference lines are automatically
loaded from config/common_params.yaml (or --params-file) and dynamically synchronized with
the running localization node via ROS 2 parameter services and /parameter_events.

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
if __debug__ and sys.argv and os.path.isfile(sys.argv[0]):
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

try:
    import yaml
except ImportError:
    yaml = None


def get_default_thresholds():
    """Returns baseline threshold dictionary matching C++ struct defaults."""
    return {
        "cost_threshold_3d": 0.22,
        "cost_threshold_2d": 0.08,
        "max_inlier_accept_cost": 0.22,
        "min_inlier_ratio": 0.28,
        "inlier_cost_thresh": 0.035,
        "min_inliers": 400,
        "min_3d_points": 50,
        "min_2d_points": 30,
    }


def resolve_default_config_paths():
    """Finds common_params.yaml from workspace source or installed share directory."""
    paths = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_src = os.path.abspath(os.path.join(script_dir, "..", "config", "common_params.yaml"))
    if os.path.isfile(candidate_src):
        paths.append(candidate_src)

    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory("gn10_pointcloud_localization")
        candidate_share = os.path.join(share_dir, "config", "common_params.yaml")
        if os.path.isfile(candidate_share) and candidate_share not in paths:
            paths.append(candidate_share)
    except Exception:
        pass

    return paths


def load_thresholds_from_yaml(yaml_path, thresholds=None):
    """Loads matching_params thresholds from a ROS 2 YAML parameter file."""
    if thresholds is None:
        thresholds = get_default_thresholds()
    if yaml is None or not yaml_path or not os.path.isfile(yaml_path):
        return thresholds

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, dict):
            return thresholds

        # Support /**/ros__parameters or <node_name>/ros__parameters
        ros_params = {}
        for _, node_block in doc.items():
            if isinstance(node_block, dict) and "ros__parameters" in node_block:
                if isinstance(node_block["ros__parameters"], dict):
                    ros_params.update(node_block["ros__parameters"])

        mp = ros_params.get("matching_params", {})
        if isinstance(mp, dict):
            if "cost_threshold" in mp and "cost_threshold_3d" not in mp:
                thresholds["cost_threshold_3d"] = float(mp["cost_threshold"])
            if "cost_threshold_3d" in mp:
                thresholds["cost_threshold_3d"] = float(mp["cost_threshold_3d"])
            if "cost_threshold_2d" in mp:
                thresholds["cost_threshold_2d"] = float(mp["cost_threshold_2d"])
            if "max_inlier_accept_cost" in mp:
                thresholds["max_inlier_accept_cost"] = float(mp["max_inlier_accept_cost"])
            if "min_inlier_ratio" in mp:
                thresholds["min_inlier_ratio"] = float(mp["min_inlier_ratio"])
            if "inlier_cost_thresh" in mp:
                thresholds["inlier_cost_thresh"] = float(mp["inlier_cost_thresh"])
            if "min_inliers" in mp:
                thresholds["min_inliers"] = int(mp["min_inliers"])
            if "min_3d_points" in mp:
                thresholds["min_3d_points"] = int(mp["min_3d_points"])
            if "min_2d_points" in mp:
                thresholds["min_2d_points"] = int(mp["min_2d_points"])
    except Exception as e:
        print(f"[TelemetryViewer Warning] Failed to parse '{yaml_path}': {e}", file=sys.stderr)

    return thresholds


def init_thresholds(custom_params_files=None):
    """Initializes thresholds from common_params.yaml and any user-specified YAML files."""
    thresholds = get_default_thresholds()
    default_paths = resolve_default_config_paths()
    if default_paths:
        load_thresholds_from_yaml(default_paths[0], thresholds)
    if custom_params_files:
        for path in custom_params_files:
            load_thresholds_from_yaml(path, thresholds)
    return thresholds


def sanitize_metric(values, default=0.0, max_limit=None, min_limit=0.0):
    """Sanitizes non-finite or extreme values to prevent Matplotlib GUI crashes."""
    arr = np.array(values, dtype=float)
    non_finite = ~np.isfinite(arr)
    arr[non_finite] = default
    if min_limit is not None or max_limit is not None:
        arr = np.clip(arr, min_limit, max_limit)
    return arr


def plot_static(records, thresholds=None, save_path=None, title="Localization Telemetry Analysis"):
    if not records or len(records["timestamp"]) == 0:
        print("No telemetry data to plot.")
        return

    if thresholds is None:
        thresholds = init_thresholds()

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

    th_3d = thresholds["cost_threshold_3d"]
    th_2d = thresholds["cost_threshold_2d"]
    th_max_inlier_cost = thresholds["max_inlier_accept_cost"]
    th_inlier_ratio_pct = thresholds["min_inlier_ratio"] * 100.0
    th_inlier_cost = thresholds["inlier_cost_thresh"]
    th_min_inliers = thresholds["min_inliers"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Matching Costs
    ax0 = axes[0]
    ax0.plot(t, b_cost, label="Total Cost (best_cost)", color="tab:blue", linewidth=1.5)
    ax0.plot(t, c_3d, label="3D Cost", color="tab:cyan", linestyle="--", alpha=0.8)
    ax0.plot(t, c_2d, label="2D Cost", color="tab:orange", linestyle="--", alpha=0.8)
    ax0.axhline(th_3d, color="tab:red", linestyle=":", label=f"cost_threshold_3d ({th_3d:.3f}m)")
    ax0.axhline(th_2d, color="tab:brown", linestyle=":", label=f"cost_threshold_2d ({th_2d:.3f}m)")
    ax0.axhline(
        th_max_inlier_cost,
        color="tab:purple",
        linestyle="-.",
        alpha=0.7,
        label=f"max_inlier_accept_cost ({th_max_inlier_cost:.3f}m)",
    )
    ax0.set_ylabel("Cost [m]")
    ax0.set_title("Matching Cost Components & Thresholds")
    ax0.grid(True, linestyle="--", alpha=0.6)
    ax0.legend(loc="upper right", ncol=3, fontsize=8)
    valid_costs = b_cost[np.isfinite(b_cost)]
    max_c = max(0.40, min(5.0, float(np.max(valid_costs)) * 1.15)) if len(valid_costs) > 0 else 0.40
    ax0.set_ylim(0.0, max_c)

    # 2. Inlier Metrics
    ax1 = axes[1]
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(t, i_ratio, label="Inlier Ratio (%)", color="tab:green", linewidth=1.5)
    l_th_ratio = ax1.axhline(
        th_inlier_ratio_pct,
        color="darkgreen",
        linestyle=":",
        label=f"min_inlier_ratio ({th_inlier_ratio_pct:.1f}%)",
    )
    l2 = ax1_twin.plot(t, i_cost, label="Inlier Cost [m]", color="tab:olive", linestyle="-.", alpha=0.9)
    l_th_icost = ax1_twin.axhline(
        th_inlier_cost,
        color="gray",
        linestyle=":",
        label=f"inlier_cost_thresh ({th_inlier_cost:.3f}m)",
    )
    ax1.set_ylabel("Inlier Ratio [%]")
    ax1_twin.set_ylabel("Inlier Cost [m]")
    ax1.set_title("Inlier Ratio & Inlier Cost")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_ylim(0.0, 100.0)
    valid_ic = i_cost[np.isfinite(i_cost)]
    max_ic = max(0.08, min(1.0, float(np.max(valid_ic)) * 1.15)) if len(valid_ic) > 0 else 0.08
    ax1_twin.set_ylim(0.0, max_ic)
    all_lines = l1 + [l_th_ratio] + l2 + [l_th_icost]
    all_labels = [line.get_label() for line in all_lines]
    ax1.legend(all_lines, all_labels, loc="upper right", ncol=2, fontsize=8)

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
    ax3.axhline(
        th_min_inliers,
        color="darkgreen",
        linestyle=":",
        alpha=0.8,
        label=f"min_inliers ({th_min_inliers})",
    )
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Points")
    ax3.set_title("Obstacle & Inlier Point Counts")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right", ncol=5, fontsize=8)
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


def run_csv_mode(csv_path, thresholds, save_path=None):
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
            data["timestamp"].append(
                float(row.get("header.stamp.sec", 0)) + float(row.get("header.stamp.nanosec", 0)) * 1e-9
            )
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

    plot_static(data, thresholds=thresholds, save_path=save_path, title=f"Localization Telemetry Analysis ({csv_path})")


def run_live_node(topic, thresholds, buffer_len=500, save_path=None, window_sec=15.0, target_node_name=None):
    """
    Subscribes to the telemetry topic in a background ROS 2 thread,
    queries and tracks active threshold parameters from the localization node,
    and renders real-time dynamic animated curves via matplotlib FuncAnimation.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from rcl_interfaces.msg import ParameterEvent, ParameterType
        from rcl_interfaces.srv import GetParameters
        from gn10_pointcloud_localization.msg import LocalizationTelemetry
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("ROS 2 Python packages not available. Build the workspace with:", file=sys.stderr)
        print("    colcon build --packages-select gn10_pointcloud_localization", file=sys.stderr)
        print("    source install/setup.bash", file=sys.stderr)
        sys.exit(1)

    rclpy.init()

    lock = threading.Lock()
    thresholds_dirty = [False]
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

    param_names = [
        "matching_params.cost_threshold_3d",
        "matching_params.cost_threshold_2d",
        "matching_params.max_inlier_accept_cost",
        "matching_params.min_inlier_ratio",
        "matching_params.inlier_cost_thresh",
        "matching_params.min_inliers",
        "matching_params.min_3d_points",
        "matching_params.min_2d_points",
    ]

    class TelemetrySubscriber(Node):
        def __init__(self):
            super().__init__("telemetry_plotter")
            self.sub = self.create_subscription(
                LocalizationTelemetry, topic, self.callback, 10
            )
            self.param_event_sub = self.create_subscription(
                ParameterEvent, "/parameter_events", self.on_parameter_event, 10
            )
            self.synced_from_node = False
            self.param_clients = {}
            candidate_nodes = (
                [target_node_name]
                if target_node_name
                else ["/gn10_pointcloud_localization_node", "/gn10_localization_node"]
            )
            for node_name in candidate_nodes:
                srv_name = f"{node_name.rstrip('/')}/get_parameters"
                self.param_clients[node_name] = self.create_client(GetParameters, srv_name)

            self.sync_timer = self.create_timer(1.0, self.try_sync_parameters)
            self.get_logger().info(
                f"Subscribed to {topic}. Initial thresholds: "
                f"3D={thresholds['cost_threshold_3d']:.3f}m, "
                f"2D={thresholds['cost_threshold_2d']:.3f}m, "
                f"max_inlier_cost={thresholds['max_inlier_accept_cost']:.3f}m, "
                f"min_inliers={thresholds['min_inliers']} ({thresholds['min_inlier_ratio']*100:.0f}%), "
                f"inlier_cost_th={thresholds['inlier_cost_thresh']:.3f}m"
            )

        def try_sync_parameters(self):
            if self.synced_from_node:
                return
            for node_name, client in self.param_clients.items():
                if client.service_is_ready():
                    req = GetParameters.Request()
                    req.names = param_names
                    future = client.call_async(req)
                    future.add_done_callback(
                        lambda fut, n=node_name: self.on_get_parameters_response(fut, n)
                    )
                    self.synced_from_node = True
                    return

        def on_get_parameters_response(self, future, node_name):
            try:
                resp = future.result()
                updated = False
                with lock:
                    for name, val in zip(param_names, resp.values):
                        short_key = name.split(".")[-1]
                        if val.type == ParameterType.PARAMETER_DOUBLE:
                            thresholds[short_key] = float(val.double_value)
                            updated = True
                        elif val.type == ParameterType.PARAMETER_INTEGER:
                            thresholds[short_key] = int(val.integer_value)
                            updated = True
                    if updated:
                        thresholds_dirty[0] = True
                if updated:
                    self.get_logger().info(
                        f"Synchronized active thresholds from {node_name}: "
                        f"3D={thresholds['cost_threshold_3d']:.3f}m, "
                        f"2D={thresholds['cost_threshold_2d']:.3f}m, "
                        f"max_inlier_cost={thresholds['max_inlier_accept_cost']:.3f}m, "
                        f"min_inliers={thresholds['min_inliers']} ({thresholds['min_inlier_ratio']*100:.0f}%), "
                        f"inlier_cost_th={thresholds['inlier_cost_thresh']:.3f}m"
                    )
            except Exception as ex:
                self.synced_from_node = False
                self.get_logger().warn(f"Failed to query parameters from {node_name}: {ex}")

        def on_parameter_event(self, event: ParameterEvent):
            if "localization" not in event.node:
                return
            updated = False
            with lock:
                for p in list(event.new_parameters) + list(event.changed_parameters):
                    if p.name == "matching_params.cost_threshold":
                        thresholds["cost_threshold_3d"] = float(p.value.double_value)
                        updated = True
                    elif p.name in param_names:
                        short_key = p.name.split(".")[-1]
                        if p.value.type == ParameterType.PARAMETER_DOUBLE:
                            thresholds[short_key] = float(p.value.double_value)
                            updated = True
                        elif p.value.type == ParameterType.PARAMETER_INTEGER:
                            thresholds[short_key] = int(p.value.integer_value)
                            updated = True
                if updated:
                    thresholds_dirty[0] = True
            if updated:
                self.get_logger().info(f"Updated plot thresholds from live parameter event on {event.node}")

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
    hline_th_3d = ax0.axhline(
        thresholds["cost_threshold_3d"],
        color="tab:red",
        linestyle=":",
        label=f"cost_threshold_3d ({thresholds['cost_threshold_3d']:.3f}m)",
    )
    hline_th_2d = ax0.axhline(
        thresholds["cost_threshold_2d"],
        color="tab:brown",
        linestyle=":",
        label=f"cost_threshold_2d ({thresholds['cost_threshold_2d']:.3f}m)",
    )
    hline_max_inlier_cost = ax0.axhline(
        thresholds["max_inlier_accept_cost"],
        color="tab:purple",
        linestyle="-.",
        alpha=0.7,
        label=f"max_inlier_accept_cost ({thresholds['max_inlier_accept_cost']:.3f}m)",
    )
    ax0.set_ylabel("Cost [m]")
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="upper right", ncol=3, fontsize=8)
    ax0.set_ylim(0.0, 0.40)

    # 2. Inlier Metrics
    ax1 = axes[1]
    ax1_twin = ax1.twinx()
    line_inlier_ratio, = ax1.plot([], [], label="Inlier Ratio (%)", color="tab:green", linewidth=1.5)
    hline_inlier_ratio = ax1.axhline(
        thresholds["min_inlier_ratio"] * 100.0,
        color="darkgreen",
        linestyle=":",
        label=f"min_inlier_ratio ({thresholds['min_inlier_ratio']*100.0:.1f}%)",
    )
    line_inlier_cost, = ax1_twin.plot([], [], label="Inlier Cost [m]", color="tab:olive", linestyle="-.", alpha=0.9)
    hline_inlier_cost = ax1_twin.axhline(
        thresholds["inlier_cost_thresh"],
        color="gray",
        linestyle=":",
        label=f"inlier_cost_thresh ({thresholds['inlier_cost_thresh']:.3f}m)",
    )
    ax1.set_ylabel("Inlier Ratio [%]")
    ax1_twin.set_ylabel("Inlier Cost [m]")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_ylim(0.0, 100.0)
    ax1_twin.set_ylim(0.0, 0.08)
    ax1_lines = [line_inlier_ratio, hline_inlier_ratio, line_inlier_cost, hline_inlier_cost]
    ax1.legend(ax1_lines, [l.get_label() for l in ax1_lines], loc="upper right", ncol=2, fontsize=8)

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
    hline_min_inliers = ax3.axhline(
        thresholds["min_inliers"],
        color="darkgreen",
        linestyle=":",
        alpha=0.8,
        label=f"min_inliers ({thresholds['min_inliers']})",
    )
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Points")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", ncol=5, fontsize=8)

    plt.tight_layout()

    def update_plot(_frame):
        artists = (
            line_best, line_3d, line_2d, line_inlier_ratio, line_inlier_cost,
            line_speed, line_range_xy, line_range_yaw, line_obs_tot,
            line_obs_3d, line_obs_2d, line_inliers
        )
        try:
            with lock:
                if thresholds_dirty[0]:
                    th_3d = thresholds["cost_threshold_3d"]
                    th_2d = thresholds["cost_threshold_2d"]
                    th_max_ic = thresholds["max_inlier_accept_cost"]
                    th_ir_pct = thresholds["min_inlier_ratio"] * 100.0
                    th_ic = thresholds["inlier_cost_thresh"]
                    th_min_inl = thresholds["min_inliers"]

                    hline_th_3d.set_ydata([th_3d, th_3d])
                    hline_th_3d.set_label(f"cost_threshold_3d ({th_3d:.3f}m)")
                    hline_th_2d.set_ydata([th_2d, th_2d])
                    hline_th_2d.set_label(f"cost_threshold_2d ({th_2d:.3f}m)")
                    hline_max_inlier_cost.set_ydata([th_max_ic, th_max_ic])
                    hline_max_inlier_cost.set_label(f"max_inlier_accept_cost ({th_max_ic:.3f}m)")
                    ax0.legend(loc="upper right", ncol=3, fontsize=8)

                    hline_inlier_ratio.set_ydata([th_ir_pct, th_ir_pct])
                    hline_inlier_ratio.set_label(f"min_inlier_ratio ({th_ir_pct:.1f}%)")
                    hline_inlier_cost.set_ydata([th_ic, th_ic])
                    hline_inlier_cost.set_label(f"inlier_cost_thresh ({th_ic:.3f}m)")
                    ax1.legend(ax1_lines, [l.get_label() for l in ax1_lines], loc="upper right", ncol=2, fontsize=8)

                    hline_min_inliers.set_ydata([th_min_inl, th_min_inl])
                    hline_min_inliers.set_label(f"min_inliers ({th_min_inl})")
                    ax3.legend(loc="upper right", ncol=5, fontsize=8)

                    thresholds_dirty[0] = False

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
                current_th = dict(thresholds)
            plot_static(records, thresholds=current_th, save_path=save_path)
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
    parser.add_argument(
        "--params-file",
        type=str,
        action="append",
        default=None,
        help="Optional YAML parameter file(s) to override threshold lines",
    )
    parser.add_argument(
        "--node",
        type=str,
        default=None,
        help="Target localization node name for live parameter sync",
    )
    args = parser.parse_args()

    thresholds = init_thresholds(args.params_file)

    if args.csv:
        run_csv_mode(args.csv, thresholds=thresholds, save_path=args.save)
    else:
        run_live_node(
            args.topic,
            thresholds=thresholds,
            buffer_len=args.buffer,
            save_path=args.save,
            window_sec=args.window,
            target_node_name=args.node,
        )


if __name__ == "__main__":
    main()

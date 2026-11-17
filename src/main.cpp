// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include <exception>
#include <rclcpp/rclcpp.hpp>
#include "gn10_pointcloud_localization/localization_node.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    std::shared_ptr<LocalizationNode> node;
    try {
        node = std::make_shared<LocalizationNode>();
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("gn10_localization_node"), "Initialization failed: %s", e.what());
        rclcpp::shutdown();
        return 1;
    }

    // Use a multi-threaded executor so that telemetry/status publishing,
    // 2D LiDAR deskewing, and high-frequency 3D LiDAR/IMU callbacks 
    // run independently without starvation or mutual latency spikes.
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
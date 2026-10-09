// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include <rclcpp/rclcpp.hpp>
#include "gn10_pointcloud_localization/localization_node.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocalizationNode>());
    rclcpp::shutdown();
    return 0;
}
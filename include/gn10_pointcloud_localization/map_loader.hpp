// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>

#include "gn10_pointcloud_localization/cuda/field_objects.cuh"

class MapLoader
{
public:
    // Load analytic field objects from a JSON file
    static std::vector<FieldObject> loadFromJSON(const std::string& file_path);

    // Load analytic field objects from ROS 2 parameter string array
    static std::vector<FieldObject> loadFromParams(const std::vector<std::string>& param_strings);

    // Hardcoded standard NHK 2026 field map (fallback when no external map is provided)
    static std::vector<FieldObject> createNHK2026FieldMap();
};
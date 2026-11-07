// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "gn10_pointcloud_localization/esdf_map.hpp"

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " <input.esdf> <output_prefix> [z_min] [z_max] [robot_radius] [inflation_margin]"
            << std::endl;
        std::cerr << "Example: " << argv[0] << " map.esdf /tmp/costmap 0.05 1.40 0.35 0.50"
                  << std::endl;
        return 1;
    }

    std::string esdf_path     = argv[1];
    std::string output_prefix = argv[2];
    float z_min               = (argc >= 4) ? std::stof(argv[3]) : 0.05f;
    float z_max               = (argc >= 5) ? std::stof(argv[4]) : 1.40f;
    float robot_radius        = (argc >= 6) ? std::stof(argv[5]) : 0.35f;
    float inflation_margin    = (argc >= 7) ? std::stof(argv[6]) : 0.50f;

    std::cout << "[ESDF->Costmap] Loading ESDF from " << esdf_path << "..." << std::endl;
    ESDFMap map;
    if (!map.loadBinary(esdf_path)) {
        std::cerr << "[ESDF->Costmap] Failed to load ESDF file: " << esdf_path << std::endl;
        return 1;
    }

    const auto& h = map.header();
    std::cout << "[ESDF->Costmap] ESDF dimensions: " << h.size_x << " x " << h.size_y << " x "
              << h.size_z << " (res=" << h.resolution << "m)" << std::endl;
    std::cout << "[ESDF->Costmap] Slicing range z=[" << z_min << ", " << z_max << "]m, "
              << "robot_radius=" << robot_radius << "m, margin=" << inflation_margin << "m..."
              << std::endl;

    int width = 0, height = 0;
    float origin_x = 0.0f, origin_y = 0.0f;
    std::vector<int8_t> costmap_data;

    if (!map.sliceTo2DCostmap(
            z_min,
            z_max,
            robot_radius,
            inflation_margin,
            width,
            height,
            origin_x,
            origin_y,
            costmap_data
        )) {
        std::cerr << "[ESDF->Costmap] Failed to slice ESDF to 2D costmap." << std::endl;
        return 1;
    }

    std::cout << "[ESDF->Costmap] 2D Costmap generated: " << width << " x " << height << " cells."
              << std::endl;

    // 1. Save PGM image file (P5 binary: 0=black (occupied/100), 254=white (free/0))
    std::string pgm_path = output_prefix + ".pgm";
    std::ofstream pgm_file(pgm_path, std::ios::binary);
    if (!pgm_file.is_open()) {
        std::cerr << "[ESDF->Costmap] Failed to open " << pgm_path << " for writing." << std::endl;
        return 1;
    }

    pgm_file << "P5\n" << width << " " << height << "\n255\n";
    // ROS map server convention stores rows from top to bottom (Y axis inverted)
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            int8_t c = costmap_data[static_cast<size_t>(y) * width + x];
            uint8_t pixel;
            if (c < 0) {
                pixel = 205;  // Unknown (gray)
            } else if (c == 0) {
                pixel = 254;  // Free space (white)
            } else if (c >= 100) {
                pixel = 0;  // Lethal obstacle (black)
            } else {
                // Inflated cost 1..99 -> grayscale 253..1
                pixel = static_cast<uint8_t>(std::clamp(254 - (c * 254 / 100), 1, 253));
            }
            pgm_file.put(static_cast<char>(pixel));
        }
    }
    pgm_file.close();
    std::cout << "[ESDF->Costmap] Saved map image to " << pgm_path << std::endl;

    // 2. Save YAML metadata file
    std::string yaml_path = output_prefix + ".yaml";
    std::ofstream yaml_file(yaml_path);
    if (!yaml_file.is_open()) {
        std::cerr << "[ESDF->Costmap] Failed to open " << yaml_path << " for writing." << std::endl;
        return 1;
    }

    yaml_file << "image: " << pgm_path << "\n";
    yaml_file << "resolution: " << h.resolution << "\n";
    yaml_file << "origin: [" << origin_x << ", " << origin_y << ", 0.0]\n";
    yaml_file << "negate: 0\n";
    yaml_file << "occupied_thresh: 0.65\n";
    yaml_file << "free_thresh: 0.196\n";
    yaml_file.close();
    std::cout << "[ESDF->Costmap] Saved map metadata to " << yaml_path << std::endl;

    return 0;
}

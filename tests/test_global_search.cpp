// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "gn10_pointcloud_localization/esdf_map.hpp"
#include "gn10_pointcloud_localization/global_searcher.hpp"

void testGlobalSearchDownsampledThreshold()
{
    std::cout << "[Test] testGlobalSearchDownsampledThreshold starting..." << std::endl;

    FieldObject wall{};
    wall.type     = ObjectType::BOX;
    wall.center_x = 0.0f;
    wall.center_y = 0.0f;
    wall.z_min    = 0.0f;
    wall.z_max    = 1.0f;
    wall.param1   = 0.05f;
    wall.param2   = 2.0f;

    ESDFMap esdf;
    assert(esdf.buildFromFieldObjects({wall}, 0.05f, -2.0f, 2.0f, -2.0f, 2.0f, -0.2f, 1.5f, 0.50f));

    PoseSolver solver(1000);
    solver.setESDFMap(esdf);

    GlobalSearchConfig search_cfg;
    search_cfg.range_min_x       = -1.0f;
    search_cfg.range_max_x       = 1.0f;
    search_cfg.range_min_y       = -1.0f;
    search_cfg.range_max_y       = 1.0f;
    search_cfg.step_xy           = 0.25f;
    search_cfg.step_yaw          = 0.25f;
    search_cfg.downsample_stride = 2;
    search_cfg.top_k_candidates  = 2;

    GlobalSearcher searcher(search_cfg, rclcpp::get_logger("test_global_search"));

    GroundFilterParams gf_params{};
    gf_params.range_max        = 5.0f;
    gf_params.robot_radius     = 0.01f;
    gf_params.robot_height_min = 0.0f;
    gf_params.robot_height_max = 1.0f;
    gf_params.ground_z_thresh  = 0.05f;

    MatchingParams match_params{};
    match_params.range_xy           = 0.30f;
    match_params.step_xy            = 0.10f;
    match_params.range_yaw          = 0.20f;
    match_params.step_yaw           = 0.10f;
    match_params.max_dist_thresh    = 0.50f;
    match_params.cost_threshold_3d  = 0.23f;
    match_params.cost_threshold_2d  = 0.23f;
    match_params.min_3d_points      = 200;
    match_params.min_2d_points      = 50;
    match_params.fine_refine        = true;
    match_params.fine_refine_levels = 1;
    match_params.useESDFBounds(esdf.header());

    const float identity_tf[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    // 1. 250 full-res 3D points with stride=2 -> 125 downsampled points >= (200 / 2 = 100).
    // Must NOT abort (previously aborted because 125 <= 200).
    std::vector<float> cloud_250;
    cloud_250.reserve(250 * 3);
    for (int i = 0; i < 250; ++i) {
        cloud_250.push_back(0.05f);
        cloud_250.push_back(-1.0f + 2.0f * (static_cast<float>(i) / 250.0f));
        cloud_250.push_back(0.5f);
    }

    float best_cost = std::numeric_limits<float>::max();
    PoseCandidate pose = searcher.search(
        solver, cloud_250, identity_tf, gf_params, match_params, 0.0f, 0.2f, best_cost
    );
    assert(best_cost < match_params.cost_threshold_3d);
    assert(std::abs(pose.x) < 0.2f);

    // 2. 100 full-res 3D points with stride=2 -> 50 downsampled points < 100, and no 2D points.
    // Must abort cleanly.
    std::vector<float> cloud_100(cloud_250.begin(), cloud_250.begin() + 100 * 3);
    best_cost = 0.0f;
    searcher.search(
        solver, cloud_100, identity_tf, gf_params, match_params, 0.0f, 0.2f, best_cost
    );
    assert(best_cost == std::numeric_limits<float>::max());

    // 3. Same 100 3D points (50 downsampled < 100), but 60 2D points (>= min_2d_points=50).
    // Must NOT abort and should find a low-cost pose using 2D coverage.
    std::vector<float> cloud_2d;
    cloud_2d.reserve(60 * 3);
    for (int i = 0; i < 60; ++i) {
        cloud_2d.push_back(0.05f);
        cloud_2d.push_back(-1.0f + 2.0f * (static_cast<float>(i) / 60.0f));
        cloud_2d.push_back(0.2f);
    }
    best_cost = std::numeric_limits<float>::max();
    pose = searcher.search(
        solver, cloud_100, identity_tf, gf_params, match_params, 0.0f, 0.2f, best_cost,
        nullptr, nullptr, &cloud_2d
    );
    assert(best_cost < match_params.cost_threshold_2d);
    assert(std::abs(pose.x) < 0.2f);

    std::cout << "[Test] testGlobalSearchDownsampledThreshold PASSED!" << std::endl;
}

int main()
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "No CUDA device available. Skipping global search test." << std::endl;
        return 77;
    }

    testGlobalSearchDownsampledThreshold();
    std::cout << "All global search tests PASSED successfully!" << std::endl;
    return 0;
}

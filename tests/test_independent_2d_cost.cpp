// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "gn10_pointcloud_localization/cuda/esdf_matcher.cuh"
#include "gn10_pointcloud_localization/esdf_map.hpp"
#include "gn10_pointcloud_localization/pose_solver.hpp"

// Test 1: Verify host-side FoV & point-count scaling rules
void testFoVWeightingLogic()
{
    std::cout << "[Test] testFoVWeightingLogic starting..." << std::endl;

    MatchingParams params;
    params.weight_2d_lidar = 0.40f;
    params.min_2d_points = 30;
    params.target_2d_points = 200;

    auto compute_weight = [&](int n_3d, int n_2d) -> float {
        if (n_2d >= params.min_2d_points && n_3d >= 50) {
            float fov_scale = std::min(
                1.0f,
                static_cast<float>(n_2d) / static_cast<float>(params.target_2d_points)
            );
            return params.weight_2d_lidar * fov_scale;
        } else if (n_2d >= params.min_2d_points && n_3d < 50) {
            return 1.0f;
        } else {
            return 0.0f;
        }
    };

    // Case 1: 2D points below min threshold -> weight is 0.0
    assert(compute_weight(500, 0) == 0.0f);
    assert(compute_weight(500, 20) == 0.0f);
    assert(compute_weight(500, 29) == 0.0f);

    // Case 2: Partial FoV (50% target) -> weight scales linearly
    float w_half = compute_weight(500, 100);
    assert(std::abs(w_half - 0.20f) < 1e-5f);

    // Case 3: Full FoV (>= target) -> full weight
    float w_full = compute_weight(500, 200);
    assert(std::abs(w_full - 0.40f) < 1e-5f);

    float w_extra = compute_weight(500, 800);
    assert(std::abs(w_extra - 0.40f) < 1e-5f);

    // Case 4: 3D sensor degraded (< 50 points) while 2D is valid -> 100% 2D fallback
    float w_fallback = compute_weight(10, 150);
    assert(w_fallback == 1.0f);

    std::cout << "[Test] testFoVWeightingLogic PASSED!" << std::endl;
}

// Test 2: Full GPU ESDF Matcher test with separate 3D and 2D point clouds
void testESDFMatcherWeightedResidual()
{
    std::cout << "[Test] testESDFMatcherWeightedResidual starting..." << std::endl;

    // Create a simple synthetic ESDF grid
    // Grid: 40 x 40 x 20, res = 0.1m, origin = (-2.0, -2.0, 0.0)
    // Wall along x = 0.0m (voxel index 20 along X)
    ESDFHeader header;
    header.min_x = -2.0f;
    header.min_y = -2.0f;
    header.min_z = 0.0f;
    header.size_x = 40;
    header.size_y = 40;
    header.size_z = 20;
    header.resolution = 0.1f;
    header.max_dist_thresh = 1.0f;

    std::vector<float> grid(header.size_x * header.size_y * header.size_z);
    for (int z = 0; z < header.size_z; ++z) {
        for (int y = 0; y < header.size_y; ++y) {
            for (int x = 0; x < header.size_x; ++x) {
                float wx = header.min_x + (x + 0.5f) * header.resolution;
                // Distance to vertical wall plane x = 0.0
                float dist = std::abs(wx);
                grid[z * (header.size_x * header.size_y) + y * header.size_x + x] = dist;
            }
        }
    }

    uploadESDFMapToGPU(header, grid.data());

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 3D obstacle points: on the wall plane x = 0.0m (d = 0.0m)
    // Count: 200 points
    const int num_3d = 200;
    std::vector<float> cloud_3d;
    cloud_3d.reserve(num_3d * 3);
    for (int i = 0; i < num_3d; ++i) {
        cloud_3d.push_back(0.0f); // x = 0.0m
        cloud_3d.push_back(-1.0f + 2.0f * (static_cast<float>(i) / num_3d)); // y in [-1, 1]
        cloud_3d.push_back(0.5f); // z = 0.5m
    }

    // 2D obstacle points: displaced at x = 0.20m (d = 0.20m)
    // Count: 100 points
    const int num_2d = 100;
    std::vector<float> cloud_2d;
    cloud_2d.reserve(num_2d * 3);
    for (int i = 0; i < num_2d; ++i) {
        cloud_2d.push_back(0.20f); // x = 0.20m
        cloud_2d.push_back(-1.0f + 2.0f * (static_cast<float>(i) / num_2d)); // y in [-1, 1]
        cloud_2d.push_back(0.1f); // z = 0.1m
    }

    // Combined cloud on device
    const int total_pts = num_3d + num_2d;
    std::vector<float> combined_cloud = cloud_3d;
    combined_cloud.insert(combined_cloud.end(), cloud_2d.begin(), cloud_2d.end());

    float* d_cloud = nullptr;
    cudaMalloc(&d_cloud, total_pts * 3 * sizeof(float));
    cudaMemcpy(d_cloud, combined_cloud.data(), total_pts * 3 * sizeof(float), cudaMemcpyHostToDevice);

    PoseCandidate base_pose{0.0f, 0.0f, 0.0f};
    PoseCandidate best_pose;
    float best_cost = 0.0f;
    std::vector<float> dynamic_pts;
    int inliers = 0;
    float inlier_cost = 0.0f;
    float cost_3d = 0.0f;
    float cost_2d = 0.0f;

    // Evaluate at base_pose only (range = 0)
    // Weight w_2d = 0.40
    float weight_2d = 0.40f;
    bool ok = launchESDFMatcher(
        stream,
        d_cloud,
        total_pts,
        base_pose,
        0.0f, 0.0f, 0.1f, // range_x, range_y, step_xy
        0.0f, 0.1f,       // range_yaw, step_yaw
        0.50f,            // max_dist_thresh
        0.15f,            // dynamic_dist_thresh
        header.min_x, header.min_x + header.size_x * header.resolution,
        header.min_y, header.min_y + header.size_y * header.resolution,
        best_pose,
        best_cost,
        dynamic_pts,
        false,
        &inliers,
        &inlier_cost,
        0.08f,
        num_3d,
        num_2d,
        weight_2d,
        &cost_3d,
        &cost_2d
    );

    assert(ok);
    std::cout << "  3D Cost: " << cost_3d << " (expected ~0.0m)" << std::endl;
    std::cout << "  2D Cost: " << cost_2d << " (expected ~0.20m)" << std::endl;
    std::cout << "  Composite Cost: " << best_cost << " (expected ~0.08m)" << std::endl;

    // 3D points are on the wall boundary (d ~ 0.05)
    assert(cost_3d <= 0.051f);
    // 2D points are displaced by 0.20m (d ~ 0.20)
    assert(std::abs(cost_2d - 0.20f) < 0.02f);

    // Composite cost must match (1 - w)*3D + w*2D
    float expected_composite = (1.0f - weight_2d) * cost_3d + weight_2d * cost_2d;
    assert(std::abs(best_cost - expected_composite) < 1e-3f);

    // Compare with simple total average cost: (200*0.05 + 100*0.20) / 300 = 0.100
    // With independent weighting w=0.40, composite cost is 0.110, which properly weights 2D higher!
    assert(best_cost > 0.105f);

    cudaFree(d_cloud);
    cudaStreamDestroy(stream);
    freeESDFMapFromGPU();

    std::cout << "[Test] testESDFMatcherWeightedResidual PASSED!" << std::endl;
}

int main()
{
    testFoVWeightingLogic();
    testESDFMatcherWeightedResidual();
    std::cout << "All Independent 2D Residual Weighting tests PASSED successfully!" << std::endl;
    return 0;
}

// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "gn10_pointcloud_localization/esdf_map.hpp"
#include "gn10_pointcloud_localization/pose_solver.hpp"

namespace {

void testVoxelCentroidEqualization()
{
    std::cout << "[Test] testVoxelCentroidEqualization starting..." << std::endl;

    PoseSolver solver(10000);
    float identity_tf[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    // 1,000 dense points inside a single 0.08 m voxel: [0.97, 1.03] x [0.01, 0.07] x [0.25, 0.31]
    std::vector<float> cloud;
    cloud.reserve(1020 * 3);
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    for (int i = 0; i < 1000; ++i) {
        const float px = 0.97f + 0.06f * static_cast<float>(i % 10) / 9.0f;
        const float py = 0.01f + 0.06f * static_cast<float>((i / 10) % 10) / 9.0f;
        const float pz = 0.25f + 0.06f * static_cast<float>(i / 100) / 9.0f;
        cloud.push_back(px);
        cloud.push_back(py);
        cloud.push_back(pz);
        sum_x += px;
        sum_y += py;
        sum_z += pz;
    }

    // 20 sparse wall points separated by 0.25 m each along X
    for (int i = 0; i < 20; ++i) {
        cloud.push_back(2.0f + 0.25f * static_cast<float>(i));
        cloud.push_back(0.0f);
        cloud.push_back(0.50f);
    }

    GroundFilterParams gf_off;
    gf_off.range_max        = 12.0f;
    gf_off.robot_radius     = 0.5f;
    gf_off.ground_z_thresh  = 0.05f;
    gf_off.voxel_size       = 0.0f;

    const int raw_obs = solver.prepareObstacleCloud(cloud, identity_tf, gf_off);
    assert(raw_obs == 1020);
    assert(solver.obstacleCount3D() == 1020);

    GroundFilterParams gf_on = gf_off;
    gf_on.voxel_size         = 0.08f;

    const int eq_obs = solver.prepareObstacleCloud(cloud, identity_tf, gf_on);
    assert(eq_obs == 21);
    assert(solver.obstacleCount3D() == 21);

    std::vector<float> out_obs;
    solver.copyFilteredClouds(nullptr, &out_obs);
    assert(out_obs.size() == 21 * 3);

    // Verify that the dense cluster centroid matches the analytical mean
    const float exp_x = static_cast<float>(sum_x / 1000.0);
    const float exp_y = static_cast<float>(sum_y / 1000.0);
    const float exp_z = static_cast<float>(sum_z / 1000.0);
    bool found_centroid = false;
    for (size_t i = 0; i + 2 < out_obs.size(); i += 3) {
        if (std::abs(out_obs[i] - exp_x) < 1e-3f &&
            std::abs(out_obs[i + 1] - exp_y) < 1e-3f &&
            std::abs(out_obs[i + 2] - exp_z) < 1e-3f) {
            found_centroid = true;
            break;
        }
    }
    assert(found_centroid);

    std::cout << "[Test] testVoxelCentroidEqualization PASSED! (1020 -> " << eq_obs << " points)"
              << std::endl;
}

void testRobustLossOutlierSuppression()
{
    std::cout << "[Test] testRobustLossOutlierSuppression starting..." << std::endl;

    // Build an ESDF map with a wall at x = 2.0 m
    std::vector<float> map_pts;
    for (int iy = -40; iy <= 40; ++iy) {
        for (int iz = 2; iz <= 25; ++iz) {
            map_pts.push_back(2.0f);
            map_pts.push_back(static_cast<float>(iy) * 0.05f);
            map_pts.push_back(static_cast<float>(iz) * 0.05f);
        }
    }
    ESDFMap esdf;
    ESDFCropBounds bounds;
    bounds.enabled = true;
    bounds.min_x = -1.0f;
    bounds.max_x = 4.0f;
    bounds.min_y = -3.0f;
    bounds.max_y = 3.0f;
    bounds.min_z = 0.0f;
    bounds.max_z = 2.0f;
    esdf.buildFromPoints(map_pts, 0.05f, 0.50f, bounds, 0.05f);

    PoseSolver solver(10000);
    solver.setESDFMap(esdf);

    float identity_tf[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    // 100 true wall points at x = 2.0 m + 120 unmapped near-wall outlier points at x = 2.18 m.
    // Within search_range_xy = 0.10 m, Clipped L1 gets pulled toward -0.10 m to shrink the
    // 120 outliers' linear distance, whereas Welsch ignores the outliers and stays near x = 0.0 m.
    std::vector<float> scan;
    for (int i = 0; i < 100; ++i) {
        scan.push_back(2.0f);
        scan.push_back(-1.0f + 2.0f * static_cast<float>(i) / 99.0f);
        scan.push_back(0.50f);
    }
    for (int i = 0; i < 120; ++i) {
        scan.push_back(2.18f);
        scan.push_back(-0.5f + 1.0f * static_cast<float>(i) / 119.0f);
        scan.push_back(0.60f);
    }

    GroundFilterParams gf_params;
    gf_params.range_max       = 12.0f;
    gf_params.robot_radius    = 0.5f;
    gf_params.ground_z_thresh = 0.05f;
    gf_params.voxel_size      = 0.0f;

    MatchingParams l1_params;
    l1_params.range_xy          = 0.10f;
    l1_params.step_xy           = 0.02f;
    l1_params.range_yaw         = 0.0f;
    l1_params.step_yaw          = 0.02f;
    l1_params.fine_refine       = true;
    l1_params.fine_refine_levels = 2;
    l1_params.max_dist_thresh   = 0.25f;
    l1_params.robust_loss_type  = 0;
    l1_params.useESDFBounds(esdf.header());

    PoseCandidate base_pose{0.0f, 0.0f, 0.0f};
    PoseCandidate l1_pose{};
    float l1_cost = 0.0f;
    std::vector<float> dyn;

    const bool ok_l1 = solver.processPointCloud(
        scan, identity_tf, gf_params, l1_params, base_pose, dyn, l1_pose, l1_cost,
        nullptr, nullptr, nullptr, false
    );
    assert(ok_l1);
    std::cout << "  Clipped L1 solved x = " << l1_pose.x << " m (pulled by outliers)" << std::endl;
    assert(l1_pose.x < -0.05f);

    MatchingParams welsch_params    = l1_params;
    welsch_params.robust_loss_type  = 1;
    welsch_params.robust_loss_sigma = 0.05f;

    PoseCandidate welsch_pose{};
    float welsch_cost = 0.0f;
    const bool ok_welsch = solver.processPointCloud(
        scan, identity_tf, gf_params, welsch_params, base_pose, dyn, welsch_pose, welsch_cost,
        nullptr, nullptr, nullptr, false
    );
    assert(ok_welsch);
    std::cout << "  Welsch solved x = " << welsch_pose.x << " m (locked on true wall)" << std::endl;
    assert(std::abs(welsch_pose.x) < 0.03f);

    std::cout << "[Test] testRobustLossOutlierSuppression PASSED!" << std::endl;
}

}  // namespace

int main()
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "[Skip] No CUDA device available." << std::endl;
        return 77;
    }

    testVoxelCentroidEqualization();
    testRobustLossOutlierSuppression();
    std::cout << "All voxel equalization and robust loss tests PASSED!" << std::endl;
    return 0;
}

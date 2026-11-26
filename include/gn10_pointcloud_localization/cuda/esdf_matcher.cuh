// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cuda_runtime.h>

#include <vector>

#include "gn10_pointcloud_localization/cuda/field_objects.cuh"
#include "gn10_pointcloud_localization/esdf_map.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// Upload and bind the 3D ESDF voxel grid as a hardware-interpolated 3D CUDA texture
void uploadESDFMapToGPU(const ESDFHeader& header, const float* h_grid_data);
void freeESDFMapFromGPU();

// Launch GPU 3D-texture ESDF matcher over a 3D (x, y, yaw) candidate grid
bool launchESDFMatcher(
    cudaStream_t stream,
    const float* d_obstacle_cloud,
    int num_points,
    const PoseCandidate& base_pose,
    float range_x,
    float range_y,
    float step_xy,
    float range_yaw,
    float step_yaw,
    float max_dist_thresh,
    float dynamic_dist_thresh,
    float field_min_x,
    float field_max_x,
    float field_min_y,
    float field_max_y,
    PoseCandidate& out_best_pose,
    float& out_best_cost,
    std::vector<float>& out_dynamic_pts,
    bool extract_dynamic     = true,
    int* out_inlier_count    = nullptr,
    float* out_inlier_cost   = nullptr,
    float inlier_dist_thresh = 0.08f,
    int num_3d_points        = -1,
    int num_2d_points        = 0,
    float weight_2d          = 0.0f,
    float* out_cost_3d       = nullptr,
    float* out_cost_2d       = nullptr,
    std::vector<float>* out_all_costs = nullptr,
    int point_stride         = 1
);

#ifdef __cplusplus
}
#endif

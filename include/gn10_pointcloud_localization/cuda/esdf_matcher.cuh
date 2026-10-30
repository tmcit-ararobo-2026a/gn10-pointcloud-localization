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

// ESDF 3D テクスチャのアップロード & バインド
void uploadESDFMapToGPU(const ESDFHeader& header, const float* h_grid_data);
void freeESDFMapFromGPU();

// 3D Texture を使用した ESDF マッチャー
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
    float inlier_dist_thresh = 0.08f
);

// 確定姿勢を用いた動的点群の高速抽出 (Pose探索をスキップ)
bool launchESDFDynamicFilterOnly(
    cudaStream_t stream,
    const float* d_cloud,
    int num_points,
    const PoseCandidate& confirmed_pose,
    float dynamic_dist_thresh,
    float field_min_x,
    float field_max_x,
    float field_min_y,
    float field_max_y,
    std::vector<float>& out_dynamic_pts,
        const float* tracking_sphere = nullptr
);

#ifdef __cplusplus
}
#endif

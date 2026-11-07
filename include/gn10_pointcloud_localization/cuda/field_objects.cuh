// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

enum ObjectType { CYLINDER, BOX, VISUAL_BOX };

struct FieldObject {
    ObjectType type;
    float center_x, center_y;
    float z_min, z_max;
    float param1, param2;

    // Convenience accessors for cylinder radius and box half-extents
    __host__ __device__ float radius() const { return param1; }
    __host__ __device__ float half_x() const { return param1; }
    __host__ __device__ float half_y() const { return param2; }
    __host__ __device__ void set_radius(float r)
    {
        param1 = r;
        param2 = 0.0f;
    }
    __host__ __device__ void set_half_sizes(float hx, float hy)
    {
        param1 = hx;
        param2 = hy;
    }
};

struct PoseCandidate {
    float x, y, yaw;
};

/**
 * @brief 3D grid definition for GPU pose candidate generation without H2D candidate transfers.
 */
struct SearchGrid {
    float base_x;
    float base_y;
    float base_yaw;
    float step_xy;
    float step_yaw;
    int nx;
    int ny;
    int nyaw;

    /**
     * @brief Returns total number of candidate poses in the 3D grid.
     */
    __host__ __device__ inline int totalCandidates() const
    {
        return nx * ny * nyaw;
    }

    /**
     * @brief Reconstructs candidate pose (rx, ry, ryaw) from a 1D linear index.
     */
    __host__ __device__ inline PoseCandidate indexToPose(int pose_idx) const
    {
        const int iyaw = pose_idx % nyaw;
        const int temp = pose_idx / nyaw;
        const int iy   = temp % ny;
        const int ix   = temp / ny;
        return PoseCandidate{
            base_x + static_cast<float>(ix) * step_xy,
            base_y + static_cast<float>(iy) * step_xy,
            base_yaw + static_cast<float>(iyaw) * step_yaw
        };
    }
};

/**
 * @brief Constructs a SearchGrid from a center pose, half-ranges, and step sizes.
 */
inline SearchGrid createSearchGrid(
    const PoseCandidate& base_pose,
    float range_x,
    float range_y,
    float step_xy,
    float range_yaw,
    float step_yaw
)
{
    SearchGrid grid{};
    grid.base_x   = base_pose.x - range_x;
    grid.base_y   = base_pose.y - range_y;
    grid.base_yaw = base_pose.yaw - range_yaw;
    grid.step_xy  = step_xy;
    grid.step_yaw = step_yaw;
    if (step_xy > 0.0f && step_yaw > 0.0f && range_x >= 0.0f && range_y >= 0.0f &&
        range_yaw >= 0.0f) {
        grid.nx   = static_cast<int>(std::floor((2.0f * range_x + 1e-5f) / step_xy)) + 1;
        grid.ny   = static_cast<int>(std::floor((2.0f * range_y + 1e-5f) / step_xy)) + 1;
        grid.nyaw = static_cast<int>(std::floor((2.0f * range_yaw + 1e-5f) / step_yaw)) + 1;
    } else {
        grid.nx   = 0;
        grid.ny   = 0;
        grid.nyaw = 0;
    }
    return grid;
}

#ifdef __cplusplus
extern "C" {
#endif

void uploadFieldMapToGPU(const std::vector<FieldObject>& host_map);

bool launchFieldSDFMatcher(
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
    bool extract_dynamic     = true,  // If false, skips the dynamic point extraction kernel and D2H copy
    int* out_inlier_count    = nullptr,
    float* out_inlier_cost   = nullptr,
    float inlier_dist_thresh = 0.08f,
    std::vector<float>* out_all_costs = nullptr,
    int point_stride         = 1
);

#ifdef __cplusplus
}
#endif

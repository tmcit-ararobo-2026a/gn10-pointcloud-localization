// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/**
 * @brief Point structure with per-point nanosecond timestamp for deskewing.
 */
struct TimedScanPoint {
    float x;
    float y;
    float z;
    int64_t stamp_ns;
};

/**
 * @brief Integrated IMU yaw sample structure for GPU binary-search interpolation.
 */
struct IMUSample {
    int64_t stamp_ns;
    float rate;
    float angle;
};

/**
 * @brief Point cloud motion-distortion compensation (deskew) parameters.
 */
struct DeskewParams {
    bool enabled{true};
    int64_t reference_ns{0};
    int64_t min_stamp_ns{0};
    float ref_yaw{0.0f};
    float vx_world{0.0f};
    float vy_world{0.0f};
    float ref_imu_angle{0.0f};
};

/**
 * @brief Launches the standard CUDA ground and footprint filter kernel.
 */
void launchGroundFilter(
    cudaStream_t stream,
    const float* d_in,
    float* d_ground,
    float* d_obstacle,
    const float* d_transform,
    int num_points,
    int max_points,
    float range_max,
    float robot_radius,
    float robot_height_min,
    float robot_height_max,
    float ground_z_thresh,
    int* d_ground_count,
    int* d_obstacle_count,
    int* h_ground_count,
    int* h_obstacle_count
);

/**
 * @brief Launches the fused CUDA kernel that performs static TF transform, IMU rotational +
 *        velocity translational deskew, self-filter, and ground/obstacle classification.
 */
void launchGroundFilterWithDeskew(
    cudaStream_t stream,
    const TimedScanPoint* d_in_timed,
    float* d_ground,
    float* d_obstacle,
    const float* d_transform,
    int num_points,
    int stride,
    int max_points,
    const DeskewParams& deskew_params,
    const IMUSample* d_imu_samples,
    int num_imu_samples,
    float range_max,
    float robot_radius,
    float robot_height_min,
    float robot_height_max,
    float ground_z_thresh,
    int* d_ground_count,
    int* d_obstacle_count,
    int* h_ground_count,
    int* h_obstacle_count
);

/**
 * @brief Performs GPU-side 3D voxel centroid equalization on an obstacle point cloud
 *        using an open-addressing hash table to eliminate near-field 1/r^2 density bias.
 *
 * @param stream CUDA stream for asynchronous execution.
 * @param d_points Device buffer of XYZ points (modified in-place with equalized centroids).
 * @param num_points Number of input obstacle points in d_points.
 * @param max_points Capacity of d_points (in points).
 * @param voxel_size Edge length of each 3D voxel [m]. Values <= 0.0f disable equalization.
 * @param d_hash_keys Pre-allocated device buffer of hash_capacity keys.
 * @param d_hash_sums Pre-allocated device buffer of hash_capacity * 3 coordinate sums.
 * @param d_hash_counts Pre-allocated device buffer of hash_capacity point counts.
 * @param hash_capacity Power-of-two capacity of the open-addressing hash table.
 * @param d_out_count Device scalar int receiving compacted output point count.
 * @param h_out_count Host scalar int receiving compacted output point count after sync.
 */
void launchVoxelCentroidEqualization(
    cudaStream_t stream,
    float* d_points,
    int num_points,
    int max_points,
    float voxel_size,
    unsigned long long* d_hash_keys,
    float* d_hash_sums,
    int* d_hash_counts,
    int hash_capacity,
    int* d_out_count,
    int* h_out_count,
    int* d_dropped_count = nullptr,
    int* h_dropped_count = nullptr
);

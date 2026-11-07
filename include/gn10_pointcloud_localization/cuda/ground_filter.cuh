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

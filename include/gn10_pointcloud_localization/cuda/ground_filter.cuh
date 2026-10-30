// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/**
 * @brief タイムスタンプ付き点構造体
 */
struct TimedScanPoint {
    float x;
    float y;
    float z;
    int64_t stamp_ns;
};

/**
 * @brief IMUの角度積分サンプル構造体
 */
struct IMUSample {
    int64_t stamp_ns;
    float rate;
    float angle;
};

/**
 * @brief 点群の歪み補正(Deskew)パラメータ
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
 * @brief 従来のGround Filterカーネル起動関数
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
 * @brief Deskew 座標変換と Ground Filter を統合した CUDA カーネル起動関数
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

// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include "gn10_pointcloud_localization/cuda/ground_filter.cuh"

#include <cuda_runtime.h>
#include <cmath>

/**
 * @brief Standard Ground Filter CUDA kernel (without per-point deskew).
 */
__global__ void filterGroundKernel(
    const float* in_cloud,
    float* out_ground,
    float* out_obstacle,
    const float* transform,  // 3x4 float matrix (R|t)
    int num_points,
    float range_max,
    float robot_radius,
    float robot_height_min,
    float robot_height_max,
    float ground_z_thresh,
    int* ground_count,
    int* obstacle_count
)
{
    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= num_points) {
        return;
    }

    // Point coordinates in sensor frame (livox_frame)
    float xs = in_cloud[point_index * 3 + 0];
    float ys = in_cloud[point_index * 3 + 1];
    float zs = in_cloud[point_index * 3 + 2];

    // Affine transformation into base_link frame
    float xb = transform[0] * xs + transform[1] * ys + transform[2] * zs + transform[3];
    float yb = transform[4] * xs + transform[5] * ys + transform[6] * zs + transform[7];
    float zb = transform[8] * xs + transform[9] * ys + transform[10] * zs + transform[11];

    // Reject points beyond max horizontal detection range from base_link origin
    float dist_sq = xb * xb + yb * yb;
    if (dist_sq > range_max * range_max) {
        return;
    }

    // Remove robot self-returns (cylinder of radius robot_radius and height [robot_height_min, robot_height_max])
    if (dist_sq <= robot_radius * robot_radius) {
        if (zb >= robot_height_min && zb <= robot_height_max) {
            return;
        }
    }

    // Classify ground vs. obstacle points based on base_link Z threshold
    if (zb <= ground_z_thresh) {
        int g_idx = atomicAdd(ground_count, 1);
        out_ground[g_idx * 3 + 0] = xb;
        out_ground[g_idx * 3 + 1] = yb;
        out_ground[g_idx * 3 + 2] = zb;
    } else {
        int o_idx = atomicAdd(obstacle_count, 1);
        out_obstacle[o_idx * 3 + 0] = xb;
        out_obstacle[o_idx * 3 + 1] = yb;
        out_obstacle[o_idx * 3 + 2] = zb;
    }
}

/**
 * @brief Device function to evaluate the relative yaw angle at target_stamp_ns from IMU samples.
 * @param target_stamp_ns Target point timestamp [ns]
 * @param samples Buffered IMU sample array in device memory
 * @param num_samples Number of IMU samples
 * @param ref_angle Integrated IMU yaw angle at the scan reference timestamp [rad]
 * @return Relative yaw rotation angle from the reference timestamp [rad]
 */
__device__ inline float evaluateYawDevice(
    int64_t target_stamp_ns,
    const IMUSample* samples,
    int num_samples,
    float ref_angle
)
{
    if (num_samples <= 0) {
        return 0.0f;
    }

    const IMUSample& first = samples[0];
    const IMUSample& last = samples[num_samples - 1];

    // Linear extrapolation outside sample bounds (clamped to +/- 100 ms to prevent runaway rotation)
    if (target_stamp_ns < first.stamp_ns) {
        float time_diff_s = static_cast<float>(target_stamp_ns - first.stamp_ns) * 1e-9f;
        time_diff_s = fmaxf(-0.10f, fminf(0.10f, time_diff_s));
        float calculated_angle = first.angle + first.rate * time_diff_s;
        return calculated_angle - ref_angle;
    }
    if (target_stamp_ns >= last.stamp_ns) {
        float time_diff_s = static_cast<float>(target_stamp_ns - last.stamp_ns) * 1e-9f;
        time_diff_s = fmaxf(-0.10f, fminf(0.10f, time_diff_s));
        float calculated_angle = last.angle + last.rate * time_diff_s;
        return calculated_angle - ref_angle;
    }

    // Binary search for the bracketing IMU sample interval
    int index_low = 0;
    int index_high = num_samples - 1;
    while (index_low < index_high - 1) {
        int index_mid = (index_low + index_high) / 2;
        if (samples[index_mid].stamp_ns <= target_stamp_ns) {
            index_low = index_mid;
        } else {
            index_high = index_mid;
        }
    }

    // Piecewise quadratic (trapezoidal angular velocity) interpolation
    const IMUSample& sample_low = samples[index_low];
    const IMUSample& sample_high = samples[index_high];
    float dt_s = static_cast<float>(target_stamp_ns - sample_low.stamp_ns) * 1e-9f;
    float interval_s = static_cast<float>(sample_high.stamp_ns - sample_low.stamp_ns) * 1e-9f;
    float calculated_angle = 0.0f;
    if (interval_s > 1e-7f) {
        float rate_diff = sample_high.rate - sample_low.rate;
        calculated_angle = sample_low.angle + sample_low.rate * dt_s + 0.5f * rate_diff * dt_s * dt_s / interval_s;
    } else {
        calculated_angle = sample_low.angle;
    }

    return calculated_angle - ref_angle;
}

/**
 * @brief Fused CUDA kernel combining static TF transform, IMU rotational + translational deskew,
 *        self-filter, and ground/obstacle classification.
 */
__global__ void filterGroundWithDeskewKernel(
    const TimedScanPoint* in_cloud,
    float* out_ground,
    float* out_obstacle,
    const float* transform,
    int num_points,
    int stride,
    DeskewParams deskew,
    const IMUSample* imu_samples,
    int num_imu_samples,
    float range_max,
    float robot_radius,
    float robot_height_min,
    float robot_height_max,
    float ground_z_thresh,
    int* ground_count,
    int* obstacle_count
)
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int point_index = thread_idx * stride;
    if (point_index >= num_points) {
        return;
    }

    const TimedScanPoint& point = in_cloud[point_index];

    // Reject NaN and Inf coordinates
    if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z)) {
        return;
    }

    // Validate point timestamp range against the active accumulation window
    if (deskew.enabled) {
        if (point.stamp_ns > deskew.reference_ns) {
            return;
        }
        if (point.stamp_ns < deskew.min_stamp_ns) {
            return;
        }
    }

    // 1. Static affine transform from sensor frame (livox_frame) to base_link frame
    float xs = point.x;
    float ys = point.y;
    float zs = point.z;

    float xb = transform[0] * xs + transform[1] * ys + transform[2] * zs + transform[3];
    float yb = transform[4] * xs + transform[5] * ys + transform[6] * zs + transform[7];
    float zb = transform[8] * xs + transform[9] * ys + transform[10] * zs + transform[11];

    // 2. Motion-distortion compensation (deskew) using IMU yaw integration and linear velocity
    if (deskew.enabled && point.stamp_ns != deskew.reference_ns) {
        // Interpolate relative yaw rotation angle from IMU samples
        float delta_yaw = evaluateYawDevice(point.stamp_ns, imu_samples, num_imu_samples, deskew.ref_imu_angle);

        // Compute translational displacement in base_link frame
        float dt_sec = static_cast<float>(point.stamp_ns - deskew.reference_ns) * 1e-9f;
        float cos_ref_yaw = cosf(deskew.ref_yaw);
        float sin_ref_yaw = sinf(deskew.ref_yaw);
        float vx_local = cos_ref_yaw * deskew.vx_world + sin_ref_yaw * deskew.vy_world;
        float vy_local = -sin_ref_yaw * deskew.vx_world + cos_ref_yaw * deskew.vy_world;
        float dx = vx_local * dt_sec;
        float dy = vy_local * dt_sec;

        // Apply rotational and translational deskew
        float cos_delta = cosf(delta_yaw);
        float sin_delta = sinf(delta_yaw);
        float x_deskewed = cos_delta * xb - sin_delta * yb + dx;
        float y_deskewed = sin_delta * xb + cos_delta * yb + dy;

        xb = x_deskewed;
        yb = y_deskewed;
    }

    // 3. Reject points outside maximum horizontal detection range
    float dist_sq = xb * xb + yb * yb;
    if (dist_sq > range_max * range_max) {
        return;
    }

    // 4. Reject points inside the robot body footprint cylinder
    if (dist_sq <= robot_radius * robot_radius) {
        if (zb >= robot_height_min && zb <= robot_height_max) {
            return;
        }
    }

    // 5. Partition into ground and obstacle point clouds
    if (zb <= ground_z_thresh) {
        int ground_index = atomicAdd(ground_count, 1);
        out_ground[ground_index * 3 + 0] = xb;
        out_ground[ground_index * 3 + 1] = yb;
        out_ground[ground_index * 3 + 2] = zb;
    } else {
        int obstacle_index = atomicAdd(obstacle_count, 1);
        out_obstacle[obstacle_index * 3 + 0] = xb;
        out_obstacle[obstacle_index * 3 + 1] = yb;
        out_obstacle[obstacle_index * 3 + 2] = zb;
    }
}

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
)
{
    cudaMemsetAsync(d_ground_count, 0, sizeof(int), stream);
    cudaMemsetAsync(d_obstacle_count, 0, sizeof(int), stream);

    int threads_per_block = 256;
    int blocks_per_grid = (num_points + threads_per_block - 1) / threads_per_block;

    filterGroundKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_in,
        d_ground,
        d_obstacle,
        d_transform,
        num_points,
        range_max,
        robot_radius,
        robot_height_min,
        robot_height_max,
        ground_z_thresh,
        d_ground_count,
        d_obstacle_count
    );

    cudaMemcpyAsync(h_ground_count, d_ground_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(
        h_obstacle_count, d_obstacle_count, sizeof(int), cudaMemcpyDeviceToHost, stream
    );
}

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
)
{
    cudaMemsetAsync(d_ground_count, 0, sizeof(int), stream);
    cudaMemsetAsync(d_obstacle_count, 0, sizeof(int), stream);

    int safe_stride = stride;
    if (safe_stride < 1) {
        safe_stride = 1;
    }
    int actual_points = (num_points + safe_stride - 1) / safe_stride;
    if (actual_points <= 0) {
        *h_ground_count = 0;
        *h_obstacle_count = 0;
        return;
    }

    int threads_per_block = 256;
    int blocks_per_grid = (actual_points + threads_per_block - 1) / threads_per_block;

    filterGroundWithDeskewKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_in_timed,
        d_ground,
        d_obstacle,
        d_transform,
        num_points,
        safe_stride,
        deskew_params,
        d_imu_samples,
        num_imu_samples,
        range_max,
        robot_radius,
        robot_height_min,
        robot_height_max,
        ground_z_thresh,
        d_ground_count,
        d_obstacle_count
    );

    cudaMemcpyAsync(h_ground_count, d_ground_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(
        h_obstacle_count, d_obstacle_count, sizeof(int), cudaMemcpyDeviceToHost, stream
    );
}

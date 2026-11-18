// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include "gn10_pointcloud_localization/cuda/ground_filter.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

/**
 * @brief Standard Ground Filter CUDA kernel (without per-point deskew).
 */
__global__ void filterGroundKernel(
    const float* in_cloud,
    float* out_ground,
    float* out_obstacle,
    const float* transform,  // 3x4 float matrix (R|t)
    int num_points,
    int max_points,
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
        if (g_idx < max_points) {
            out_ground[g_idx * 3 + 0] = xb;
            out_ground[g_idx * 3 + 1] = yb;
            out_ground[g_idx * 3 + 2] = zb;
        }
    } else {
        int o_idx = atomicAdd(obstacle_count, 1);
        if (o_idx < max_points) {
            out_obstacle[o_idx * 3 + 0] = xb;
            out_obstacle[o_idx * 3 + 1] = yb;
            out_obstacle[o_idx * 3 + 2] = zb;
        }
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
    int max_points,
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
        if (ground_index < max_points) {
            out_ground[ground_index * 3 + 0] = xb;
            out_ground[ground_index * 3 + 1] = yb;
            out_ground[ground_index * 3 + 2] = zb;
        }
    } else {
        int obstacle_index = atomicAdd(obstacle_count, 1);
        if (obstacle_index < max_points) {
            out_obstacle[obstacle_index * 3 + 0] = xb;
            out_obstacle[obstacle_index * 3 + 1] = yb;
            out_obstacle[obstacle_index * 3 + 2] = zb;
        }
    }
}

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
        max_points,
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
        max_points,
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

namespace {

constexpr unsigned long long kEmptyVoxelKey = 0xFFFFFFFFFFFFFFFFULL;
constexpr int32_t kVoxelCoordBias           = 1 << 20;
constexpr uint64_t kVoxelCoordMask          = 0x1FFFFFULL;  // 21 bits per axis (63 bits total)
constexpr int kMaxVoxelHashProbes           = 1024;

__device__ inline uint32_t hashVoxelKey(uint64_t key, uint32_t mask)
{
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return static_cast<uint32_t>(key) & mask;
}

__global__ void accumulateVoxelCentroidsKernel(
    const float* __restrict__ in_points,
    int num_points,
    float inv_voxel_size,
    unsigned long long* __restrict__ hash_keys,
    float* __restrict__ hash_sums,
    int* __restrict__ hash_counts,
    uint32_t hash_mask,
    int* __restrict__ dropped_count
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) {
        return;
    }

    const float x = in_points[idx * 3 + 0];
    const float y = in_points[idx * 3 + 1];
    const float z = in_points[idx * 3 + 2];
    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
        return;
    }

    int32_t ix = static_cast<int32_t>(floorf(x * inv_voxel_size));
    int32_t iy = static_cast<int32_t>(floorf(y * inv_voxel_size));
    int32_t iz = static_cast<int32_t>(floorf(z * inv_voxel_size));

    ix = max(-kVoxelCoordBias, min(kVoxelCoordBias - 1, ix));
    iy = max(-kVoxelCoordBias, min(kVoxelCoordBias - 1, iy));
    iz = max(-kVoxelCoordBias, min(kVoxelCoordBias - 1, iz));

    const uint64_t key =
        (static_cast<uint64_t>(ix + kVoxelCoordBias) & kVoxelCoordMask) |
        ((static_cast<uint64_t>(iy + kVoxelCoordBias) & kVoxelCoordMask) << 21) |
        ((static_cast<uint64_t>(iz + kVoxelCoordBias) & kVoxelCoordMask) << 42);

    uint32_t slot = hashVoxelKey(key, hash_mask);
    for (int probe = 0; probe < kMaxVoxelHashProbes; ++probe) {
        const unsigned long long prev = atomicCAS(
            &hash_keys[slot], kEmptyVoxelKey, static_cast<unsigned long long>(key)
        );
        if (prev == kEmptyVoxelKey || prev == static_cast<unsigned long long>(key)) {
            atomicAdd(&hash_sums[slot * 3 + 0], x);
            atomicAdd(&hash_sums[slot * 3 + 1], y);
            atomicAdd(&hash_sums[slot * 3 + 2], z);
            atomicAdd(&hash_counts[slot], 1);
            return;
        }
        slot = (slot + 1U) & hash_mask;
    }
    if (dropped_count) {
        atomicAdd(dropped_count, 1);
    }
}

__global__ void compactVoxelCentroidsKernel(
    const float* __restrict__ hash_sums,
    const int* __restrict__ hash_counts,
    int hash_capacity,
    float* __restrict__ out_points,
    int max_points,
    int* __restrict__ out_count
)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= hash_capacity) {
        return;
    }

    const int count = hash_counts[slot];
    if (count > 0) {
        const int out_idx = atomicAdd(out_count, 1);
        if (out_idx < max_points) {
            const float inv_count = 1.0f / static_cast<float>(count);
            out_points[out_idx * 3 + 0] = hash_sums[slot * 3 + 0] * inv_count;
            out_points[out_idx * 3 + 1] = hash_sums[slot * 3 + 1] * inv_count;
            out_points[out_idx * 3 + 2] = hash_sums[slot * 3 + 2] * inv_count;
        }
    }
}

}  // namespace

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
    int* d_dropped_count,
    int* h_dropped_count
)
{
    if (num_points <= 0 || voxel_size <= 0.0f || hash_capacity <= 0 ||
        (hash_capacity & (hash_capacity - 1)) != 0 ||
        !d_points || !d_hash_keys || !d_hash_sums || !d_hash_counts ||
        !d_out_count || !h_out_count) {
        if (h_out_count) {
            *h_out_count = num_points;
        }
        if (h_dropped_count) {
            *h_dropped_count = 0;
        }
        return;
    }

    cudaMemsetAsync(
        d_hash_keys, 0xFF, static_cast<size_t>(hash_capacity) * sizeof(unsigned long long), stream
    );
    cudaMemsetAsync(d_hash_sums, 0, static_cast<size_t>(hash_capacity) * 3 * sizeof(float), stream);
    cudaMemsetAsync(d_hash_counts, 0, static_cast<size_t>(hash_capacity) * sizeof(int), stream);
    cudaMemsetAsync(d_out_count, 0, sizeof(int), stream);
    if (d_dropped_count) {
        cudaMemsetAsync(d_dropped_count, 0, sizeof(int), stream);
    }

    const int threads_per_block = 256;
    const int accum_blocks      = (num_points + threads_per_block - 1) / threads_per_block;
    const uint32_t hash_mask    = static_cast<uint32_t>(hash_capacity - 1);
    const float inv_voxel_size  = 1.0f / voxel_size;

    accumulateVoxelCentroidsKernel<<<accum_blocks, threads_per_block, 0, stream>>>(
        d_points,
        num_points,
        inv_voxel_size,
        d_hash_keys,
        d_hash_sums,
        d_hash_counts,
        hash_mask,
        d_dropped_count
    );

    const int compact_blocks = (hash_capacity + threads_per_block - 1) / threads_per_block;
    compactVoxelCentroidsKernel<<<compact_blocks, threads_per_block, 0, stream>>>(
        d_hash_sums,
        d_hash_counts,
        hash_capacity,
        d_points,
        max_points,
        d_out_count
    );

    cudaMemcpyAsync(h_out_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (d_dropped_count && h_dropped_count) {
        cudaMemcpyAsync(h_dropped_count, d_dropped_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    } else if (h_dropped_count) {
        *h_dropped_count = 0;
    }
}

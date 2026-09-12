#include <cuda_runtime.h>

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // センサ（livox_frame）座標系の点
    float xs = in_cloud[idx * 3 + 0];
    float ys = in_cloud[idx * 3 + 1];
    float zs = in_cloud[idx * 3 + 2];

    // base_link 座標系へのアフィン変換
    float xb = transform[0] * xs + transform[1] * ys + transform[2] * zs + transform[3];
    float yb = transform[4] * xs + transform[5] * ys + transform[6] * zs + transform[7];
    float zb = transform[8] * xs + transform[9] * ys + transform[10] * zs + transform[11];

    // 1. 最大検出範囲外（base_link 原点からの水平距離）を無視
    float dist_sq = xb * xb + yb * yb;
    if (dist_sq > range_max * range_max) return;

    // 2. ロボット本体の除去 (半径 0.6m かつ 高さ 0.0m ~ 1.2m 領域)
    if (dist_sq <= robot_radius * robot_radius && zb >= robot_height_min &&
        zb <= robot_height_max) {
        return;
    }

    // 3. 地面 / 障害物の分離 (base_link 基準の Z 閾値)
    if (zb <= ground_z_thresh) {
        int g_idx                 = atomicAdd(ground_count, 1);
        out_ground[g_idx * 3 + 0] = xb;
        out_ground[g_idx * 3 + 1] = yb;
        out_ground[g_idx * 3 + 2] = zb;
    } else {
        int o_idx                   = atomicAdd(obstacle_count, 1);
        out_obstacle[o_idx * 3 + 0] = xb;
        out_obstacle[o_idx * 3 + 1] = yb;
        out_obstacle[o_idx * 3 + 2] = zb;
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

    int threadsPerBlock = 256;
    int blocksPerGrid   = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    filterGroundKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
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
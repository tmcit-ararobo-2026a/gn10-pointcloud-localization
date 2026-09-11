#include <cuda_runtime.h>

__global__ void filterGroundKernel(
    const float* in_cloud,
    float* out_ground,
    float* out_obstacle,
    int num_points,
    float range_max,
    float ground_z_thresh,
    int* ground_count,
    int* obstacle_count
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float x = in_cloud[idx * 3 + 0];
    float y = in_cloud[idx * 3 + 1];
    float z = in_cloud[idx * 3 + 2];

    // 12m 範囲外（xy平面）を無視
    if (x * x + y * y > range_max * range_max) return;

    if (z <= ground_z_thresh) {
        int g_idx                 = atomicAdd(ground_count, 1);
        out_ground[g_idx * 3 + 0] = x;
        out_ground[g_idx * 3 + 1] = y;
        out_ground[g_idx * 3 + 2] = z;
    } else {
        int o_idx                   = atomicAdd(obstacle_count, 1);
        out_obstacle[o_idx * 3 + 0] = x;
        out_obstacle[o_idx * 3 + 1] = y;
        out_obstacle[o_idx * 3 + 2] = z;
    }
}

// C++ 呼び出し用ラッパー
void launchGroundFilter(
    const float* d_in,
    float* d_ground,
    float* d_obstacle,
    int num_points,
    float range_max,
    float ground_z_thresh,
    int* d_ground_count,
    int* d_obstacle_count,
    int* h_ground_count,
    int* h_obstacle_count
)
{
    cudaMemset(d_ground_count, 0, sizeof(int));
    cudaMemset(d_obstacle_count, 0, sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid   = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    filterGroundKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_in,
        d_ground,
        d_obstacle,
        num_points,
        range_max,
        ground_z_thresh,
        d_ground_count,
        d_obstacle_count
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_ground_count, d_ground_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_obstacle_count, d_obstacle_count, sizeof(int), cudaMemcpyDeviceToHost);
}
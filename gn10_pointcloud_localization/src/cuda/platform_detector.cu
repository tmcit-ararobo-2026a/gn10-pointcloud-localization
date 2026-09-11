#include <curand_kernel.h>

#include <cmath>
#include <vector>

#include "gn10_pointcloud_localization/cuda/platform_detector.cuh"

__global__ void extractEdgePointsKernel(
    const float* cloud,
    int num_points,
    float* edge_pts_xy,
    int* edge_count,
    float z_min,
    float z_max
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float x = cloud[idx * 3 + 0];
    float y = cloud[idx * 3 + 1];
    float z = cloud[idx * 3 + 2];

    if (z >= z_min && z <= z_max) {
        int e_idx                  = atomicAdd(edge_count, 1);
        edge_pts_xy[e_idx * 2 + 0] = x;
        edge_pts_xy[e_idx * 2 + 1] = y;
    }
}

__global__ void ransacLineKernel(
    const float* edge_pts,
    int num_edges,
    float threshold,
    unsigned int seed,
    float* block_lines,
    int* block_inliers
)
{
    int block_id = blockIdx.x;
    int tid      = threadIdx.x;

    __shared__ float s_line[3];
    __shared__ int s_inliers;

    if (tid == 0) {
        s_inliers = 0;
        curandState state;
        curand_init(seed, block_id, 0, &state);

        int idx1 = curand(&state) % num_edges;
        int idx2 = curand(&state) % num_edges;

        // 同一点が選ばれた場合の回避
        if (idx1 == idx2) {
            idx2 = (idx1 + 1) % num_edges;
        }

        float x1 = edge_pts[idx1 * 2 + 0], y1 = edge_pts[idx1 * 2 + 1];
        float x2 = edge_pts[idx2 * 2 + 0], y2 = edge_pts[idx2 * 2 + 1];

        float dx = x2 - x1, dy = y2 - y1;
        float norm = sqrtf(dx * dx + dy * dy);

        if (norm > 1e-4f) {
            s_line[0] = -dy / norm;
            s_line[1] = dx / norm;
            s_line[2] = -(s_line[0] * x1 + s_line[1] * y1);
        } else {
            s_line[0] = 1.0f;
            s_line[1] = 0.0f;
            s_line[2] = 0.0f;
        }
    }
    // s_inliers = 0 の書き込みを全スレッドで同期
    __syncthreads();

    int local_inliers = 0;
    float a = s_line[0], b = s_line[1], c = s_line[2];

    for (int i = tid; i < num_edges; i += blockDim.x) {
        float x = edge_pts[i * 2 + 0];
        float y = edge_pts[i * 2 + 1];
        if (fabsf(a * x + b * y + c) <= threshold) {
            local_inliers++;
        }
    }

    atomicAdd(&s_inliers, local_inliers);
    __syncthreads();

    if (tid == 0) {
        block_lines[block_id * 3 + 0] = a;
        block_lines[block_id * 3 + 1] = b;
        block_lines[block_id * 3 + 2] = c;
        block_inliers[block_id]       = s_inliers;
    }
}

void launchPlatformDetector(
    const float* d_obstacle_cloud,
    int num_obstacle_points,
    float z_min,
    float z_max,
    int max_ransac_iter,
    float ransac_thresh,
    PlatformConstraint& out_constraint
)
{
    out_constraint.valid = false;
    if (num_obstacle_points < 30) return;

    float* d_edge_pts_xy;
    int* d_edge_count;
    cudaMalloc(&d_edge_pts_xy, num_obstacle_points * 2 * sizeof(float));
    cudaMalloc(&d_edge_count, sizeof(int));
    cudaMemset(d_edge_count, 0, sizeof(int));

    int threads = 256;
    int blocks  = (num_obstacle_points + threads - 1) / threads;
    extractEdgePointsKernel<<<blocks, threads>>>(
        d_obstacle_cloud, num_obstacle_points, d_edge_pts_xy, d_edge_count, z_min, z_max
    );

    int h_edge_count = 0;
    cudaMemcpy(&h_edge_count, d_edge_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_edge_count >= 30) {
        float* d_block_lines;
        int* d_block_inliers;
        cudaMalloc(&d_block_lines, max_ransac_iter * 3 * sizeof(float));
        cudaMalloc(&d_block_inliers, max_ransac_iter * sizeof(int));

        ransacLineKernel<<<max_ransac_iter, threads>>>(
            d_edge_pts_xy, h_edge_count, ransac_thresh, rand(), d_block_lines, d_block_inliers
        );

        // 動的配列を使用してメモリ破壊を防止
        std::vector<float> h_lines(max_ransac_iter * 3);
        std::vector<int> h_inliers(max_ransac_iter);

        cudaMemcpy(
            h_lines.data(),
            d_block_lines,
            max_ransac_iter * 3 * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        cudaMemcpy(
            h_inliers.data(), d_block_inliers, max_ransac_iter * sizeof(int), cudaMemcpyDeviceToHost
        );

        int best_idx = 0, max_inliers = -1;
        for (int i = 0; i < max_ransac_iter; ++i) {
            if (h_inliers[i] > max_inliers) {
                max_inliers = h_inliers[i];
                best_idx    = i;
            }
        }

        // platform_detector.cu 内の出力判定部分
        if ((float)max_inliers / h_edge_count > 0.20f && max_inliers >= 15) {
            float a = h_lines[best_idx * 3 + 0];
            float b = h_lines[best_idx * 3 + 1];
            float c = h_lines[best_idx * 3 + 2];

            // 原点から直線へ下ろした垂線の足 (x0, y0)
            float norm_sq             = a * a + b * b;
            out_constraint.center_x   = -a * c / norm_sq;
            out_constraint.center_y   = -b * c / norm_sq;
            out_constraint.distance_m = std::sqrt(
                out_constraint.center_x * out_constraint.center_x +
                out_constraint.center_y * out_constraint.center_y
            );

            // (a,b) は法線ベクトルなので、直線の接線方向（教壇の壁方向）の Yaw 角を計算
            // 法線 (a, b) に対して 90度回転させた方向 (-b, a)
            out_constraint.yaw_error_rad = std::atan2(a, -b);
            out_constraint.valid         = true;
        }

        cudaFree(d_block_lines);
        cudaFree(d_block_inliers);
    }

    cudaFree(d_edge_pts_xy);
    cudaFree(d_edge_count);
}
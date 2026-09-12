#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

#include "gn10_pointcloud_localization/cuda/field_objects.cuh"

// GPU 定数メモリの定義
__constant__ FieldObject c_map_objects[128];
static int g_num_map_objects = 0;

// GPU グローバルバッファ（ファイルスコープで定義）
static PoseCandidate* d_candidates = nullptr;
static float* d_costs              = nullptr;
static int g_max_candidates        = 0;

// ★ 動的判定用バッファの定義を追加
static uint8_t* d_is_dynamic = nullptr;
static int g_max_dynamic_pts = 0;

// 点 (px, py) と 軸平行ボックスの2D最短距離
__device__ float distToBox2D(float px, float py, float cx, float cy, float half_w, float half_d)
{
    float dx = fabsf(px - cx) - half_w;
    float dy = fabsf(py - cy) - half_d;
    float ax = fmaxf(dx, 0.0f);
    float ay = fmaxf(dy, 0.0f);
    return sqrtf(ax * ax + ay * ay) + fminf(fmaxf(dx, dy), 0.0f);
}

// 点 (px, py) と 円筒の2D最短距離
__device__ float distToCylinder2D(float px, float py, float cx, float cy, float radius)
{
    float dx          = px - cx;
    float dy          = py - cy;
    float dist_center = sqrtf(dx * dx + dy * dy);
    return fabsf(dist_center - radius);
}

// 全姿勢候補の残差計算カーネル
__global__ void evaluateFieldSDFKernel(
    const float* __restrict__ cloud,
    int num_points,
    int num_objects,
    const PoseCandidate* __restrict__ candidates,
    float* __restrict__ out_costs,
    float max_dist_thresh
)
{
    int pose_idx = blockIdx.x;
    int tid      = threadIdx.x;

    float rx   = candidates[pose_idx].x;
    float ry   = candidates[pose_idx].y;
    float ryaw = candidates[pose_idx].yaw;

    float cos_y = cosf(ryaw);
    float sin_y = sinf(ryaw);

    __shared__ float s_cost[256];
    s_cost[tid] = 0.0f;

    for (int i = tid; i < num_points; i += blockDim.x) {
        float lx = cloud[i * 3 + 0];
        float ly = cloud[i * 3 + 1];
        float lz = cloud[i * 3 + 2];

        float wx = cos_y * lx - sin_y * ly + rx;
        float wy = sin_y * lx + cos_y * ly + ry;
        float wz = lz;

        float min_d = max_dist_thresh;

        for (int o = 0; o < num_objects; ++o) {
            const FieldObject obj = c_map_objects[o];
            if (wz >= (obj.z_min - 0.1f) && wz <= (obj.z_max + 0.1f)) {
                float d = max_dist_thresh;
                if (obj.type == CYLINDER) {
                    d = distToCylinder2D(wx, wy, obj.center_x, obj.center_y, obj.param1);
                } else if (obj.type == BOX) {
                    d = distToBox2D(wx, wy, obj.center_x, obj.center_y, obj.param1, obj.param2);
                }
                if (d < min_d) min_d = d;
            }
        }
        s_cost[tid] += (min_d < max_dist_thresh) ? min_d : max_dist_thresh;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_cost[tid] += s_cost[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_costs[pose_idx] = s_cost[0] / static_cast<float>(num_points);
    }
}

// 確定姿勢に対する動的障害物判定カーネル
__global__ void filterDynamicPointsKernel(
    const float* __restrict__ cloud,
    int num_points,
    int num_objects,
    PoseCandidate best_pose,
    float dynamic_dist_thresh,
    uint8_t* __restrict__ out_is_dynamic
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    float rx   = best_pose.x;
    float ry   = best_pose.y;
    float ryaw = best_pose.yaw;

    float cos_y = cosf(ryaw);
    float sin_y = sinf(ryaw);

    float lx = cloud[i * 3 + 0];
    float ly = cloud[i * 3 + 1];
    float lz = cloud[i * 3 + 2];

    float wx = cos_y * lx - sin_y * ly + rx;
    float wy = sin_y * lx + cos_y * ly + ry;
    float wz = lz;

    float min_d = dynamic_dist_thresh;

    for (int o = 0; o < num_objects; ++o) {
        const FieldObject obj = c_map_objects[o];
        if (wz >= (obj.z_min - 0.1f) && wz <= (obj.z_max + 0.1f)) {
            float d = dynamic_dist_thresh;
            if (obj.type == CYLINDER) {
                d = distToCylinder2D(wx, wy, obj.center_x, obj.center_y, obj.param1);
            } else if (obj.type == BOX) {
                d = distToBox2D(wx, wy, obj.center_x, obj.center_y, obj.param1, obj.param2);
            }
            if (d < min_d) min_d = d;
        }
    }

    out_is_dynamic[i] = (min_d >= dynamic_dist_thresh) ? 1 : 0;
}

extern "C" {

void uploadFieldMapToGPU(const std::vector<FieldObject>& host_map)
{
    g_num_map_objects = static_cast<int>(host_map.size());
    if (g_num_map_objects > 128) g_num_map_objects = 128;

    size_t bytes = g_num_map_objects * sizeof(FieldObject);
    cudaMemcpyToSymbol(c_map_objects, host_map.data(), bytes);
}

bool launchFieldSDFMatcher(
    cudaStream_t stream,
    const float* d_obstacle_cloud,
    int num_points,
    const PoseCandidate& base_pose,
    float range_xy,
    float step_xy,
    float range_yaw,
    float step_yaw,
    float max_dist_thresh,
    float dynamic_dist_thresh,
    PoseCandidate& out_best_pose,
    float& out_best_cost,
    std::vector<float>& out_dynamic_pts
)
{
    if (num_points <= 0 || g_num_map_objects <= 0) return false;

    // グリッド候補姿勢の生成
    std::vector<PoseCandidate> h_candidates;
    for (float dx = -range_xy; dx <= range_xy + 1e-5f; dx += step_xy) {
        for (float dy = -range_xy; dy <= range_xy + 1e-5f; dy += step_xy) {
            for (float dyaw = -range_yaw; dyaw <= range_yaw + 1e-5f; dyaw += step_yaw) {
                h_candidates.push_back({base_pose.x + dx, base_pose.y + dy, base_pose.yaw + dyaw});
            }
        }
    }

    int num_candidates = static_cast<int>(h_candidates.size());
    if (num_candidates == 0) return false;

    // GPU メモリバッファの再確保判定
    if (num_candidates > g_max_candidates) {
        if (d_candidates) cudaFree(d_candidates);
        if (d_costs) cudaFree(d_costs);
        g_max_candidates = num_candidates * 2;
        cudaMalloc(&d_candidates, g_max_candidates * sizeof(PoseCandidate));
        cudaMalloc(&d_costs, g_max_candidates * sizeof(float));
    }

    if (num_points > g_max_dynamic_pts) {
        if (d_is_dynamic) cudaFree(d_is_dynamic);
        g_max_dynamic_pts = num_points * 2;
        cudaMalloc(&d_is_dynamic, g_max_dynamic_pts * sizeof(uint8_t));
    }

    cudaMemcpyAsync(
        d_candidates,
        h_candidates.data(),
        num_candidates * sizeof(PoseCandidate),
        cudaMemcpyHostToDevice,
        stream
    );

    // SDF Matching カーネル実行
    int sdf_threads = 256;
    int sdf_blocks  = num_candidates;

    evaluateFieldSDFKernel<<<sdf_blocks, sdf_threads, 0, stream>>>(
        d_obstacle_cloud, num_points, g_num_map_objects, d_candidates, d_costs, max_dist_thresh
    );

    std::vector<float> h_costs(num_candidates);
    cudaMemcpyAsync(
        h_costs.data(), d_costs, num_candidates * sizeof(float), cudaMemcpyDeviceToHost, stream
    );

    // コスト比較のために同期（必要最小限の同期）
    cudaStreamSynchronize(stream);

    int best_idx   = 0;
    float min_cost = FLT_MAX;
    for (int i = 0; i < num_candidates; ++i) {
        if (h_costs[i] < min_cost) {
            min_cost = h_costs[i];
            best_idx = i;
        }
    }

    out_best_pose = h_candidates[best_idx];
    out_best_cost = min_cost;

    // 最良姿勢をもとに動的点群をフィルタリング
    int dyn_threads = 256;
    int dyn_blocks  = (num_points + dyn_threads - 1) / dyn_threads;

    filterDynamicPointsKernel<<<dyn_blocks, dyn_threads, 0, stream>>>(
        d_obstacle_cloud,
        num_points,
        g_num_map_objects,
        out_best_pose,
        dynamic_dist_thresh,
        d_is_dynamic
    );

    std::vector<uint8_t> h_is_dynamic(num_points);
    std::vector<float> h_raw_cloud(num_points * 3);

    cudaMemcpyAsync(
        h_is_dynamic.data(),
        d_is_dynamic,
        num_points * sizeof(uint8_t),
        cudaMemcpyDeviceToHost,
        stream
    );
    cudaMemcpyAsync(
        h_raw_cloud.data(),
        d_obstacle_cloud,
        num_points * 3 * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream
    );

    cudaStreamSynchronize(stream);

    out_dynamic_pts.clear();
    out_dynamic_pts.reserve(num_points * 3);

    for (int i = 0; i < num_points; ++i) {
        if (h_is_dynamic[i]) {
            out_dynamic_pts.push_back(h_raw_cloud[i * 3 + 0]);
            out_dynamic_pts.push_back(h_raw_cloud[i * 3 + 1]);
            out_dynamic_pts.push_back(h_raw_cloud[i * 3 + 2]);
        }
    }

    return true;
}

}  // extern "C"
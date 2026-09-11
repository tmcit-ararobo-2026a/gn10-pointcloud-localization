#include <cfloat>
#include <cmath>
#include <vector>

#include "gn10_pointcloud_localization/cuda/field_objects.cuh"

// Constant memory または グローバルメモリの定義
static FieldObject* d_map_objects = nullptr;
static int g_num_map_objects      = 0;

static PoseCandidate* d_candidates = nullptr;
static float* d_costs              = nullptr;
static int g_max_candidates        = 0;

// 点 (px, py) と 軸平行ボックス (half_w, half_d) の2D最短距離
__device__ float distToBox2D(float px, float py, float cx, float cy, float half_w, float half_d)
{
    float dx = fabsf(px - cx) - half_w;
    float dy = fabsf(py - cy) - half_d;
    float ax = fmaxf(dx, 0.0f);
    float ay = fmaxf(dy, 0.0f);
    return sqrtf(ax * ax + ay * ay) + fminf(fmaxf(dx, dy), 0.0f);
}

// 点 (px, py) と 円筒 (radius) の2D最短距離
__device__ float distToCylinder2D(float px, float py, float cx, float cy, float radius)
{
    float dx          = px - cx;
    float dy          = py - cy;
    float dist_center = sqrtf(dx * dx + dy * dy);
    return fabsf(dist_center - radius);
}

// LiDAR点群と全マップオブジェクトとの最小残差を計算する CUDA カーネル
__global__ void evaluateFieldSDFKernel(
    const float* __restrict__ cloud,
    int num_points,
    const FieldObject* __restrict__ objects,
    int num_objects,
    const PoseCandidate* __restrict__ candidates,
    float* __restrict__ out_costs,
    float max_dist_thresh
)
{
    int pose_idx = blockIdx.x;  // 1ブロック = 1つのロボット仮定姿勢
    int tid      = threadIdx.x;

    float rx   = candidates[pose_idx].x;
    float ry   = candidates[pose_idx].y;
    float ryaw = candidates[pose_idx].yaw;

    float cos_y = cosf(ryaw);
    float sin_y = sinf(ryaw);

    __shared__ float s_cost[256];
    s_cost[tid] = 0.0f;

    // 各スレッドが割当られた点群の残差を計算
    for (int i = tid; i < num_points; i += blockDim.x) {
        float lx = cloud[i * 3 + 0];
        float ly = cloud[i * 3 + 1];
        float lz = cloud[i * 3 + 2];

        // ロボットローカル座標 (base_link) -> ワールド仮定座標への変換
        float wx = cos_y * lx - sin_y * ly + rx;
        float wy = sin_y * lx + cos_y * ly + ry;
        float wz = lz;  // 高さZ

        float min_d = max_dist_thresh;

        // 全フィールドオブジェクトに対する距離評価
        for (int o = 0; o < num_objects; ++o) {
            FieldObject obj = objects[o];

            // 高さ (Z) の合致判定
            if (wz >= obj.z_min && wz <= obj.z_max) {
                float d = max_dist_thresh;
                if (obj.type == CYLINDER) {
                    d = distToCylinder2D(wx, wy, obj.center_x, obj.center_y, obj.param1);
                } else if (obj.type == BOX) {
                    d = distToBox2D(wx, wy, obj.center_x, obj.center_y, obj.param1, obj.param2);
                }
                if (d < min_d) min_d = d;
            }
        }

        // コストの加算（Huber損失的なクリッピング）
        s_cost[tid] += (min_d < max_dist_thresh) ? min_d : max_dist_thresh;
    }
    __syncthreads();

    // ブロック内並列リダクション
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

extern "C" {

void uploadFieldMapToGPU(const std::vector<FieldObject>& host_map)
{
    if (d_map_objects) cudaFree(d_map_objects);
    g_num_map_objects = host_map.size();
    size_t bytes      = g_num_map_objects * sizeof(FieldObject);
    cudaMalloc(&d_map_objects, bytes);
    cudaMemcpy(d_map_objects, host_map.data(), bytes, cudaMemcpyHostToDevice);
}

bool launchFieldSDFMatcher(
    const float* d_obstacle_cloud,
    int num_points,
    const PoseCandidate& base_pose,
    float range_xy,
    float step_xy,
    float range_yaw,
    float step_yaw,
    float max_dist_thresh,
    PoseCandidate& out_best_pose,
    float& out_best_cost
)
{
    if (num_points <= 0 || g_num_map_objects <= 0) return false;

    // グリッド候補姿勢の生成
    std::vector<PoseCandidate> h_candidates;
    for (float dx = -range_xy; dx <= range_xy; dx += step_xy) {
        for (float dy = -range_xy; dy <= range_xy; dy += step_xy) {
            for (float dyaw = -range_yaw; dyaw <= range_yaw; dyaw += step_yaw) {
                h_candidates.push_back({base_pose.x + dx, base_pose.y + dy, base_pose.yaw + dyaw});
            }
        }
    }

    int num_candidates = h_candidates.size();
    if (num_candidates == 0) return false;

    // バッファ確保
    if (num_candidates > g_max_candidates) {
        if (d_candidates) cudaFree(d_candidates);
        if (d_costs) cudaFree(d_costs);
        g_max_candidates = num_candidates;
        cudaMalloc(&d_candidates, g_max_candidates * sizeof(PoseCandidate));
        cudaMalloc(&d_costs, g_max_candidates * sizeof(float));
    }

    cudaMemcpy(
        d_candidates,
        h_candidates.data(),
        num_candidates * sizeof(PoseCandidate),
        cudaMemcpyHostToDevice
    );

    // カーネル起動 (1ブロック = 1姿勢候補)
    int threads_per_block = 256;
    int blocks_per_grid   = num_candidates;

    evaluateFieldSDFKernel<<<blocks_per_grid, threads_per_block>>>(
        d_obstacle_cloud,
        num_points,
        d_map_objects,
        g_num_map_objects,
        d_candidates,
        d_costs,
        max_dist_thresh
    );

    std::vector<float> h_costs(num_candidates);
    cudaMemcpy(h_costs.data(), d_costs, num_candidates * sizeof(float), cudaMemcpyDeviceToHost);

    // 最小コストの探索
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
    return true;
}

}  // extern "C"
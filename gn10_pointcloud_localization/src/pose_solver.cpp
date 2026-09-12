#include "gn10_pointcloud_localization/pose_solver.hpp"

#include <cmath>
#include <cstring>
#include <limits>

#include "gn10_pointcloud_localization/cuda/ground_filter.cuh"

PoseSolver::PoseSolver(int max_points) : max_points_(max_points)
{
    cudaStreamCreate(&stream_);

    cudaMalloc(&d_in_, max_points_ * 3 * sizeof(float));
    cudaMalloc(&d_ground_, max_points_ * 3 * sizeof(float));
    cudaMalloc(&d_obstacle_, max_points_ * 3 * sizeof(float));
    cudaMalloc(&d_ground_count_, sizeof(int));
    cudaMalloc(&d_obstacle_count_, sizeof(int));
    cudaMalloc(&d_transform_, 12 * sizeof(float));

    cudaMallocHost(&h_in_, max_points_ * 3 * sizeof(float));
    cudaMallocHost(&h_out_ground_, max_points_ * 3 * sizeof(float));
    cudaMallocHost(&h_out_obstacle_, max_points_ * 3 * sizeof(float));
}

PoseSolver::~PoseSolver()
{
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }

    cudaFree(d_in_);
    cudaFree(d_ground_);
    cudaFree(d_obstacle_);
    cudaFree(d_ground_count_);
    cudaFree(d_obstacle_count_);
    cudaFree(d_transform_);

    cudaFreeHost(h_in_);
    cudaFreeHost(h_out_ground_);
    cudaFreeHost(h_out_obstacle_);
}

void PoseSolver::setMap(const std::vector<FieldObject>& host_map)
{
    uploadFieldMapToGPU(host_map);
}

bool PoseSolver::processPointCloud(
    const std::vector<float>& h_raw_cloud,
    const float h_transform[12],
    const GroundFilterParams& filter_params,
    const MatchingParams& match_params,
    const PoseCandidate& search_base_pose,
    std::vector<float>& out_ground_pts,
    std::vector<float>& out_obstacle_pts,
    std::vector<float>& out_dynamic_pts,
    PoseCandidate& out_best_pose,
    float& out_best_cost
)
{
    int num_points = static_cast<int>(h_raw_cloud.size() / 3);
    if (num_points == 0 || num_points > max_points_) {
        out_best_cost = std::numeric_limits<float>::max();
        return false;
    }

    std::memcpy(h_in_, h_raw_cloud.data(), num_points * 3 * sizeof(float));

    cudaMemcpyAsync(d_in_, h_in_, num_points * 3 * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_transform_, h_transform, 12 * sizeof(float), cudaMemcpyHostToDevice, stream_);

    int h_ground_count   = 0;
    int h_obstacle_count = 0;

    launchGroundFilter(
        stream_,
        d_in_,
        d_ground_,
        d_obstacle_,
        d_transform_,
        num_points,
        filter_params.range_max,
        filter_params.robot_radius,
        filter_params.robot_height_min,
        filter_params.robot_height_max,
        filter_params.ground_z_thresh,
        d_ground_count_,
        d_obstacle_count_,
        &h_ground_count,
        &h_obstacle_count
    );

    cudaStreamSynchronize(stream_);

    bool pose_matched = false;
    if (h_obstacle_count > 50) {
        pose_matched = launchFieldSDFMatcher(
            stream_,
            d_obstacle_,
            h_obstacle_count,
            search_base_pose,
            match_params.range_xy,
            match_params.step_xy,
            match_params.range_yaw,
            match_params.step_yaw,
            match_params.max_dist_thresh,
            match_params.dynamic_dist_thresh,
            out_best_pose,
            out_best_cost,
            out_dynamic_pts
        );
    } else {
        // 点群数が不十分な場合は明確に最大コストをセット
        out_best_cost = std::numeric_limits<float>::max();
        out_dynamic_pts.clear();
    }

    // デバッグ出力用の参照が渡されている場合のみ Host へコピー
    cudaMemcpyAsync(
        h_out_ground_,
        d_ground_,
        h_ground_count * 3 * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream_
    );
    cudaMemcpyAsync(
        h_out_obstacle_,
        d_obstacle_,
        h_obstacle_count * 3 * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream_
    );
    cudaStreamSynchronize(stream_);

    out_ground_pts.assign(h_out_ground_, h_out_ground_ + h_ground_count * 3);
    out_obstacle_pts.assign(h_out_obstacle_, h_out_obstacle_ + h_obstacle_count * 3);

    return pose_matched;
}
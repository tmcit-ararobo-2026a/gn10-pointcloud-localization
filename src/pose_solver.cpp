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
}

void PoseSolver::setMap(const std::vector<FieldObject>& host_map)
{
    std::vector<FieldObject> matching_map;
    matching_map.reserve(host_map.size());
    for (const auto& object : host_map) {
        if (object.type != VISUAL_BOX) matching_map.push_back(object);
    }
    uploadFieldMapToGPU(matching_map);
}

bool PoseSolver::processPointCloud(
    const std::vector<float>& h_raw_cloud,
    const float h_transform[12],
    const GroundFilterParams& filter_params,
    const MatchingParams& match_params,
    const PoseCandidate& search_base_pose,
    std::vector<float>& out_dynamic_pts,
    PoseCandidate& out_best_pose,
    float& out_best_cost
)
{
    int num_points = static_cast<int>(h_raw_cloud.size() / 3);
    ground_count_ = 0;
    obstacle_count_ = 0;
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
    ground_count_ = h_ground_count;
    obstacle_count_ = h_obstacle_count;

    bool pose_matched = false;
    if (h_obstacle_count > 50) {
        PoseCandidate coarse_pose{};
        float coarse_cost = std::numeric_limits<float>::max();
        std::vector<float> ignored_dynamic;
        pose_matched = launchFieldSDFMatcher(
            stream_,
            d_obstacle_,
            h_obstacle_count,
            search_base_pose,
            match_params.range_xy,
            match_params.range_xy,
            match_params.step_xy,
            match_params.range_yaw,
            match_params.step_yaw,
            match_params.max_dist_thresh,
            match_params.dynamic_dist_thresh,
            match_params.field_min_x,
            match_params.field_max_x,
            match_params.field_min_y,
            match_params.field_max_y,
            coarse_pose,
            coarse_cost,
            match_params.fine_refine ? ignored_dynamic : out_dynamic_pts,
            !match_params.fine_refine
        );
        out_best_pose = coarse_pose;
        out_best_cost = coarse_cost;
        if (pose_matched && match_params.fine_refine) {
            // Five samples per axis around the coarse winner. This adds 125
            // candidates without expanding the full-field search.
            const float fine_xy = match_params.step_xy / 5.0f;
            const float fine_yaw = match_params.step_yaw / 5.0f;
            pose_matched = launchFieldSDFMatcher(
                stream_, d_obstacle_, h_obstacle_count, coarse_pose,
                2.0f * fine_xy, 2.0f * fine_xy, fine_xy,
                2.0f * fine_yaw, fine_yaw,
                match_params.max_dist_thresh, match_params.dynamic_dist_thresh,
                match_params.field_min_x, match_params.field_max_x,
                match_params.field_min_y, match_params.field_max_y,
                out_best_pose, out_best_cost, out_dynamic_pts, true
            );
        }
    } else {
        out_best_cost = std::numeric_limits<float>::max();
        out_dynamic_pts.clear();
    }

    return pose_matched;
}

int PoseSolver::prepareObstacleCloud(
    const std::vector<float>& h_raw_cloud,
    const float h_transform[12],
    const GroundFilterParams& filter_params
)
{
    int num_points = static_cast<int>(h_raw_cloud.size() / 3);
    ground_count_ = 0;
    obstacle_count_ = 0;
    if (num_points == 0 || num_points > max_points_) {
        return 0;
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
    ground_count_ = h_ground_count;
    obstacle_count_ = h_obstacle_count;
    return h_obstacle_count;
}

void PoseSolver::copyFilteredClouds(
    std::vector<float>* ground, std::vector<float>* obstacle
)
{
    if (ground) {
        ground->resize(static_cast<size_t>(ground_count_) * 3);
        if (ground_count_ > 0) {
            cudaMemcpy(
                ground->data(), d_ground_, ground->size() * sizeof(float),
                cudaMemcpyDeviceToHost
            );
        }
    }
    if (obstacle) {
        obstacle->resize(static_cast<size_t>(obstacle_count_) * 3);
        if (obstacle_count_ > 0) {
            cudaMemcpy(
                obstacle->data(), d_obstacle_, obstacle->size() * sizeof(float),
                cudaMemcpyDeviceToHost
            );
        }
    }
}

bool PoseSolver::evaluateGlobalSDF(
    int obstacle_count,
    const PoseCandidate& base_pose,
    float range_x,
    float range_y,
    float step_xy,
    float range_yaw,
    float step_yaw,
    const MatchingParams& match_params,
    PoseCandidate& out_best_pose,
    float& out_best_cost
)
{
    std::vector<float> dummy_dynamic;
    return launchFieldSDFMatcher(
        stream_,
        d_obstacle_,
        obstacle_count,
        base_pose,
        range_x,
        range_y,
        step_xy,
        range_yaw,
        step_yaw,
        match_params.max_dist_thresh,
        match_params.dynamic_dist_thresh,
        match_params.field_min_x,
        match_params.field_max_x,
        match_params.field_min_y,
        match_params.field_max_y,
        out_best_pose,
        out_best_cost,
        dummy_dynamic,
        false
    );
}

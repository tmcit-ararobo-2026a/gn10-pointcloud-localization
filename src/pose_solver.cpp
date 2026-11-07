// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include "gn10_pointcloud_localization/pose_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "gn10_pointcloud_localization/cuda/ground_filter.cuh"

PoseSolver::PoseSolver(int max_points) : max_points_(max_points)
{
    cudaStreamCreate(&stream_);

    cudaMalloc(&d_in_, max_points_ * 3 * sizeof(float));
    cudaMalloc(&d_timed_in_, max_points_ * sizeof(TimedScanPoint));
    cudaMalloc(&d_imu_samples_, kMaxImuSamples * sizeof(IMUSample));
    cudaMalloc(&d_ground_, max_points_ * 3 * sizeof(float));
    cudaMalloc(&d_obstacle_, max_points_ * 3 * sizeof(float));
    cudaMalloc(&d_ground_count_, sizeof(int));
    cudaMalloc(&d_obstacle_count_, sizeof(int));
    cudaMalloc(&d_transform_, 12 * sizeof(float));

    cudaMallocHost(&h_in_, max_points_ * 3 * sizeof(float));
    cudaMallocHost(&h_timed_in_, max_points_ * sizeof(TimedScanPoint));
}

PoseSolver::~PoseSolver()
{
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }

    cudaFree(d_in_);
    cudaFree(d_timed_in_);
    cudaFree(d_imu_samples_);
    cudaFree(d_ground_);
    cudaFree(d_obstacle_);
    cudaFree(d_ground_count_);
    cudaFree(d_obstacle_count_);
    cudaFree(d_transform_);

    cudaFreeHost(h_in_);
    cudaFreeHost(h_timed_in_);
}

void PoseSolver::setMap(const std::vector<FieldObject>& host_map)
{
    std::vector<FieldObject> matching_map;
    matching_map.reserve(host_map.size());
    for (const auto& object : host_map) {
        if (object.type != VISUAL_BOX) matching_map.push_back(object);
    }
    uploadFieldMapToGPU(matching_map);
    use_esdf_ = false;
}

void PoseSolver::setESDFMap(const ESDFMap& esdf_map)
{
    if (esdf_map.empty()) return;
    uploadESDFMapToGPU(esdf_map.header(), esdf_map.data().data());
    use_esdf_ = true;
}

bool PoseSolver::processPointCloud(
    const std::vector<float>& h_raw_cloud,
    const float h_transform[12],
    const GroundFilterParams& filter_params,
    const MatchingParams& match_params,
    const PoseCandidate& search_base_pose,
    std::vector<float>& out_dynamic_pts,
    PoseCandidate& out_best_pose,
    float& out_best_cost,
    int* out_inlier_count,
    float* out_inlier_cost,
    const std::vector<float>* extra_base_link_points,
    bool extract_dynamic
)
{
    std::lock_guard<std::recursive_mutex> lock(solver_mutex_);
    const int h_obstacle_count =
        prepareObstacleCloud(h_raw_cloud, h_transform, filter_params, extra_base_link_points);
    if (h_obstacle_count == 0) {
        out_best_cost = std::numeric_limits<float>::max();
        if (out_inlier_count != nullptr) {
            *out_inlier_count = 0;
        }
        if (out_inlier_cost != nullptr) {
            *out_inlier_cost = 1.0f;
        }
        out_dynamic_pts.clear();
        return false;
    }

    return matchObstacleCloud(
        h_obstacle_count,
        match_params,
        search_base_pose,
        out_dynamic_pts,
        out_best_pose,
        out_best_cost,
        out_inlier_count,
        out_inlier_cost,
        extract_dynamic
    );
}

bool PoseSolver::processPointCloud(
    const std::vector<TimedScanPoint>& timed_points,
    const float h_transform[12],
    const DeskewParams& deskew_params,
    const std::vector<IMUSample>& imu_samples,
    int stride,
    const GroundFilterParams& filter_params,
    const MatchingParams& match_params,
    const PoseCandidate& search_base_pose,
    std::vector<float>& out_dynamic_pts,
    PoseCandidate& out_best_pose,
    float& out_best_cost,
    int* out_inlier_count,
    float* out_inlier_cost,
    const std::vector<float>* extra_base_link_points,
    bool extract_dynamic
)
{
    std::lock_guard<std::recursive_mutex> lock(solver_mutex_);
    const int h_obstacle_count = prepareObstacleCloud(
        timed_points,
        h_transform,
        deskew_params,
        imu_samples,
        stride,
        filter_params,
        extra_base_link_points
    );
    if (h_obstacle_count == 0) {
        out_best_cost = std::numeric_limits<float>::max();
        if (out_inlier_count != nullptr) {
            *out_inlier_count = 0;
        }
        if (out_inlier_cost != nullptr) {
            *out_inlier_cost = 1.0f;
        }
        out_dynamic_pts.clear();
        return false;
    }

    return matchObstacleCloud(
        h_obstacle_count,
        match_params,
        search_base_pose,
        out_dynamic_pts,
        out_best_pose,
        out_best_cost,
        out_inlier_count,
        out_inlier_cost,
        extract_dynamic
    );
}

float PoseSolver::computeEffectiveWeight2D(const MatchingParams& match_params) const
{
    if (num_2d_obstacles_ >= match_params.min_2d_points &&
        num_3d_obstacles_ >= match_params.min_3d_points) {
        const float fov_scale = std::min(
            1.0f,
            static_cast<float>(num_2d_obstacles_) /
                static_cast<float>(match_params.target_2d_points)
        );
        return match_params.weight_2d_lidar * fov_scale;
    }
    if (num_2d_obstacles_ >= match_params.min_2d_points &&
        num_3d_obstacles_ < match_params.min_3d_points) {
        return 1.0f;
    }
    return 0.0f;
}

int PoseSolver::computeCoarsePointStride() const
{
    if (num_3d_obstacles_ > 1500) {
        return std::max(1, num_3d_obstacles_ / 1200);
    }
    return 1;
}

bool PoseSolver::matchObstacleCloud(
    int h_obstacle_count,
    const MatchingParams& match_params,
    const PoseCandidate& search_base_pose,
    std::vector<float>& out_dynamic_pts,
    PoseCandidate& out_best_pose,
    float& out_best_cost,
    int* out_inlier_count,
    float* out_inlier_cost,
    bool extract_dynamic
)
{
    std::lock_guard<std::recursive_mutex> lock(solver_mutex_);
    out_dynamic_pts.clear();
    bool pose_matched = false;
    auto t_match_start = std::chrono::high_resolution_clock::now();
    if (num_3d_obstacles_ >= match_params.min_3d_points ||
        num_2d_obstacles_ >= match_params.min_2d_points) {
        // Compute dynamic 2D LiDAR weighting based on active point count and FoV coverage
        const float effective_weight_2d = computeEffectiveWeight2D(match_params);
        last_weight_2d_ = effective_weight_2d;

        int local_inliers = 0;
        float local_inlier_cost = 1.0f;

        auto run_local_match = [&](const float* d_obs_ptr,
                                   int active_total_count,
                                   int active_3d_count,
                                   int active_2d_count,
                                   float weight_2d,
                                   float* out_c3d,
                                   float* out_c2d) -> bool {
            out_dynamic_pts.clear();
            PoseCandidate coarse_pose{};
            float coarse_cost = std::numeric_limits<float>::max();
            std::vector<float> ignored_dynamic;

            int* step_inlier_count = nullptr;
            float* step_inlier_cost = nullptr;
            bool output_coarse_dynamic = false;
            int coarse_point_stride = (active_3d_count > 1500)
                                          ? std::max(1, active_3d_count / 1200)
                                          : 1;
            if (!match_params.fine_refine) {
                step_inlier_count = &local_inliers;
                step_inlier_cost = &local_inlier_cost;
                output_coarse_dynamic = extract_dynamic;
                coarse_point_stride = 1;
            }

            std::vector<float>& target_dynamic_buf =
                output_coarse_dynamic ? out_dynamic_pts : ignored_dynamic;

            bool matched = false;
            if (use_esdf_) {
                matched = launchESDFMatcher(
                    stream_,
                    d_obs_ptr,
                    active_total_count,
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
                    target_dynamic_buf,
                    output_coarse_dynamic,
                    step_inlier_count,
                    step_inlier_cost,
                    match_params.inlier_dist_thresh,
                    active_3d_count,
                    active_2d_count,
                    weight_2d,
                    (!match_params.fine_refine) ? out_c3d : nullptr,
                    (!match_params.fine_refine) ? out_c2d : nullptr,
                    nullptr,
                    coarse_point_stride
                );
            } else {
                matched = launchFieldSDFMatcher(
                    stream_,
                    d_obs_ptr,
                    active_total_count,
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
                    target_dynamic_buf,
                    output_coarse_dynamic,
                    step_inlier_count,
                    step_inlier_cost,
                    match_params.inlier_dist_thresh,
                    nullptr,
                    coarse_point_stride
                );
            }
            out_best_pose = coarse_pose;
            out_best_cost = coarse_cost;
            if (matched && match_params.fine_refine) {
                float fine_xy = match_params.step_xy;
                float fine_yaw = match_params.step_yaw;
                for (int level = 0; level < match_params.fine_refine_levels && matched; ++level) {
                    fine_xy /= 5.0f;
                    fine_yaw /= 5.0f;
                    const PoseCandidate center = out_best_pose;
                    const bool is_final_level = (level + 1 == match_params.fine_refine_levels);
                    const bool extract_level_dynamic = is_final_level && extract_dynamic;

                    int* fine_inlier_count = is_final_level ? &local_inliers : nullptr;
                    float* fine_inlier_cost = is_final_level ? &local_inlier_cost : nullptr;

                    if (use_esdf_) {
                        matched = launchESDFMatcher(
                            stream_,
                            d_obs_ptr,
                            active_total_count,
                            center,
                            2.0f * fine_xy,
                            2.0f * fine_xy,
                            fine_xy,
                            2.0f * fine_yaw,
                            fine_yaw,
                            match_params.max_dist_thresh,
                            match_params.dynamic_dist_thresh,
                            match_params.field_min_x,
                            match_params.field_max_x,
                            match_params.field_min_y,
                            match_params.field_max_y,
                            out_best_pose,
                            out_best_cost,
                            out_dynamic_pts,
                            extract_level_dynamic,
                            fine_inlier_count,
                            fine_inlier_cost,
                            match_params.inlier_dist_thresh,
                            active_3d_count,
                            active_2d_count,
                            weight_2d,
                            is_final_level ? out_c3d : nullptr,
                            is_final_level ? out_c2d : nullptr
                        );
                    } else {
                        matched = launchFieldSDFMatcher(
                            stream_,
                            d_obs_ptr,
                            active_total_count,
                            center,
                            2.0f * fine_xy,
                            2.0f * fine_xy,
                            fine_xy,
                            2.0f * fine_yaw,
                            fine_yaw,
                            match_params.max_dist_thresh,
                            match_params.dynamic_dist_thresh,
                            match_params.field_min_x,
                            match_params.field_max_x,
                            match_params.field_min_y,
                            match_params.field_max_y,
                            out_best_pose,
                            out_best_cost,
                            out_dynamic_pts,
                            extract_level_dynamic,
                            fine_inlier_count,
                            fine_inlier_cost,
                            match_params.inlier_dist_thresh
                        );
                    }
                }
            }
            return matched;
        };

        auto is_3d_passed = [&](float c3d, int inliers, float inlier_c) -> bool {
            if (num_3d_obstacles_ < match_params.min_3d_points) {
                return false;
            }
            const bool inlier_ok =
                (inliers >= match_params.min_inliers) &&
                (inlier_c < match_params.inlier_cost_thresh) &&
                (static_cast<float>(inliers) / static_cast<float>(num_3d_obstacles_) >=
                 match_params.min_inlier_ratio) &&
                (c3d < match_params.max_inlier_accept_cost);
            return (c3d < match_params.cost_threshold_3d) || inlier_ok;
        };

        auto is_2d_passed = [&](float c2d) -> bool {
            if (num_2d_obstacles_ < match_params.min_2d_points) {
                return false;
            }
            return c2d < match_params.cost_threshold_2d;
        };

        if (num_3d_obstacles_ >= match_params.min_3d_points) {
            pose_matched = run_local_match(
                d_obstacle_,
                h_obstacle_count,
                num_3d_obstacles_,
                num_2d_obstacles_,
                effective_weight_2d,
                &last_cost_3d_,
                &last_cost_2d_
            );

            if (use_esdf_ && pose_matched && effective_weight_2d > 0.0f &&
                num_2d_obstacles_ >= match_params.min_2d_points) {
                bool pass_3d = is_3d_passed(last_cost_3d_, local_inliers, local_inlier_cost);
                const bool pass_2d = is_2d_passed(last_cost_2d_);

                if (!pass_2d) {
                    // 2D LiDAR failed its threshold: re-run with 3D LiDAR only so corrupted
                    // 2D scans or unmapped 2D obstacles cannot pull the 3D pose off-course.
                    last_weight_2d_ = 0.0f;
                    pose_matched = run_local_match(
                        d_obstacle_,
                        num_3d_obstacles_,
                        num_3d_obstacles_,
                        0,
                        0.0f,
                        &last_cost_3d_,
                        nullptr
                    );
                    pass_3d = pose_matched &&
                              is_3d_passed(last_cost_3d_, local_inliers, local_inlier_cost);
                }

                if (!pass_3d) {
                    // 3D LiDAR failed its threshold: re-run with 2D LiDAR only so occluded
                    // or degraded 3D scans cannot pull the valid 2D pose off-course.
                    last_weight_2d_ = 1.0f;
                    pose_matched = run_local_match(
                        d_obstacle_ + num_3d_obstacles_ * 3,
                        num_2d_obstacles_,
                        0,
                        num_2d_obstacles_,
                        1.0f,
                        nullptr,
                        &last_cost_2d_
                    );
                }
            }
        } else {
            // Only 2D LiDAR has sufficient obstacle points
            last_cost_3d_ = match_params.max_dist_thresh;
            last_weight_2d_ = 1.0f;
            pose_matched = run_local_match(
                d_obstacle_ + num_3d_obstacles_ * 3,
                num_2d_obstacles_,
                0,
                num_2d_obstacles_,
                1.0f,
                nullptr,
                &last_cost_2d_
            );
        }

        if (out_inlier_count != nullptr) {
            *out_inlier_count = local_inliers;
        }
        if (out_inlier_cost != nullptr) {
            *out_inlier_cost = local_inlier_cost;
        }
    } else {
        out_best_cost = std::numeric_limits<float>::max();
        if (out_inlier_count != nullptr) {
            *out_inlier_count = 0;
        }
        if (out_inlier_cost != nullptr) {
            *out_inlier_cost = 1.0f;
        }
        out_dynamic_pts.clear();
    }

    auto t_match_end = std::chrono::high_resolution_clock::now();
    last_metrics_.time_matching_ms =
        std::chrono::duration<double, std::milli>(t_match_end - t_match_start).count();
    size_t inlier_bytes = 0;
    if (out_inlier_count != nullptr) {
        inlier_bytes = sizeof(int) + sizeof(float);
    }
    last_metrics_.d2h_transfer_bytes = out_dynamic_pts.size() * sizeof(float) + inlier_bytes;

    return pose_matched;
}

int PoseSolver::prepareObstacleCloud(
    const std::vector<TimedScanPoint>& timed_points,
    const float h_transform[12],
    const DeskewParams& deskew_params,
    const std::vector<IMUSample>& imu_samples,
    int stride,
    const GroundFilterParams& filter_params,
    const std::vector<float>* extra_base_link_points
)
{
    std::lock_guard<std::recursive_mutex> lock(solver_mutex_);
    int num_points = static_cast<int>(timed_points.size());
    ground_count_     = 0;
    obstacle_count_   = 0;
    num_3d_obstacles_ = 0;
    num_2d_obstacles_ = 0;
    const bool has_extra = (extra_base_link_points != nullptr && !extra_base_link_points->empty());
    if (num_points > max_points_ || (num_points == 0 && !has_extra)) {
        return 0;
    }

    last_metrics_ = ProfileMetrics{};
    auto t_gf_start = std::chrono::high_resolution_clock::now();

    int h_ground_count   = 0;
    int h_obstacle_count = 0;

    if (num_points > 0) {
        std::memcpy(h_timed_in_, timed_points.data(), num_points * sizeof(TimedScanPoint));
        cudaMemcpyAsync(
            d_timed_in_,
            h_timed_in_,
            num_points * sizeof(TimedScanPoint),
            cudaMemcpyHostToDevice,
            stream_
        );
        cudaMemcpyAsync(
            d_transform_, h_transform, 12 * sizeof(float), cudaMemcpyHostToDevice, stream_
        );

        const size_t total_samples = imu_samples.size();
        const int num_samples =
            static_cast<int>(std::min(total_samples, static_cast<size_t>(kMaxImuSamples)));
        const size_t start_offset = total_samples - static_cast<size_t>(num_samples);
        if (num_samples > 0) {
            cudaMemcpyAsync(
                d_imu_samples_,
                imu_samples.data() + start_offset,
                num_samples * sizeof(IMUSample),
                cudaMemcpyHostToDevice,
                stream_
            );
        }

        launchGroundFilterWithDeskew(
            stream_,
            d_timed_in_,
            d_ground_,
            d_obstacle_,
            d_transform_,
            num_points,
            stride,
            deskew_params,
            d_imu_samples_,
            num_samples,
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
    }
    auto t_gf_end = std::chrono::high_resolution_clock::now();
    last_metrics_.time_ground_filter_ms =
        std::chrono::duration<double, std::milli>(t_gf_end - t_gf_start).count();

    num_3d_obstacles_ = h_obstacle_count;
    num_2d_obstacles_ = 0;

    // Append 2D LiDAR points in base_link frame into the obstacle buffer on GPU
    if (extra_base_link_points != nullptr && !extra_base_link_points->empty()) {
        const int extra_points = static_cast<int>(extra_base_link_points->size() / 3);
        if (h_obstacle_count + extra_points <= max_points_) {
            cudaMemcpyAsync(
                d_obstacle_ + h_obstacle_count * 3,
                extra_base_link_points->data(),
                extra_points * 3 * sizeof(float),
                cudaMemcpyHostToDevice,
                stream_
            );
            num_2d_obstacles_ = extra_points;
            h_obstacle_count += extra_points;
            cudaStreamSynchronize(stream_);
        }
    }

    ground_count_   = h_ground_count;
    obstacle_count_ = h_obstacle_count;
    return h_obstacle_count;
}

int PoseSolver::prepareObstacleCloud(
    const std::vector<float>& h_raw_cloud,
    const float h_transform[12],
    const GroundFilterParams& filter_params,
    const std::vector<float>* extra_base_link_points
)
{
    std::lock_guard<std::recursive_mutex> lock(solver_mutex_);
    int num_points  = static_cast<int>(h_raw_cloud.size() / 3);
    ground_count_     = 0;
    obstacle_count_   = 0;
    num_3d_obstacles_ = 0;
    num_2d_obstacles_ = 0;
    const bool has_extra = (extra_base_link_points != nullptr && !extra_base_link_points->empty());
    if (num_points > max_points_ || (num_points == 0 && !has_extra)) {
        return 0;
    }

    last_metrics_ = ProfileMetrics{};
    auto t_gf_start = std::chrono::high_resolution_clock::now();

    int h_ground_count   = 0;
    int h_obstacle_count = 0;

    if (num_points > 0) {
        std::memcpy(h_in_, h_raw_cloud.data(), num_points * 3 * sizeof(float));

        cudaMemcpyAsync(
            d_in_, h_in_, num_points * 3 * sizeof(float), cudaMemcpyHostToDevice, stream_
        );
        cudaMemcpyAsync(
            d_transform_, h_transform, 12 * sizeof(float), cudaMemcpyHostToDevice, stream_
        );

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
    }
    auto t_gf_end = std::chrono::high_resolution_clock::now();
    last_metrics_.time_ground_filter_ms =
        std::chrono::duration<double, std::milli>(t_gf_end - t_gf_start).count();

    num_3d_obstacles_ = h_obstacle_count;
    num_2d_obstacles_ = 0;

    // Append 2D LiDAR points in base_link frame into the obstacle buffer on GPU
    if (extra_base_link_points != nullptr && !extra_base_link_points->empty()) {
        const int extra_points = static_cast<int>(extra_base_link_points->size() / 3);
        if (h_obstacle_count + extra_points <= max_points_) {
            cudaMemcpyAsync(
                d_obstacle_ + h_obstacle_count * 3,
                extra_base_link_points->data(),
                extra_points * 3 * sizeof(float),
                cudaMemcpyHostToDevice,
                stream_
            );
            num_2d_obstacles_ = extra_points;
            h_obstacle_count += extra_points;
            cudaStreamSynchronize(stream_);
        }
    }

    ground_count_   = h_ground_count;
    obstacle_count_ = h_obstacle_count;
    return h_obstacle_count;
}

void PoseSolver::copyFilteredClouds(std::vector<float>* ground, std::vector<float>* obstacle)
{
    std::lock_guard<std::recursive_mutex> lock(solver_mutex_);
    if (ground) {
        ground->resize(static_cast<size_t>(ground_count_) * 3);
        if (ground_count_ > 0) {
            cudaMemcpy(
                ground->data(), d_ground_, ground->size() * sizeof(float), cudaMemcpyDeviceToHost
            );
        }
    }
    if (obstacle) {
        obstacle->resize(static_cast<size_t>(obstacle_count_) * 3);
        if (obstacle_count_ > 0) {
            cudaMemcpy(
                obstacle->data(),
                d_obstacle_,
                obstacle->size() * sizeof(float),
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
    std::vector<PoseCandidate>& out_top_poses,
    int top_k
)
{
    std::lock_guard<std::recursive_mutex> lock(solver_mutex_);
    out_top_poses.clear();
    if (top_k < 1) {
        top_k = 1;
    }

    std::vector<float> dummy_dynamic;
    std::vector<float> all_costs;
    PoseCandidate best_pose{};
    float best_cost = std::numeric_limits<float>::max();

    const float effective_weight_2d = computeEffectiveWeight2D(match_params);
    const int coarse_point_stride   = computeCoarsePointStride();
    last_weight_2d_ = effective_weight_2d;

    bool ok = false;
    if (use_esdf_) {
        ok = launchESDFMatcher(
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
            best_pose,
            best_cost,
            dummy_dynamic,
            false,
            nullptr,
            nullptr,
            match_params.inlier_dist_thresh,
            num_3d_obstacles_,
            num_2d_obstacles_,
            effective_weight_2d,
            nullptr,
            nullptr,
            &all_costs,
            coarse_point_stride
        );
    } else {
        ok = launchFieldSDFMatcher(
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
            best_pose,
            best_cost,
            dummy_dynamic,
            false,
            nullptr,
            nullptr,
            match_params.inlier_dist_thresh,
            &all_costs,
            coarse_point_stride
        );
    }

    if (!ok || all_costs.empty()) {
        return false;
    }

    const SearchGrid grid = createSearchGrid(
        base_pose, range_x, range_y, step_xy, range_yaw, step_yaw
    );
    const int num_candidates = static_cast<int>(all_costs.size());
    if (grid.totalCandidates() != num_candidates) {
        out_top_poses.push_back(best_pose);
        return true;
    }

    // Sort candidate indices in ascending order of cost
    std::vector<int> sorted_indices(static_cast<size_t>(num_candidates));
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&all_costs](int lhs, int rhs) {
        return all_costs[static_cast<size_t>(lhs)] < all_costs[static_cast<size_t>(rhs)];
    });

    // Apply Non-Maximum Suppression (NMS) to select top K spatially/angularly distinct poses
    constexpr float NMS_MIN_DIST_XY_M     = 1.0f;
    constexpr float NMS_MIN_ANGLE_YAW_RAD = 0.5f;
    out_top_poses.reserve(static_cast<size_t>(top_k));

    for (int candidate_idx : sorted_indices) {
        if (!std::isfinite(all_costs[static_cast<size_t>(candidate_idx)])) {
            continue;
        }
        PoseCandidate cand = grid.indexToPose(candidate_idx);
        cand.yaw           = normalizeAngle(cand.yaw);

        bool suppressed = false;
        for (const auto& selected : out_top_poses) {
            const float dist_xy  = std::hypot(cand.x - selected.x, cand.y - selected.y);
            const float diff_yaw = std::abs(normalizeAngle(cand.yaw - selected.yaw));
            if (dist_xy < NMS_MIN_DIST_XY_M && diff_yaw < NMS_MIN_ANGLE_YAW_RAD) {
                suppressed = true;
                break;
            }
        }

        if (!suppressed) {
            out_top_poses.push_back(cand);
            if (static_cast<int>(out_top_poses.size()) >= top_k) {
                break;
            }
        }
    }

    if (out_top_poses.empty()) {
        out_top_poses.push_back(best_pose);
    }
    return true;
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
    const float effective_weight_2d = computeEffectiveWeight2D(match_params);
    const int coarse_point_stride   = computeCoarsePointStride();
    last_weight_2d_ = effective_weight_2d;
    if (use_esdf_) {
        return launchESDFMatcher(
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
            false,
            nullptr,
            nullptr,
            match_params.inlier_dist_thresh,
            num_3d_obstacles_,
            num_2d_obstacles_,
            effective_weight_2d,
            nullptr,
            nullptr,
            nullptr,
            coarse_point_stride
        );
    }

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
        false,
        nullptr,
        nullptr,
        match_params.inlier_dist_thresh,
        nullptr,
        coarse_point_stride
    );
}

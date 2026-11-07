// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <sensor_msgs/msg/laser_scan.hpp>
#include "gn10_pointcloud_localization/scan_accumulator.hpp"

namespace gn10_pointcloud_localization {

struct LaserScanDeskewConfig {
    bool enable_deskew{true};
    float range_max_limit{12.0f};
    float robot_radius{0.8f};
    float vx_local{0.0f};
    float vy_local{0.0f};
    double time_offset_sec{0.0};
};

/**
 * @brief Applies per-beam IMU rotational deskew and translational velocity deskew to a 2D
 *        LaserScan message, transforming valid beams into the base_link frame.
 *
 * @param scan Input LaserScan message
 * @param tf_x Sensor translation X in base_link frame [m]
 * @param tf_y Sensor translation Y in base_link frame [m]
 * @param tf_z Sensor translation Z in base_link frame [m]
 * @param tf_yaw Sensor mounting yaw angle in base_link frame [rad]
 * @param yaw_history Integrated IMU yaw history (YawHistory)
 * @param config Deskew configuration (enable flag, max range, robot footprint radius, local velocity, time offset)
 * @param out_ref_stamp_ns Output reference timestamp at the end (or start) of the scan [ns]
 * @return std::vector<float> Deskewed point cloud in base_link frame (interleaved x, y, z)
 */
inline std::vector<float> deskewLaserScan(
    const sensor_msgs::msg::LaserScan& scan,
    double tf_x,
    double tf_y,
    double tf_z,
    double tf_yaw,
    const YawHistory& yaw_history,
    const LaserScanDeskewConfig& config,
    int64_t& out_ref_stamp_ns
)
{
    const size_t num_ranges = scan.ranges.size();
    if (num_ranges == 0) {
        out_ref_stamp_ns = 0;
        return {};
    }

    const int64_t time_offset_ns =
        static_cast<int64_t>(std::round(config.time_offset_sec * 1e9));

    const int64_t start_stamp_ns =
        static_cast<int64_t>(scan.header.stamp.sec) * 1000000000LL +
        static_cast<int64_t>(scan.header.stamp.nanosec) +
        time_offset_ns;

    double time_inc = 0.0;
    if (std::isfinite(scan.time_increment) && scan.time_increment > 0.0) {
        time_inc = static_cast<double>(scan.time_increment);
    } else if (std::isfinite(scan.scan_time) && scan.scan_time > 0.0 && num_ranges > 1) {
        time_inc = static_cast<double>(scan.scan_time) / static_cast<double>(num_ranges - 1);
    }
    if (num_ranges > 1 && time_inc > 0.0) {
        const double total_scan_duration = static_cast<double>(num_ranges - 1) * time_inc;
        if (total_scan_duration > 1.0) {
            time_inc = 0.0;
        }
    }

    const int64_t end_stamp_ns =
        start_stamp_ns + static_cast<int64_t>(std::round((num_ranges - 1) * time_inc * 1e9));

    out_ref_stamp_ns = (time_inc > 0.0) ? end_stamp_ns : start_stamp_ns;

    double ref_imu_angle = 0.0;
    const bool has_ref_imu =
        config.enable_deskew && (time_inc > 0.0) &&
        yaw_history.get_angle_at(out_ref_stamp_ns, ref_imu_angle);

    const bool has_trans_deskew =
        config.enable_deskew && (time_inc > 0.0) &&
        (config.vx_local != 0.0f || config.vy_local != 0.0f);

    const float tf_cos = static_cast<float>(std::cos(tf_yaw));
    const float tf_sin = static_cast<float>(std::sin(tf_yaw));

    const float angle_min = scan.angle_min;
    const float angle_inc = scan.angle_increment;
    const float range_min = scan.range_min;
    const float range_max = std::min(scan.range_max, config.range_max_limit);
    const float radius_sq = config.robot_radius * config.robot_radius;

    std::vector<float> points;
    points.reserve(num_ranges * 3);

    for (size_t i = 0; i < num_ranges; ++i) {
        const float r = scan.ranges[i];
        if (!std::isfinite(r) || r < range_min || r > range_max) {
            continue;
        }

        const float beam_angle = angle_min + static_cast<float>(i) * angle_inc;
        const float xl = r * std::cos(beam_angle);
        const float yl = r * std::sin(beam_angle);

        // Transform beam endpoint from sensor frame to base_link frame via static extrinsic TF
        float xb = tf_cos * xl - tf_sin * yl + static_cast<float>(tf_x);
        float yb = tf_sin * xl + tf_cos * yl + static_cast<float>(tf_y);
        const float zb = static_cast<float>(tf_z);

        const int64_t beam_stamp_ns =
            start_stamp_ns + static_cast<int64_t>(std::round(i * time_inc * 1e9));

        // Per-beam IMU rotational deskew (project to reference timestamp orientation)
        if (has_ref_imu) {
            double beam_imu_angle = 0.0;
            if (yaw_history.get_angle_at(beam_stamp_ns, beam_imu_angle)) {
                const float delta_yaw = static_cast<float>(beam_imu_angle - ref_imu_angle);
                const float cos_d = std::cos(delta_yaw);
                const float sin_d = std::sin(delta_yaw);
                const float x_rot = cos_d * xb - sin_d * yb;
                const float y_rot = sin_d * xb + cos_d * yb;
                xb = x_rot;
                yb = y_rot;
            }
        }

        // Per-beam translational deskew (project to reference timestamp position)
        if (has_trans_deskew) {
            const float dt_sec =
                static_cast<float>(beam_stamp_ns - out_ref_stamp_ns) * 1e-9f;
            const float dx = config.vx_local * dt_sec;
            const float dy = config.vy_local * dt_sec;
            xb += dx;
            yb += dy;
        }

        // Exclude points inside the robot footprint radius
        if (xb * xb + yb * yb <= radius_sq) {
            continue;
        }

        points.push_back(xb);
        points.push_back(yb);
        points.push_back(zb);
    }

    return points;
}

/**
 * @brief Spatiotemporally aligns a 2D point cloud from source_stamp_ns to a target reference
 *        timestamp (e.g., the 3D LiDAR scan end timestamp reference_stamp) using IMU rotation
 *        and local translational velocity.
 *
 * @param points Input point cloud in base_link frame (interleaved x, y, z)
 * @param source_stamp_ns Source reference timestamp of the 2D scan [ns]
 * @param target_stamp_ns Target reference timestamp to align to [ns]
 * @param yaw_history Integrated IMU yaw history
 * @param vx_local Local linear velocity X in base_link frame at the target timestamp [m/s]
 * @param vy_local Local linear velocity Y in base_link frame at the target timestamp [m/s]
 * @return std::vector<float> Timestamp-aligned point cloud in base_link frame (interleaved x, y, z)
 */
inline std::vector<float> alignPointsToTimestamp(
    const std::vector<float>& points,
    int64_t source_stamp_ns,
    int64_t target_stamp_ns,
    const YawHistory& yaw_history,
    float vx_local,
    float vy_local
)
{
    if (points.empty() || source_stamp_ns <= 0 || target_stamp_ns <= 0) {
        return points;
    }

    const double dt_sec = static_cast<double>(target_stamp_ns - source_stamp_ns) * 1e-9;
    if (std::abs(dt_sec) < 1e-4 && vx_local == 0.0f && vy_local == 0.0f) {
        return points;
    }

    double delta_yaw = 0.0;
    yaw_history.delta(source_stamp_ns, target_stamp_ns, delta_yaw);

    const float cos_d = static_cast<float>(std::cos(delta_yaw));
    const float sin_d = static_cast<float>(std::sin(delta_yaw));
    const float dx = static_cast<float>(vx_local * dt_sec);
    const float dy = static_cast<float>(vy_local * dt_sec);

    std::vector<float> aligned;
    aligned.reserve(points.size());
    const size_t num_pts = points.size() / 3;

    for (size_t i = 0; i < num_pts; ++i) {
        const float x = points[i * 3 + 0];
        const float y = points[i * 3 + 1];
        const float z = points[i * 3 + 2];

        // R(-delta_yaw) * P - V_local * dt
        const float x_rot = cos_d * x + sin_d * y;
        const float y_rot = -sin_d * x + cos_d * y;

        aligned.push_back(x_rot - dx);
        aligned.push_back(y_rot - dy);
        aligned.push_back(z);
    }

    return aligned;
}

}  // namespace gn10_pointcloud_localization

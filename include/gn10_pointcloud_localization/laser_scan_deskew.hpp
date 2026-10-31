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
};

/**
 * @brief 2D LaserScan にビーム単位の IMU 回転歪み補正 (Deskew) を適用し、base_link 座標系の点群に変換する
 *
 * @param scan 入力 LaserScan メッセージ
 * @param tf_x base_link から見たセンサの並進 X [m]
 * @param tf_y base_link から見たセンサの並進 Y [m]
 * @param tf_z base_link から見たセンサの並進 Z [m]
 * @param tf_yaw base_link から見たセンサの取り付け Yaw 角 [rad]
 * @param yaw_history IMU 積分履歴 (YawHistory)
 * @param config 設定パラメータ (Deskew有効化、最大距離、ロボット半径)
 * @param out_ref_stamp_ns スキャン末尾の基準時刻 [ns]
 * @return std::vector<float> base_link 座標系の点群 (x, y, z のインターリーブ配列)
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

    const int64_t start_stamp_ns =
        static_cast<int64_t>(scan.header.stamp.sec) * 1000000000LL +
        static_cast<int64_t>(scan.header.stamp.nanosec);

    const double time_inc =
        (std::isfinite(scan.time_increment) && scan.time_increment > 0.0)
            ? static_cast<double>(scan.time_increment)
            : 0.0;

    const int64_t end_stamp_ns =
        start_stamp_ns + static_cast<int64_t>((num_ranges - 1) * time_inc * 1e9);

    out_ref_stamp_ns = (time_inc > 0.0) ? end_stamp_ns : start_stamp_ns;

    double ref_imu_angle = 0.0;
    const bool has_ref_imu =
        config.enable_deskew && (time_inc > 0.0) &&
        yaw_history.get_angle_at(out_ref_stamp_ns, ref_imu_angle);

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

        // 静的 TF による base_link への変換
        float xb = tf_cos * xl - tf_sin * yl + static_cast<float>(tf_x);
        float yb = tf_sin * xl + tf_cos * yl + static_cast<float>(tf_y);
        const float zb = static_cast<float>(tf_z);

        // ビームごとの IMU Deskew 回転補正 (基準時刻姿勢への復元)
        if (has_ref_imu) {
            const int64_t beam_stamp_ns =
                start_stamp_ns + static_cast<int64_t>(i * time_inc * 1e9);
            double beam_imu_angle = 0.0;
            if (yaw_history.get_angle_at(beam_stamp_ns, beam_imu_angle)) {
                const float delta_yaw = static_cast<float>(ref_imu_angle - beam_imu_angle);
                const float cos_d = std::cos(delta_yaw);
                const float sin_d = std::sin(delta_yaw);
                const float x_rot = cos_d * xb - sin_d * yb;
                const float y_rot = sin_d * xb + cos_d * yb;
                xb = x_rot;
                yb = y_rot;
            }
        }

        // ロボットフットプリント内の除外
        if (xb * xb + yb * yb <= radius_sq) {
            continue;
        }

        points.push_back(xb);
        points.push_back(yb);
        points.push_back(zb);
    }

    return points;
}

}  // namespace gn10_pointcloud_localization

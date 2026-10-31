// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <sensor_msgs/msg/laser_scan.hpp>
#include "gn10_pointcloud_localization/laser_scan_deskew.hpp"

using namespace gn10_pointcloud_localization;

void testStationaryDeskew()
{
    std::cout << "[Test] testStationaryDeskew starting..." << std::endl;

    sensor_msgs::msg::LaserScan scan;
    scan.header.stamp.sec = 1000;
    scan.header.stamp.nanosec = 0;
    scan.angle_min = -static_cast<float>(M_PI) / 4.0f; // -45 deg
    scan.angle_max =  static_cast<float>(M_PI) / 4.0f; // +45 deg
    scan.angle_increment = static_cast<float>(M_PI) / 180.0f; // 1 deg step
    scan.time_increment = 0.001f; // 1ms per beam (~90ms scan)
    scan.scan_time = 0.09f;
    scan.range_min = 0.1f;
    scan.range_max = 10.0f;

    // 前方 x = 2.0m にある直線平坦壁 (y in [-2, 2])
    // r * cos(theta) = 2.0  => r = 2.0 / cos(theta)
    const size_t num_beams = static_cast<size_t>(
        std::round((scan.angle_max - scan.angle_min) / scan.angle_increment) + 1
    );
    scan.ranges.resize(num_beams);
    for (size_t i = 0; i < num_beams; ++i) {
        float theta = scan.angle_min + static_cast<float>(i) * scan.angle_increment;
        scan.ranges[i] = 2.0f / std::cos(theta);
    }

    YawHistory imu;
    int64_t start_ns = 1000LL * 1000000000LL;
    imu.add(start_ns - 10000000LL, 0.0);
    imu.add(start_ns + 100000000LL, 0.0);

    LaserScanDeskewConfig config;
    config.enable_deskew = true;
    config.range_max_limit = 12.0f;
    config.robot_radius = 0.5f;

    int64_t out_ref_ns = 0;
    std::vector<float> points = deskewLaserScan(
        scan, 0.0, 0.0, 0.1, 0.0, imu, config, out_ref_ns
    );

    assert(!points.empty());
    assert(points.size() / 3 == num_beams);

    // 静止時は全点の x 座標が 2.0m ± 1e-4 になるはず
    for (size_t i = 0; i < points.size() / 3; ++i) {
        float x = points[i * 3 + 0];
        assert(std::abs(x - 2.0f) < 1e-3f);
    }

    std::cout << "[Test] testStationaryDeskew PASSED!" << std::endl;
}

void testHighSpeedRotationDeskew()
{
    std::cout << "[Test] testHighSpeedRotationDeskew starting..." << std::endl;

    sensor_msgs::msg::LaserScan scan;
    scan.header.stamp.sec = 2000;
    scan.header.stamp.nanosec = 0;
    scan.angle_min = -static_cast<float>(M_PI) / 4.0f;
    scan.angle_max =  static_cast<float>(M_PI) / 4.0f;
    scan.angle_increment = static_cast<float>(M_PI) / 180.0f; // 1 deg
    scan.time_increment = 0.001f; // 1ms per beam
    scan.range_min = 0.1f;
    scan.range_max = 10.0f;

    const size_t num_beams = static_cast<size_t>(
        std::round((scan.angle_max - scan.angle_min) / scan.angle_increment) + 1
    );
    scan.ranges.resize(num_beams);

    // ロボットが角速度 omega = 2.0 rad/s (~115 deg/s) で反時計回りに高速旋回中
    // 基準時刻 t_ref はスキャン末尾時刻 (i = num_beams - 1)
    // ビーム i の照射時刻 t_i において、ロボットの向きは theta_robot(t_i) = omega * (t_i - t_ref)
    // 壁はワールド座標系で x_world = 2.0m に存在
    const double omega = 2.0; // rad/s
    const double dt_beam = scan.time_increment;
    const double t_ref = (num_beams - 1) * dt_beam;

    for (size_t i = 0; i < num_beams; ++i) {
        double t_i = i * dt_beam;
        double rot_offset = omega * (t_i - t_ref); // 基準姿勢からのロボット回転角
        float beam_angle_sensor = scan.angle_min + static_cast<float>(i) * scan.angle_increment;
        // ワールド基準でのビームの照射角: phi_world = rot_offset + beam_angle_sensor
        double phi_world = rot_offset + beam_angle_sensor;
        // x_world = 2.0m の壁にヒットする距離: r = 2.0 / cos(phi_world)
        scan.ranges[i] = static_cast<float>(2.0 / std::cos(phi_world));
    }

    // IMU 履歴の構築 (omega = 2.0 rad/s)
    YawHistory imu;
    int64_t start_ns = 2000LL * 1000000000LL;
    int64_t end_ns = start_ns + static_cast<int64_t>(t_ref * 1e9);
    for (int step = -10; step <= static_cast<int>(num_beams) + 10; step += 10) {
        int64_t stamp = start_ns + static_cast<int64_t>(step * dt_beam * 1e9);
        imu.add(stamp, omega);
    }

    LaserScanDeskewConfig config;
    config.range_max_limit = 12.0f;
    config.robot_radius = 0.5f;

    // 1. Deskew 未適用 (OFF) の場合: 旋回歪みにより x 座標が 2.0m から大きく乖離する
    config.enable_deskew = false;
    int64_t out_ref_ns_no_deskew = 0;
    std::vector<float> pts_no_deskew = deskewLaserScan(
        scan, 0.0, 0.0, 0.1, 0.0, imu, config, out_ref_ns_no_deskew
    );
    float max_error_no_deskew = 0.0f;
    for (size_t i = 0; i < pts_no_deskew.size() / 3; ++i) {
        float x = pts_no_deskew[i * 3 + 0];
        max_error_no_deskew = std::max(max_error_no_deskew, std::abs(x - 2.0f));
    }
    std::cout << "  [No Deskew] Max error from wall (x=2.0m): " << max_error_no_deskew << " m" << std::endl;
    // 歪みにより数cm以上の誤差が生じることを確認
    assert(max_error_no_deskew > 0.05f);

    // 2. Deskew 適用 (ON) の場合: 歪みが補正され、全点が基準時刻の平坦壁 (x=2.0m) に一致する
    config.enable_deskew = true;
    int64_t out_ref_ns_deskew = 0;
    std::vector<float> pts_deskew = deskewLaserScan(
        scan, 0.0, 0.0, 0.1, 0.0, imu, config, out_ref_ns_deskew
    );
    assert(out_ref_ns_deskew == end_ns);

    float max_error_deskew = 0.0f;
    for (size_t i = 0; i < pts_deskew.size() / 3; ++i) {
        float x = pts_deskew[i * 3 + 0];
        max_error_deskew = std::max(max_error_deskew, std::abs(x - 2.0f));
    }
    std::cout << "  [With Deskew] Max error from wall (x=2.0m): " << max_error_deskew << " m" << std::endl;
    // 補正後は残差が 2mm 未満に収束
    assert(max_error_deskew < 2e-3f);

    std::cout << "[Test] testHighSpeedRotationDeskew PASSED!" << std::endl;
}

int main()
{
    testStationaryDeskew();
    testHighSpeedRotationDeskew();
    std::cout << "All 2D LiDAR Deskew tests PASSED successfully!" << std::endl;
    return 0;
}


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

    // Straight flat wall at x = 2.0m in front (y in [-2, 2])
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

    // When stationary, all points should have x = 2.0m +/- 1e-3
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

    // Robot rotating counter-clockwise at omega = 2.0 rad/s (~115 deg/s).
    // Reference timestamp t_ref is the end of the scan (i = num_beams - 1).
    // At beam i timestamp t_i, robot heading is theta_robot(t_i) = omega * (t_i - t_ref).
    // The wall lies at x_world = 2.0m in the world frame.
    const double omega = 2.0; // rad/s
    const double dt_beam = scan.time_increment;
    const double t_ref = (num_beams - 1) * dt_beam;

    for (size_t i = 0; i < num_beams; ++i) {
        double t_i = i * dt_beam;
        double rot_offset = omega * (t_i - t_ref); // Robot rotation relative to reference pose
        float beam_angle_sensor = scan.angle_min + static_cast<float>(i) * scan.angle_increment;
        // Beam angle in world frame: phi_world = rot_offset + beam_angle_sensor
        double phi_world = rot_offset + beam_angle_sensor;
        // Range hitting wall at x_world = 2.0m: r = 2.0 / cos(phi_world)
        scan.ranges[i] = static_cast<float>(2.0 / std::cos(phi_world));
    }

    // Build IMU history (omega = 2.0 rad/s)
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

    // 1. Without deskew (OFF): rotational distortion causes x coordinates to deviate from 2.0m
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
    // Verify distortion produces > 5 cm error
    assert(max_error_no_deskew > 0.05f);

    // 2. With deskew (ON): distortion is compensated and all points align with x = 2.0m
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
    // Residual error after deskew converges below 2 mm
    assert(max_error_deskew < 2e-3f);

    std::cout << "[Test] testHighSpeedRotationDeskew PASSED!" << std::endl;
}

void testHighSpeedTranslationDeskew()
{
    std::cout << "[Test] testHighSpeedTranslationDeskew starting..." << std::endl;

    sensor_msgs::msg::LaserScan scan;
    scan.header.stamp.sec = 3000;
    scan.header.stamp.nanosec = 0;
    scan.angle_min = -static_cast<float>(M_PI) / 4.0f; // -45 deg
    scan.angle_max =  static_cast<float>(M_PI) / 4.0f; // +45 deg
    scan.angle_increment = static_cast<float>(M_PI) / 180.0f; // 1 deg
    scan.time_increment = 0.001f; // 1ms per beam (~90ms)
    scan.range_min = 0.1f;
    scan.range_max = 10.0f;

    const size_t num_beams = static_cast<size_t>(
        std::round((scan.angle_max - scan.angle_min) / scan.angle_increment) + 1
    );
    scan.ranges.resize(num_beams);

    // Robot translating forward at vx = 2.0 m/s.
    // Reference timestamp t_ref is the end of the scan (i = num_beams - 1).
    // At beam i timestamp t_i, robot x position is x_robot(t_i) = vx * (t_i - t_ref).
    // The wall lies at x_world = 2.0m in the world frame.
    const double vx = 2.0; // m/s
    const double dt_beam = scan.time_increment;
    const double t_ref = (num_beams - 1) * dt_beam;

    for (size_t i = 0; i < num_beams; ++i) {
        double t_i = i * dt_beam;
        double x_robot_i = vx * (t_i - t_ref);
        float beam_angle = scan.angle_min + static_cast<float>(i) * scan.angle_increment;
        // Range to wall at x_world = 2.0m: r_i = (2.0 - x_robot_i) / cos(beam_angle)
        scan.ranges[i] = static_cast<float>((2.0 - x_robot_i) / std::cos(beam_angle));
    }

    YawHistory imu; // Zero rotation (omega = 0)
    int64_t start_ns = 3000LL * 1000000000LL;
    imu.add(start_ns - 10000000LL, 0.0);
    imu.add(start_ns + 100000000LL, 0.0);

    LaserScanDeskewConfig config;
    config.range_max_limit = 12.0f;
    config.robot_radius = 0.5f;

    // 1. Without translational deskew (vx = 0.0): motion distortion shifts x away from 2.0m
    config.enable_deskew = true;
    config.vx_local = 0.0f;
    config.vy_local = 0.0f;
    int64_t out_ref_ns_no_trans = 0;
    std::vector<float> pts_no_trans = deskewLaserScan(
        scan, 0.0, 0.0, 0.1, 0.0, imu, config, out_ref_ns_no_trans
    );
    float max_error_no_trans = 0.0f;
    for (size_t i = 0; i < pts_no_trans.size() / 3; ++i) {
        float x = pts_no_trans[i * 3 + 0];
        max_error_no_trans = std::max(max_error_no_trans, std::abs(x - 2.0f));
    }
    std::cout << "  [No Trans Deskew] Max error from wall (x=2.0m): " << max_error_no_trans << " m" << std::endl;
    // 90ms * 2.0m/s = 0.18m distortion
    assert(max_error_no_trans > 0.15f);

    // 2. With translational deskew (vx = 2.0 m/s): distortion is compensated and all points align with x = 2.0m
    config.vx_local = static_cast<float>(vx);
    config.vy_local = 0.0f;
    int64_t out_ref_ns_trans = 0;
    std::vector<float> pts_trans = deskewLaserScan(
        scan, 0.0, 0.0, 0.1, 0.0, imu, config, out_ref_ns_trans
    );
    float max_error_trans = 0.0f;
    for (size_t i = 0; i < pts_trans.size() / 3; ++i) {
        float x = pts_trans[i * 3 + 0];
        max_error_trans = std::max(max_error_trans, std::abs(x - 2.0f));
    }
    std::cout << "  [With Trans Deskew] Max error from wall (x=2.0m): " << max_error_trans << " m" << std::endl;
    assert(max_error_trans < 2e-3f);

    std::cout << "[Test] testHighSpeedTranslationDeskew PASSED!" << std::endl;
}

void testTimeOffsetAndAlignment()
{
    std::cout << "[Test] testTimeOffsetAndAlignment starting..." << std::endl;

    // 1. Test time_offset_sec parameter
    sensor_msgs::msg::LaserScan scan;
    scan.header.stamp.sec = 4000;
    scan.header.stamp.nanosec = 0;
    scan.angle_min = 0.0f;
    scan.angle_max = 0.1f;
    scan.angle_increment = 0.1f;
    scan.time_increment = 0.01f;
    scan.range_min = 0.1f;
    scan.range_max = 10.0f;
    scan.ranges = {2.0f, 2.0f};

    YawHistory imu;
    int64_t base_ns = 4000LL * 1000000000LL;
    imu.add(base_ns - 100000000LL, 0.0);
    imu.add(base_ns + 500000000LL, 0.0);

    LaserScanDeskewConfig config;
    config.enable_deskew = true;
    config.time_offset_sec = 0.050; // +50ms offset
    int64_t out_ref_ns = 0;
    deskewLaserScan(scan, 0.0, 0.0, 0.0, 0.0, imu, config, out_ref_ns);

    // Reference timestamp (end of scan) should be 4000s + 50ms + 10ms = 4000.060s
    int64_t expected_ref_ns = base_ns + 50000000LL + 10000000LL;
    assert(out_ref_ns == expected_ref_ns);
    std::cout << "  [TimeOffset] Reference timestamp correctly shifted by time_offset_sec." << std::endl;

    // 2. Test alignPointsToTimestamp (synchronizing 2D cloud to 3D accumulation reference timestamp)
    // Wall point at reference time t_2d (x = 2.0m, y = 0.0m, z = 0.0m)
    std::vector<float> points_2d = {2.0f, 0.0f, 0.0f};
    int64_t t_2d_ns = 4000LL * 1000000000LL;
    // 3D LiDAR accumulation reference timestamp t_3d (+50ms later)
    int64_t t_3d_ns = t_2d_ns + 50000000LL; // +50ms

    // Robot moving forward at vx = 2.0 m/s (advances 0.10m over 50ms).
    // At 3D reference time t_3d, the wall should appear at x = 1.90m in front of the robot.
    std::vector<float> aligned = alignPointsToTimestamp(
        points_2d, t_2d_ns, t_3d_ns, imu, 2.0f, 0.0f
    );
    assert(aligned.size() == 3);
    assert(std::abs(aligned[0] - 1.90f) < 1e-4f);
    assert(std::abs(aligned[1] - 0.0f) < 1e-4f);
    std::cout << "  [AlignPoints] Translational forward projection correctly adjusted distance (2.00m -> 1.90m)." << std::endl;

    // Rotational alignment test (omega = 1.0 rad/s, delta_yaw = 0.05 rad over 50ms)
    YawHistory rot_imu;
    rot_imu.add(t_2d_ns, 0.0);
    rot_imu.add(t_3d_ns, 1.0); // omega = 1.0 rad/s -> integral = 0.05 rad
    std::vector<float> rot_aligned = alignPointsToTimestamp(
        points_2d, t_2d_ns, t_3d_ns, rot_imu, 0.0f, 0.0f
    );
    // Counter-clockwise left turn (+delta_yaw) shifts a front wall toward the right (-y)
    double delta_yaw = 0.0;
    rot_imu.delta(t_2d_ns, t_3d_ns, delta_yaw);
    float exp_x = static_cast<float>(2.0 * std::cos(delta_yaw));
    float exp_y = static_cast<float>(-2.0 * std::sin(delta_yaw));
    assert(std::abs(rot_aligned[0] - exp_x) < 1e-4f);
    assert(std::abs(rot_aligned[1] - exp_y) < 1e-4f);
    std::cout << "  [AlignPoints] Rotational alignment correctly rotated coordinates." << std::endl;

    // 3. Guard test against corrupted time_increment (e.g. uninitialized driver first-scan value)
    sensor_msgs::msg::LaserScan corrupted_scan = scan;
    corrupted_scan.time_increment = 1000.0f; // Corrupted 1000 s/beam value
    config.time_offset_sec = 0.0;
    int64_t corrupted_ref_ns = 0;
    deskewLaserScan(corrupted_scan, 0.0, 0.0, 0.0, 0.0, imu, config, corrupted_ref_ns);
    // Unrealistic scan durations (> 1.0s) are safely ignored (time_inc = 0 -> start_stamp_ns)
    assert(corrupted_ref_ns == base_ns);
    std::cout << "  [TimeIncrementGuard] Corrupted time_increment safely ignored without fake frequency assumptions." << std::endl;

    std::cout << "[Test] testTimeOffsetAndAlignment PASSED!" << std::endl;
}

int main()
{
    testStationaryDeskew();
    testHighSpeedRotationDeskew();
    testHighSpeedTranslationDeskew();
    testTimeOffsetAndAlignment();
    std::cout << "All 2D LiDAR Deskew tests PASSED successfully!" << std::endl;
    return 0;
}

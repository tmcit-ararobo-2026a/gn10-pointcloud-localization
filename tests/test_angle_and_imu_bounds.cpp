// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "gn10_pointcloud_localization/pose_solver.hpp"
#include "gn10_pointcloud_localization/scan_accumulator.hpp"

void testNormalizeAngle()
{
    std::cout << "[Test] testNormalizeAngle starting..." << std::endl;

    // Basic range check [-pi, pi]
    assert(std::abs(normalizeAngle(0.0f)) < 1e-6f);
    assert(std::abs(normalizeAngle(static_cast<float>(M_PI))) <= static_cast<float>(M_PI));
    assert(std::abs(normalizeAngle(-static_cast<float>(M_PI))) <= static_cast<float>(M_PI));

    // 2*pi periodicity check
    float angle_base = 0.5f;
    for (int k = -10; k <= 10; ++k) {
        float test_val = angle_base + static_cast<float>(k * 2.0 * M_PI);
        float norm = normalizeAngle(test_val);
        assert(std::abs(norm - angle_base) < 1e-4f);
    }

    // Huge angle values (prevents while-loop hang when exceeding float32 mantissa precision)
    float huge_angle = 448353440.0f; // 1.79e9 s * 0.25 rad/s
    float huge_norm = normalizeAngle(huge_angle);
    assert(std::isfinite(huge_norm));
    assert(huge_norm >= -static_cast<float>(M_PI) && huge_norm <= static_cast<float>(M_PI));

    float huge_neg_angle = -500000000.0f;
    float huge_neg_norm = normalizeAngle(huge_neg_angle);
    assert(std::isfinite(huge_neg_norm));
    assert(huge_neg_norm >= -static_cast<float>(M_PI) && huge_neg_norm <= static_cast<float>(M_PI));

    // Non-finite values (NaN, Inf) safe fallback without infinite loops
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    assert(normalizeAngle(nan_val) == 0.0f);

    float inf_val = std::numeric_limits<float>::infinity();
    assert(normalizeAngle(inf_val) == 0.0f);

    float neg_inf_val = -std::numeric_limits<float>::infinity();
    assert(normalizeAngle(neg_inf_val) == 0.0f);

    std::cout << "[Test] testNormalizeAngle PASSED!" << std::endl;
}

void testYawHistoryDeltaBounds()
{
    std::cout << "[Test] testYawHistoryDeltaBounds starting..." << std::endl;

    YawHistory imu;
    double delta_rad = 0.0;

    // 1. Empty history returns false
    assert(!imu.delta(1000000000, 1000050000, delta_rad));

    // Append samples (100Hz: 10ms interval)
    int64_t base_stamp = 1793413750000000000LL;
    imu.add(base_stamp + 0, 0.2);
    imu.add(base_stamp + 10000000LL, 0.2); // +10ms
    imu.add(base_stamp + 20000000LL, 0.2); // +20ms
    imu.add(base_stamp + 30000000LL, 0.2); // +30ms

    // 2. Uninitialized timestamps (from <= 0 or to <= 0) always return false
    assert(!imu.delta(0, base_stamp + 20000000LL, delta_rad));
    assert(!imu.delta(-100, base_stamp + 20000000LL, delta_rad));
    assert(!imu.delta(base_stamp + 10000000LL, 0, delta_rad));

    // 3. Timestamps far outside sample range (e.g. epoch 1970) do not extrapolate and return false
    int64_t way_in_past = 1000000000LL;
    assert(!imu.delta(way_in_past, base_stamp + 20000000LL, delta_rad));

    // 4. Valid integration within the sampling interval returns true
    assert(imu.delta(base_stamp, base_stamp + 20000000LL, delta_rad));
    // 0.2 rad/s * 0.02s = 0.004 rad
    assert(std::abs(delta_rad - 0.004) < 1e-4);

    std::cout << "[Test] testYawHistoryDeltaBounds PASSED!" << std::endl;
}

void testGpuDeskewLargeImuBuffer()
{
    std::cout << "[Test] testGpuDeskewLargeImuBuffer starting..." << std::endl;

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "[Test] Skipping GPU test (no CUDA device)." << std::endl;
        return;
    }

    YawHistory imu;
    const int64_t base_stamp = 1793413750000000000LL;
    const int64_t dt_ns      = 5000000LL;  // 200Hz (5ms step)

    // Generate 2.0s of IMU history (401 samples, exceeding the old 256-sample limit).
    // First 1.28s (samples 0..256) is stationary (0.0 rad/s), remaining samples (257..400) rotate at 2.0 rad/s.
    for (int i = 0; i <= 400; ++i) {
        int64_t t = base_stamp + static_cast<int64_t>(i) * dt_ns;
        double omega = (i <= 256) ? 0.0 : 2.0;
        imu.add(t, omega);
    }

    const int64_t reference_ns = base_stamp + 400LL * dt_ns;
    const int64_t scan_mid_ns  = reference_ns - 50000000LL;  // 50ms earlier (2.0 rad/s * -0.05s = -0.10 rad)

    std::vector<IMUSample> imu_samples;
    imu.get_imu_samples(imu_samples);
    assert(imu_samples.size() > 256);

    double ref_angle = 0.0;
    assert(imu.get_angle_at(reference_ns, ref_angle));

    DeskewParams deskew_params{};
    deskew_params.enabled       = true;
    deskew_params.reference_ns  = reference_ns;
    deskew_params.min_stamp_ns  = 0;
    deskew_params.ref_yaw       = 0.0f;
    deskew_params.vx_world      = 0.0f;
    deskew_params.vy_world      = 0.0f;
    deskew_params.ref_imu_angle = static_cast<float>(ref_angle);

    // Two obstacle points:
    // Point 0: at reference_ns      -> rotational correction 0 rad     -> remains at (2.0, 0.0, 0.5)
    // Point 1: at reference_ns-50ms -> rotational correction -0.10 rad -> rotates to (2.0*cos(-0.1), 2.0*sin(-0.1), 0.5)
    std::vector<TimedScanPoint> timed_points(2);
    timed_points[0] = TimedScanPoint{2.0f, 0.0f, 0.5f, reference_ns};
    timed_points[1] = TimedScanPoint{2.0f, 0.0f, 0.5f, scan_mid_ns};

    const float h_transform[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    GroundFilterParams filter_params{};
    filter_params.range_max        = 12.0f;
    filter_params.robot_radius     = 0.5f;
    filter_params.robot_height_min = 0.0f;
    filter_params.robot_height_max = 1.0f;
    filter_params.ground_z_thresh  = 0.1f;

    PoseSolver solver(1000);
    int obs_count = solver.prepareObstacleCloud(
        timed_points, h_transform, deskew_params, imu_samples, 1, filter_params, nullptr
    );
    assert(obs_count == 2);

    std::vector<float> ground_out;
    std::vector<float> obstacle_out;
    solver.copyFilteredClouds(&ground_out, &obstacle_out);
    assert(obstacle_out.size() == 6);

    // Match output points by (x, y) coordinates since atomicAdd order is non-deterministic
    bool found_ref_pt = false;
    bool found_mid_pt = false;
    const float expected_mid_x = 2.0f * std::cos(-0.10f);
    const float expected_mid_y = 2.0f * std::sin(-0.10f);

    for (int i = 0; i < 2; ++i) {
        float x = obstacle_out[i * 3 + 0];
        float y = obstacle_out[i * 3 + 1];
        if (std::abs(x - 2.0f) < 1e-3f && std::abs(y - 0.0f) < 1e-3f) {
            found_ref_pt = true;
        }
        if (std::abs(x - expected_mid_x) < 2e-3f && std::abs(y - expected_mid_y) < 2e-3f) {
            found_mid_pt = true;
        }
    }
    assert(found_ref_pt);
    assert(found_mid_pt);

    std::cout << "[Test] testGpuDeskewLargeImuBuffer PASSED!" << std::endl;
}

int main()
{
    testNormalizeAngle();
    testYawHistoryDeltaBounds();
    testGpuDeskewLargeImuBuffer();
    std::cout << "All angle and IMU bounds tests PASSED successfully!" << std::endl;
    return 0;
}

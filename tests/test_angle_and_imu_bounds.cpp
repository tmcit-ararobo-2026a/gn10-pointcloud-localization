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

    // 基本的な範囲チェック [-pi, pi]
    assert(std::abs(normalizeAngle(0.0f)) < 1e-6f);
    assert(std::abs(normalizeAngle(static_cast<float>(M_PI))) <= static_cast<float>(M_PI));
    assert(std::abs(normalizeAngle(-static_cast<float>(M_PI))) <= static_cast<float>(M_PI));

    // 2*pi の周期性チェック
    float angle_base = 0.5f;
    for (int k = -10; k <= 10; ++k) {
        float test_val = angle_base + static_cast<float>(k * 2.0 * M_PI);
        float norm = normalizeAngle(test_val);
        assert(std::abs(norm - angle_base) < 1e-4f);
    }

    // 巨大な角度 (float32 仮数部精度限界超えによる while ループハング防止検証)
    float huge_angle = 448353440.0f; // 1.79e9 s * 0.25 rad/s
    float huge_norm = normalizeAngle(huge_angle);
    assert(std::isfinite(huge_norm));
    assert(huge_norm >= -static_cast<float>(M_PI) && huge_norm <= static_cast<float>(M_PI));

    float huge_neg_angle = -500000000.0f;
    float huge_neg_norm = normalizeAngle(huge_neg_angle);
    assert(std::isfinite(huge_neg_norm));
    assert(huge_neg_norm >= -static_cast<float>(M_PI) && huge_neg_norm <= static_cast<float>(M_PI));

    // 非有限値 (NaN, Inf) での無限ループ防止と安全なフォールバック
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

    // 1. 空の履歴では false
    assert(!imu.delta(1000000000, 1000050000, delta_rad));

    // サンプルを追加 (100Hz: 10ms刻み)
    int64_t base_stamp = 1793413750000000000LL;
    imu.add(base_stamp + 0, 0.2);
    imu.add(base_stamp + 10000000LL, 0.2); // +10ms
    imu.add(base_stamp + 20000000LL, 0.2); // +20ms
    imu.add(base_stamp + 30000000LL, 0.2); // +30ms

    // 2. 未初期化タイムスタンプ (from <= 0 または to <= 0) では必ず false
    assert(!imu.delta(0, base_stamp + 20000000LL, delta_rad));
    assert(!imu.delta(-100, base_stamp + 20000000LL, delta_rad));
    assert(!imu.delta(base_stamp + 10000000LL, 0, delta_rad));

    // 3. サンプル範囲から大きく離れた時刻 (57年前や遠い過去/未来) では外挿せず false
    int64_t way_in_past = 1000000000LL; // 1970年付近
    assert(!imu.delta(way_in_past, base_stamp + 20000000LL, delta_rad));

    // 4. 正常なサンプリング範囲内での積分計算は正確に true
    assert(imu.delta(base_stamp, base_stamp + 20000000LL, delta_rad));
    // 0.2 rad/s * 0.02s = 0.004 rad
    assert(std::abs(delta_rad - 0.004) < 1e-4);

    std::cout << "[Test] testYawHistoryDeltaBounds PASSED!" << std::endl;
}

int main()
{
    testNormalizeAngle();
    testYawHistoryDeltaBounds();
    std::cout << "All angle and IMU bounds tests PASSED successfully!" << std::endl;
    return 0;
}

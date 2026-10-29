// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace gn10_pointcloud_localization
{

struct FrameProfileRecord
{
    double timestamp_sec{0.0};
    size_t raw_points{0};
    size_t obstacle_points{0};
    size_t dynamic_points{0};
    double time_deskew_ms{0.0};
    double time_ground_filter_ms{0.0};
    double time_matching_ms{0.0};
    double time_dynamic_extraction_ms{0.0};
    double time_publish_ms{0.0};
    double time_total_ms{0.0};
    size_t d2h_transfer_bytes{0};
    bool is_lost{false};
};

class BenchmarkProfiler
{
public:
    BenchmarkProfiler() = default;

    ~BenchmarkProfiler()
    {
        close();
    }

    void init(bool enable, const std::string& csv_output_path)
    {
        enabled_ = enable;
        csv_path_ = csv_output_path;
        if (!enabled_ || csv_path_.empty()) return;

        std::lock_guard<std::mutex> lock(mutex_);
        file_.open(csv_path_, std::ios::out | std::ios::trunc);
        if (file_.is_open()) {
            file_ << "timestamp_sec,raw_points,obstacle_points,dynamic_points,"
                  << "time_deskew_ms,time_ground_filter_ms,time_matching_ms,"
                  << "time_dynamic_extraction_ms,time_publish_ms,time_total_ms,"
                  << "d2h_transfer_bytes,is_lost\n";
            file_.flush();
        } else {
            std::cerr << "[BenchmarkProfiler] Failed to open CSV file: " << csv_path_ << std::endl;
        }
    }

    bool isEnabled() const { return enabled_; }

    void recordFrame(const FrameProfileRecord& record)
    {
        if (!enabled_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        records_.push_back(record);

        if (file_.is_open()) {
            file_ << std::fixed << std::setprecision(6)
                  << record.timestamp_sec << ","
                  << record.raw_points << ","
                  << record.obstacle_points << ","
                  << record.dynamic_points << ","
                  << std::setprecision(4)
                  << record.time_deskew_ms << ","
                  << record.time_ground_filter_ms << ","
                  << record.time_matching_ms << ","
                  << record.time_dynamic_extraction_ms << ","
                  << record.time_publish_ms << ","
                  << record.time_total_ms << ","
                  << record.d2h_transfer_bytes << ","
                  << (record.is_lost ? 1 : 0) << "\n";
            file_.flush();
        }
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.close();
        }
    }

    const std::vector<FrameProfileRecord>& records() const { return records_; }

private:
    bool enabled_{false};
    std::string csv_path_;
    std::ofstream file_;
    std::mutex mutex_;
    std::vector<FrameProfileRecord> records_;
};

}  // namespace gn10_pointcloud_localization


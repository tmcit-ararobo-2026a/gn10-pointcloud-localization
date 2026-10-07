#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <vector>

struct TimedScanPoint {
    float x, y, z;
    int64_t stamp_ns;
};

// A time-indexed integral prevents a cloud callback from discarding IMU
// rotation that arrived after the cloud's reference time.
class YawHistory {
public:
    void add(int64_t stamp_ns, double rate)
    {
        if (!std::isfinite(rate)) return;
        if (!samples_.empty() && stamp_ns <= samples_.back().stamp) return;
        if (!samples_.empty() && stamp_ns - samples_.back().stamp > 500000000) {
            samples_.clear();
        }
        double angle = 0;
        if (!samples_.empty()) {
            const auto& last = samples_.back();
            angle = last.angle + 0.5*(last.rate + rate)*seconds(stamp_ns - last.stamp);
        }
        samples_.push_back({stamp_ns, rate, angle});
        while (samples_.size() > 2 && stamp_ns - samples_[1].stamp > 2000000000) {
            samples_.pop_front();
        }
    }

    bool delta(int64_t from, int64_t to, double& result) const
    {
        double a, b;
        if (!at(from, a) || !at(to, b)) return false;
        result = b - a;
        return true;
    }
    void clear() { samples_.clear(); }

private:
    struct Sample { int64_t stamp; double rate, angle; };
    std::deque<Sample> samples_;
    static double seconds(int64_t dt) { return double(dt)*1e-9; }
    bool at(int64_t t, double& angle) const
    {
        if (samples_.empty()) return false;
        const auto& first = samples_.front();
        const auto& last = samples_.back();
        // Small packet reordering is allowed; stale IMU data is never used
        // to rotate a long accumulation window.
        if (t < first.stamp) {
            if (first.stamp-t > 20000000) return false;
            angle = first.angle + first.rate*seconds(t-first.stamp);
            return true;
        }
        if (t >= last.stamp) {
            if (t-last.stamp > 20000000) return false;
            angle = last.angle + last.rate*seconds(t-last.stamp);
            return true;
        }
        auto hi = std::upper_bound(samples_.begin(), samples_.end(), t,
                                   [](int64_t t, const Sample& s) { return t < s.stamp; });
        const auto& lo = *std::prev(hi);
        double dt = seconds(t-lo.stamp), interval = seconds(hi->stamp-lo.stamp);
        angle = lo.angle + lo.rate*dt + 0.5*(hi->rate-lo.rate)*dt*dt/interval;
        return true;
    }
};

class ScanAccumulator {
public:
    explicit ScanAccumulator(double window_s = 0.10) : window_ns_(int64_t(window_s*1e9)) {}

    void clear() { points_.clear(); first_stamp_ = 0; last_reference_ = 0; newest_start_ = 0; }
    bool backwards(int64_t reference) const
    {
        return last_reference_ && reference <= last_reference_;
    }
    bool append(const std::vector<TimedScanPoint>& points, int64_t reference)
    {
        if (backwards(reference)) clear();
        if (last_reference_ && reference-last_reference_ > std::max(2*window_ns_,int64_t(200000000))) {
            clear();
        }
        if (points.empty()) return false;
        if (!first_stamp_) first_stamp_ = points.front().stamp_ns;
        newest_start_ = points.front().stamp_ns;
        if (window_ns_ == 0) {
            points_ = points;
            last_reference_ = reference;
            return true;
        }
        for (const auto& p : points) points_.push_back(p);
        last_reference_ = reference;
        // Frames and their point times are ordered by the LiDAR driver.
        // Erase by time as well as by count, so storage stays bounded.
        const int64_t begin = reference-window_ns_;
        points_.erase(std::remove_if(points_.begin(), points_.end(),
                      [begin](const auto& p) { return p.stamp_ns < begin; }), points_.end());
        return window_ns_ == 0 || reference-first_stamp_ >= window_ns_-2000000;
    }

    std::vector<float> cloud(int64_t reference, const YawHistory& imu,
                             double yaw, double vx_world, double vy_world,
                             size_t max_points, bool* fully_deskewed = nullptr) const
    {
        std::vector<float> result;
        if (points_.empty() || max_points == 0) return result;
        double unused;
        const bool full_imu = imu.delta(reference,points_.front().stamp_ns,unused);
        if (fully_deskewed) *fully_deskewed = full_imu;
        result.reserve(std::min(points_.size(),max_points)*3);
        const double c = std::cos(yaw), s = std::sin(yaw);
        const double vx = c*vx_world+s*vy_world, vy = -s*vx_world+c*vy_world;
        // Uniform decimation retains all time phases when capacity is limited.
        const size_t stride = std::max(size_t(1),(points_.size()+max_points-1)/max_points);
        for (size_t i = 0; i < points_.size(); i += stride) {
            const auto& p = points_[i];
            if (p.stamp_ns > reference) continue;
            if (!full_imu && p.stamp_ns < newest_start_) continue;
            double angle = 0;
            // Without coverage, retain the newest scan at its native density
            // rather than accumulating old scans with an invented rotation.
            imu.delta(reference,p.stamp_ns,angle);
            const double dt = double(p.stamp_ns-reference)*1e-9;
            const double ca = std::cos(angle), sa = std::sin(angle);
            result.push_back(float(ca*p.x-sa*p.y+vx*dt));
            result.push_back(float(sa*p.x+ca*p.y+vy*dt));
            result.push_back(p.z);
        }
        return result;
    }

private:
    int64_t window_ns_, first_stamp_{0}, last_reference_{0}, newest_start_{0};
    std::vector<TimedScanPoint> points_;
};

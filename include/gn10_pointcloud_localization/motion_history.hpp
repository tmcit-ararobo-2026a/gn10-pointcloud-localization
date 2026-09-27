#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <optional>

namespace gn10 {
struct TimedPose {
    double stamp, x, y, yaw;
};

inline double angleDifference(double a, double b)
{
    return std::atan2(std::sin(a - b), std::cos(a - b));
}

// Standard planar odometry only. No dependency on a LiDAR odometry implementation.
class MotionHistory
{
public:
    explicit MotionHistory(double max_gap_s=0.25) : max_gap_s_(max_gap_s) {}
    bool add(TimedPose p)
    {
        if (!std::isfinite(p.stamp) || !std::isfinite(p.x) ||
            !std::isfinite(p.y) || !std::isfinite(p.yaw)) return false;
        if (!samples.empty() && p.stamp <= samples.back().stamp) return false;
        samples.push_back(p);
        while (samples.size() > 2 && p.stamp - samples.front().stamp > 5.0)
            samples.pop_front();
        return true;
    }

    std::optional<TimedPose> at(double t) const
    {
        // FAST-LIO may remove the last few returns during point filtering.
        // Hold an endpoint only within 100 us; larger gaps remain unavailable.
        constexpr double endpoint_slack_s = 1e-4;
        if (samples.empty() || t < samples.front().stamp - endpoint_slack_s ||
            t > samples.back().stamp + endpoint_slack_s) return {};
        if (t <= samples.front().stamp) return samples.front();
        if (t >= samples.back().stamp) return samples.back();
        const auto hi = std::lower_bound(samples.begin(), samples.end(), t,
            [](const auto& p, double v) { return p.stamp < v; });
        if (hi == samples.end()) return samples.back();
        if (std::abs(hi->stamp - t) < 1e-6) return *hi;
        if (hi == samples.begin()) return {};
        const auto lo = std::prev(hi);
        if (hi->stamp - lo->stamp > max_gap_s_) return {};
        const double a = (t - lo->stamp) / (hi->stamp - lo->stamp);
        return TimedPose{t, lo->x + a * (hi->x - lo->x), lo->y + a * (hi->y - lo->y),
            lo->yaw + a * angleDifference(hi->yaw, lo->yaw)};
    }

    // Coordinates observed at 'from', expressed in the robot at 'to'.
    static void transformPoint(const TimedPose& from, const TimedPose& to, float& x, float& y)
    {
        const double wx = std::cos(from.yaw) * x - std::sin(from.yaw) * y + from.x - to.x;
        const double wy = std::sin(from.yaw) * x + std::cos(from.yaw) * y + from.y - to.y;
        x = std::cos(to.yaw) * wx + std::sin(to.yaw) * wy;
        y = -std::sin(to.yaw) * wx + std::cos(to.yaw) * wy;
    }

    static TimedPose advance(const TimedPose& map, const TimedPose& from, const TimedPose& to)
    {
        const double dx = to.x - from.x, dy = to.y - from.y;
        const double c = std::cos(map.yaw - from.yaw), s = std::sin(map.yaw - from.yaw);
        return {to.stamp, map.x + c * dx - s * dy, map.y + s * dx + c * dy,
            map.yaw + angleDifference(to.yaw, from.yaw)};
    }

    std::deque<TimedPose> samples;
private:
    double max_gap_s_;
};
}  // namespace gn10

#pragma once

#include "gn10_pointcloud_localization/scan_points.hpp"
#include <livox_ros_driver2/msg/custom_msg.hpp>

namespace gn10 {
// Use the same header time as FAST-LIO, preserving offset_time in nanoseconds.
inline ScanPoints decodeScan(const livox_ros_driver2::msg::CustomMsg& msg)
{
    ScanPoints out;
    if (msg.point_num != msg.points.size()) return out;
    out.start = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;
    out.end = out.start;
    const double header_ns = static_cast<double>(msg.header.stamp.sec) * 1e9 +
        msg.header.stamp.nanosec;
    if (msg.timebase != 0 && std::abs(static_cast<double>(msg.timebase) - header_ns) > 1e6)
        return out; // inconsistent clocks must not silently enter deskew
    out.timed = true;
    out.xyz.reserve(msg.points.size() * 3);
    out.times.reserve(msg.points.size());
    for (const auto& p : msg.points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        const double dt = p.offset_time * 1e-9;
        if (dt > 0.2) return ScanPoints{};
        const double stamp = out.start + dt;
        out.end = std::max(out.end, stamp);
        out.xyz.insert(out.xyz.end(), {p.x, p.y, p.z});
        out.times.push_back(stamp);
    }
    out.valid = !out.xyz.empty();
    return out;
}
}  // namespace gn10

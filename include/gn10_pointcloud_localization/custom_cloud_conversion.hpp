#pragma once

#include "gn10_pointcloud_localization/custom_scan.hpp"
#include <stdexcept>

namespace gn10 {
inline sensor_msgs::msg::PointCloud2 customToPointCloud2(
    const livox_ros_driver2::msg::CustomMsg& input)
{
    const double header_ns = static_cast<double>(input.header.stamp.sec) * 1e9 +
        input.header.stamp.nanosec;
    if (input.point_num != input.points.size() || (input.timebase != 0 &&
        std::abs(static_cast<double>(input.timebase) - header_ns) > 1e6))
        throw std::invalid_argument("Invalid CustomMsg scan");
    sensor_msgs::msg::PointCloud2 output;
    output.header = input.header;
    output.height = 1;
    output.is_bigendian = false;
    output.is_dense = true;
    output.point_step = 24;
    auto field = [&](const std::string& name, uint32_t offset, uint8_t type) {
        sensor_msgs::msg::PointField f;
        f.name = name; f.offset = offset; f.datatype = type; f.count = 1;
        output.fields.push_back(f);
    };
    field("x", 0, sensor_msgs::msg::PointField::FLOAT32);
    field("y", 4, sensor_msgs::msg::PointField::FLOAT32);
    field("z", 8, sensor_msgs::msg::PointField::FLOAT32);
    field("intensity", 12, sensor_msgs::msg::PointField::FLOAT32);
    field("tag", 16, sensor_msgs::msg::PointField::UINT8);
    field("line", 17, sensor_msgs::msg::PointField::UINT8);
    field("offset_time", 20, sensor_msgs::msg::PointField::UINT32);
    output.data.reserve(input.points.size() * output.point_step);
    for (const auto& p : input.points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        if (p.offset_time > 200000000) throw std::invalid_argument("Invalid point time");
        const size_t offset = output.data.size();
        output.data.resize(offset + output.point_step, 0);
        auto* dst = output.data.data() + offset;
        const float intensity = p.reflectivity;
        std::memcpy(dst, &p.x, 4); std::memcpy(dst + 4, &p.y, 4);
        std::memcpy(dst + 8, &p.z, 4); std::memcpy(dst + 12, &intensity, 4);
        dst[16] = p.tag; dst[17] = p.line;
        std::memcpy(dst + 20, &p.offset_time, 4);
        ++output.width;
    }
    if (output.width == 0) throw std::invalid_argument("Empty CustomMsg scan");
    output.row_step = output.width * output.point_step;
    return output;
}
}  // namespace gn10

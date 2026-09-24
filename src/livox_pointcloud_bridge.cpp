#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace {

const sensor_msgs::msg::PointField* field(
    const sensor_msgs::msg::PointCloud2& cloud, const std::string& name, uint8_t datatype
)
{
    for (const auto& candidate : cloud.fields) {
        if (candidate.name == name && candidate.datatype == datatype &&
            candidate.offset + (datatype == sensor_msgs::msg::PointField::FLOAT64 ? 8U :
                                datatype == sensor_msgs::msg::PointField::FLOAT32 ? 4U : 1U) <=
                cloud.point_step) {
            return &candidate;
        }
    }
    return nullptr;
}

template <typename T>
T read(const uint8_t* point, uint32_t offset)
{
    T value;
    std::memcpy(&value, point + offset, sizeof(T));
    return value;
}

class LivoxPointCloudBridge : public rclcpp::Node
{
public:
    LivoxPointCloudBridge() : Node("livox_pointcloud_bridge")
    {
        const auto input = declare_parameter<std::string>("input_topic", "/livox/lidar");
        const auto output =
            declare_parameter<std::string>("output_topic", "/livox/lidar_custom");
        publisher_ = create_publisher<livox_ros_driver2::msg::CustomMsg>(
            output, rclcpp::QoS(10)
        );
        subscriber_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input, rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud) { convert(*cloud); }
        );
    }

private:
    void convert(const sensor_msgs::msg::PointCloud2& cloud)
    {
        const auto* x = field(cloud, "x", sensor_msgs::msg::PointField::FLOAT32);
        const auto* y = field(cloud, "y", sensor_msgs::msg::PointField::FLOAT32);
        const auto* z = field(cloud, "z", sensor_msgs::msg::PointField::FLOAT32);
        const auto* intensity = field(cloud, "intensity", sensor_msgs::msg::PointField::FLOAT32);
        const auto* tag = field(cloud, "tag", sensor_msgs::msg::PointField::UINT8);
        const auto* line = field(cloud, "line", sensor_msgs::msg::PointField::UINT8);
        const auto* timestamp = field(cloud, "timestamp", sensor_msgs::msg::PointField::FLOAT64);
        if (!x || !y || !z || !intensity || !tag || !line || !timestamp ||
            cloud.is_bigendian || cloud.point_step == 0 ||
            cloud.row_step < cloud.width * cloud.point_step ||
            cloud.data.size() < static_cast<size_t>(cloud.row_step) * cloud.height) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "Expected little-endian MID360 PointCloud2 fields x,y,z,intensity,tag,line,timestamp"
            );
            return;
        }

        const uint64_t base_ns = static_cast<uint64_t>(cloud.header.stamp.sec) * 1000000000ULL +
                                 cloud.header.stamp.nanosec;
        auto output = livox_ros_driver2::msg::CustomMsg();
        output.header = cloud.header;
        output.timebase = base_ns;
        output.lidar_id = 0;
        output.rsvd = {0, 0, 0};
        output.points.reserve(static_cast<size_t>(cloud.width) * cloud.height);

        for (uint32_t row = 0; row < cloud.height; ++row) {
            for (uint32_t col = 0; col < cloud.width; ++col) {
                const auto* point = cloud.data.data() +
                    static_cast<size_t>(row) * cloud.row_step +
                    static_cast<size_t>(col) * cloud.point_step;
                const double absolute_ns = read<double>(point, timestamp->offset);
                const float px = read<float>(point, x->offset);
                const float py = read<float>(point, y->offset);
                const float pz = read<float>(point, z->offset);
                const double offset_ns = absolute_ns - static_cast<double>(base_ns);
                if (!std::isfinite(absolute_ns) || !std::isfinite(px) ||
                    !std::isfinite(py) || !std::isfinite(pz) || offset_ns < -1000.0 ||
                    offset_ns > std::numeric_limits<uint32_t>::max()) {
                    continue;
                }
                livox_ros_driver2::msg::CustomPoint converted;
                converted.offset_time = static_cast<uint32_t>(std::max(0.0, std::round(offset_ns)));
                converted.x = px;
                converted.y = py;
                converted.z = pz;
                const float reflectivity = read<float>(point, intensity->offset);
                converted.reflectivity = std::isfinite(reflectivity) ? static_cast<uint8_t>(
                    std::clamp(std::round(reflectivity), 0.0f, 255.0f)) : 0;
                converted.tag = read<uint8_t>(point, tag->offset);
                converted.line = read<uint8_t>(point, line->offset);
                output.points.push_back(converted);
            }
        }
        output.point_num = static_cast<uint32_t>(output.points.size());
        if (output.point_num > 0) publisher_->publish(std::move(output));
    }

    rclcpp::Publisher<livox_ros_driver2::msg::CustomMsg>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscriber_;
};
}  // namespace

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LivoxPointCloudBridge>());
    rclcpp::shutdown();
    return 0;
}

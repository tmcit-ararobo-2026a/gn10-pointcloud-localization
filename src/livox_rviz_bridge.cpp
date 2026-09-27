#include "gn10_pointcloud_localization/custom_cloud_conversion.hpp"
#include <rclcpp/rclcpp.hpp>

class LivoxRvizBridge : public rclcpp::Node
{
public:
    LivoxRvizBridge() : Node("livox_rviz_bridge")
    {
        const auto input = declare_parameter<std::string>("input_topic", "/livox/lidar");
        const auto output = declare_parameter<std::string>("output_topic", "/gn10/rviz_cloud");
        publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(output, rclcpp::SensorDataQoS());
        subscriber_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
            input, rclcpp::SensorDataQoS(),
            [this](livox_ros_driver2::msg::CustomMsg::ConstSharedPtr msg) {
                if (publisher_->get_subscription_count() == 0) return;
                try {
                    publisher_->publish(gn10::customToPointCloud2(*msg));
                } catch (const std::invalid_argument& e) {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s", e.what());
                }
            });
    }
private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr subscriber_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LivoxRvizBridge>());
    rclcpp::shutdown();
    return 0;
}

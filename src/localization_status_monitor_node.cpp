// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <chrono>
#include <cmath>
#include <set>
#include <stdexcept>

#include "gn10_pointcloud_localization/msg/localization_status.hpp"

namespace gn10_pointcloud_localization {
/** @brief CPU monitor that distinguishes sensor receipt, processing progress and lost state. */
class LocalizationStatusMonitor : public rclcpp::Node {
public:
  using Status = msg::LocalizationStatus;
  using Clock = std::chrono::steady_clock;
  /** @brief Subscribe to sensors and completed GPU results, without creating a CUDA context. */
  LocalizationStatusMonitor() : Node("localization_status_monitor") {
    input_timeout_s_ = declare_parameter<double>("status_monitor.input_timeout_s", 0.5);
    processing_timeout_s_ = declare_parameter<double>("status_monitor.processing_timeout_s", 0.5);
    for (double limit : {input_timeout_s_, processing_timeout_s_}) {
      if (!std::isfinite(limit) || limit <= 0) {
        throw std::invalid_argument("Status deadlines must be finite and positive");
      }
    }
    frame_ = declare_parameter<std::string>("frames.map_frame", "map");
    const auto cloud_topic = declare_parameter<std::string>("topics.input_cloud", "/livox/lidar");
    const auto imu_topic = declare_parameter<std::string>("topics.input_imu", "/livox/imu");
    publisher_ = create_publisher<Status>("/localization/status", rclcpp::QoS(1).durability_volatile());
    cloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(cloud_topic,
      rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud) {
        observe(cloud->header.stamp, cloud_stamp_ns_, cloud_deadline_, have_cloud_);
      });
    imu_subscription_ = create_subscription<sensor_msgs::msg::Imu>(imu_topic,
      rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::Imu::ConstSharedPtr imu) {
        for (const auto value : {imu->angular_velocity.x, imu->angular_velocity.y,
            imu->angular_velocity.z, imu->linear_acceleration.x, imu->linear_acceleration.y,
            imu->linear_acceleration.z})
        {
          if (!std::isfinite(value)) {
            have_imu_ = false;
            return;
          }
        }
        observe(imu->header.stamp, imu_stamp_ns_, imu_deadline_, have_imu_);
      });
    raw_subscription_ = create_subscription<Status>("/localization/status_raw",
      rclcpp::QoS(1).durability_volatile(),
      [this](Status::ConstSharedPtr status) { accept_result(*status); });
    timer_ = create_wall_timer(std::chrono::milliseconds(20), [this]() { publish_status(); });
  }
private:
  /** @brief Parse valid nonnegative ROS timestamp fields before integer conversion. */
  int64_t stamp_ns(const builtin_interfaces::msg::Time & stamp) const {
    if (stamp.sec < 0 || stamp.nanosec >= 1000000000U) {
      return 0;
    }
    return rclcpp::Time(stamp).nanoseconds();
  }
  /** @brief Check a timestamp against production age and build a remaining monotonic lease. */
  bool timing(const builtin_interfaces::msg::Time & stamp, double limit, Clock::time_point & deadline) const {
    const auto measured_ns = stamp_ns(stamp);
    const double age_s = static_cast<double>(now().nanoseconds() - measured_ns) * 1e-9;
    if (measured_ns <= 0 || age_s < 0 || age_s >= limit) {
      return false;
    }
    deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double>(limit - age_s));
    return true;
  }
  /** @brief Observe strictly advancing sensor timestamps; no duplicated input can renew freshness. */
  void observe(const builtin_interfaces::msg::Time & stamp, int64_t & previous,
    Clock::time_point & deadline, bool & have_input)
  {
    const auto measured_ns = stamp_ns(stamp);
    Clock::time_point proposed;
    if (measured_ns <= previous || !timing(stamp, input_timeout_s_, proposed)) {
      return;
    }
    previous = measured_ns;
    deadline = proposed;
    have_input = true;
  }
  /** @brief Admit real processor progress only, retaining its original measurement/pose stamps. */
  void accept_result(const Status & status) {
    Clock::time_point proposed;
    if (status.header.frame_id != frame_ || status.processor_session == 0 || status.result_sequence == 0 ||
      !timing(status.measurement_stamp, processing_timeout_s_, proposed) ||
      retired_sessions_.count(status.processor_session) != 0)
    {
      return;
    }
    if (status.processor_session == result_.processor_session) {
      if (status.result_sequence <= result_.result_sequence ||
        stamp_ns(status.measurement_stamp) <= stamp_ns(result_.measurement_stamp))
      {
        return;
      }
    } else if (result_.processor_session != 0) {
      retired_sessions_.insert(result_.processor_session);
    }
    if ((status.match_metrics_valid && !std::isfinite(status.best_cost)) ||
      (status.inlier_metrics_valid && (!std::isfinite(status.inlier_cost) || status.inlier_count < 0)))
    {
      return;
    }
    result_ = status;
    processing_deadline_ = proposed;
    have_result_ = true;
  }
  /** @brief Report current freshness independently of the GPU node or input callbacks. */
  void publish_status() {
    const auto current_ns = now().nanoseconds();
    if (last_ros_now_ns_ > 0 && current_ns < last_ros_now_ns_) {
      have_cloud_ = have_imu_ = have_result_ = false;
      cloud_stamp_ns_ = imu_stamp_ns_ = 0;
    }
    last_ros_now_ns_ = current_ns;
    const auto steady_now = Clock::now();
    Status status = result_;
    status.header.stamp = now();
    status.header.frame_id = frame_;
    status.report_sequence = ++report_sequence_;
    const double cloud_age_s = static_cast<double>(current_ns - cloud_stamp_ns_) * 1e-9;
    const double imu_age_s = static_cast<double>(current_ns - imu_stamp_ns_) * 1e-9;
    const double processing_age_s = static_cast<double>(current_ns - stamp_ns(result_.measurement_stamp)) * 1e-9;
    status.cloud_fresh = have_cloud_ && steady_now < cloud_deadline_ && cloud_age_s >= 0 && cloud_age_s < input_timeout_s_;
    status.imu_fresh = have_imu_ && steady_now < imu_deadline_ && imu_age_s >= 0 && imu_age_s < input_timeout_s_;
    status.processing_fresh = have_result_ && steady_now < processing_deadline_ &&
      processing_age_s >= 0 && processing_age_s < processing_timeout_s_;
    status.covariance_valid = false;
    status.ready_quality_valid = false;
    if (!status.processing_fresh) {
      status.last_match_success = false;
      status.lost = true;
      status.match_metrics_valid = false;
      status.inlier_metrics_valid = false;
      status.reason = "Processor unavailable or result expired";
    } else if (!status.cloud_fresh || !status.imu_fresh) {
      status.reason = "Sensor input expired; Ready quality unresolved";
    } else if (!status.cloud_tf_available || !status.imu_tf_available) {
      status.reason = "Sensor TF unavailable; Ready quality unresolved";
    } else {
      status.reason += "; Ready quality unresolved, covariance is fixed";
    }
    publisher_->publish(status);
  }
  std::string frame_;
  double input_timeout_s_{}, processing_timeout_s_{};
  bool have_cloud_{}, have_imu_{}, have_result_{};
  int64_t cloud_stamp_ns_{}, imu_stamp_ns_{}, last_ros_now_ns_{};
  uint64_t report_sequence_{};
  Clock::time_point cloud_deadline_{}, imu_deadline_{}, processing_deadline_{};
  Status result_{};
  std::set<uint64_t> retired_sessions_;
  rclcpp::Publisher<Status>::SharedPtr publisher_;
  rclcpp::Subscription<Status>::SharedPtr raw_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
};
}  // namespace gn10_pointcloud_localization

/** @brief Run the independent CPU monitor without map allocation or CUDA initialization. */
int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<gn10_pointcloud_localization::LocalizationStatusMonitor>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("localization_status_monitor"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}

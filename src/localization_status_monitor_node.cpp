// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <chrono>
#include <cmath>
#include <set>
#include <random>
#include <memory>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include "ready_quality.hpp"
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
    quality_ = std::make_unique<ReadyQuality>(declare_parameter<std::string>("status_monitor.quality_profile", ""));
    std::random_device random;
    monitor_session_ = (static_cast<uint64_t>(random()) << 32) | random();
    if (monitor_session_ == 0) { monitor_session_ = 1; }
    frame_ = declare_parameter<std::string>("frames.map_frame", "map");
    base_frame_ = declare_parameter<std::string>("frames.base_frame", "base_link");
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    const auto cloud_topic = declare_parameter<std::string>("topics.input_cloud", "/livox/lidar");
    const auto imu_topic = declare_parameter<std::string>("topics.input_imu", "/livox/imu");
    publisher_ = create_publisher<Status>("/localization/status", rclcpp::QoS(1).durability_volatile());
    cloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(cloud_topic,
      rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud) {
        cloud_frame_ = cloud->header.frame_id;
        observe(cloud->header.stamp, cloud_stamp_ns_, cloud_deadline_, have_cloud_, quality_->limit("max_cloud_age", input_timeout_s_));
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
        imu_frame_ = imu->header.frame_id;
        observe(imu->header.stamp, imu_stamp_ns_, imu_deadline_, have_imu_, quality_->limit("max_imu_age", input_timeout_s_));
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
    Clock::time_point & deadline, bool & have_input, double limit_s)
  {
    const auto measured_ns = stamp_ns(stamp);
    Clock::time_point proposed;
    if (measured_ns <= previous || !timing(stamp, limit_s, proposed)) {
      return;
    }
    previous = measured_ns;
    deadline = proposed;
    have_input = true;
  }
  /** @brief Admit real processor progress only, retaining its original measurement/pose stamps. */
  void accept_result(const Status & status) {
    Clock::time_point proposed;
    if (status.processor_session == 0 || status.result_sequence == 0 ||
      retired_sessions_.count(status.processor_session) != 0) { return; }
    if (status.processor_session == result_.processor_session &&
      status.result_sequence <= result_.result_sequence) { return; }
    if (status.header.frame_id != frame_ || !timing(status.measurement_stamp,
      std::min({quality_->limit("max_result_age", processing_timeout_s_),
                quality_->limit("max_pose_age", processing_timeout_s_),
                quality_->limit("max_tf_age", processing_timeout_s_)}), proposed))
    {
      // A new failed outcome must revoke quality, rather than retain the last good one.
      if (result_.processor_session != 0 && status.processor_session != result_.processor_session) {
        retired_sessions_.insert(result_.processor_session);
      }
      result_ = status;
      result_.measurement_stamp = builtin_interfaces::msg::Time{};
      have_result_ = false;
      quality_->reset();
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
    result_ = status;
    if ((status.match_metrics_valid && !std::isfinite(status.best_cost)) ||
      (status.inlier_metrics_valid && (!std::isfinite(status.inlier_cost) || status.inlier_count < 0)))
    {
      result_.last_match_success = false;
      result_.lost = true;
      result_.match_metrics_valid = false;
      result_.inlier_metrics_valid = false;
    }
    processing_deadline_ = proposed;
    have_result_ = true;
  }
  /** @brief Report current freshness independently of the GPU node or input callbacks. */
  void publish_status() {
    const auto current_ns = now().nanoseconds();
    if (last_ros_now_ns_ > 0 && current_ns < last_ros_now_ns_) {
      have_cloud_ = have_imu_ = have_result_ = false;
      cloud_stamp_ns_ = imu_stamp_ns_ = 0;
      quality_->reset();
      result_.measurement_stamp = builtin_interfaces::msg::Time{};
      tf_buffer_->clear();
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
    status.monitor_session = monitor_session_;
    status.cloud_stamp = rclcpp::Time(cloud_stamp_ns_);
    status.imu_stamp = rclcpp::Time(imu_stamp_ns_);
    status.cloud_fresh = status.cloud_fresh && cloud_age_s < quality_->limit("max_cloud_age", input_timeout_s_);
    status.imu_fresh = status.imu_fresh && imu_age_s < quality_->limit("max_imu_age", input_timeout_s_);
    status.processing_fresh = status.processing_fresh && processing_age_s < quality_->limit("max_result_age", processing_timeout_s_);
    status.pose_valid = validate_pose(status);
    bool pose_tf = false;
    if (status.pose_valid) {
      pose_tf = matching_pose_tf(status);
      status.cloud_tf_available = status.cloud_tf_available && sensor_tf(cloud_frame_, status.measurement_stamp);
      status.imu_tf_available = status.imu_tf_available && sensor_tf(imu_frame_, status.imu_stamp);
    }
    status.covariance_valid = false;
    status.ready_quality_valid = quality_->evaluate(status, pose_tf,
      std::chrono::duration<double>(steady_now.time_since_epoch()).count());
    const double validity_ns = std::min({
      static_cast<double>(cloud_stamp_ns_) + quality_->limit("max_cloud_age", input_timeout_s_) * 1e9,
      static_cast<double>(imu_stamp_ns_) + quality_->limit("max_imu_age", input_timeout_s_) * 1e9,
      static_cast<double>(stamp_ns(status.measurement_stamp)) + quality_->limit("max_result_age", processing_timeout_s_) * 1e9,
      static_cast<double>(stamp_ns(status.pose_stamp)) + std::min(quality_->limit("max_pose_age"), quality_->limit("max_tf_age")) * 1e9});
    if (validity_ns > 0) { status.quality_valid_until = rclcpp::Time(static_cast<int64_t>(validity_ns)); }
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
      if (status.ready_quality_valid) {
        status.reason = "Synthetic quality approved; fixed covariance excluded";
      } else {
        status.reason = "Quality profile, matching Pose/TF or stable interval unavailable";
      }
    }
    publisher_->publish(status);
  }
  /** @brief Require a finite level Pose at exactly the successful measurement timestamp. */
  bool validate_pose(Status & status) const {
    if (!status.pose_valid || status.pose.header.frame_id != frame_ ||
      stamp_ns(status.pose.header.stamp) != stamp_ns(status.measurement_stamp) ||
      stamp_ns(status.pose_stamp) != stamp_ns(status.measurement_stamp)) { return false; }
    Clock::time_point ignored;
    if (!timing(status.pose_stamp, quality_->limit("max_pose_age"), ignored)) { return false; }
    auto & orientation = status.pose.pose.orientation;
    const auto & position = status.pose.pose.position;
    for (const double value : {position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w}) {
      if (!std::isfinite(value)) { return false; }
    }
    const double norm = std::sqrt(orientation.x * orientation.x + orientation.y * orientation.y +
      orientation.z * orientation.z + orientation.w * orientation.w);
    if (norm < 1e-9) { return false; }
    orientation.x /= norm; orientation.y /= norm; orientation.z /= norm; orientation.w /= norm;
    return std::abs(orientation.x) < 1e-4 && std::abs(orientation.y) < 1e-4;
  }
  /** @brief Compare Pose with the TF corresponding to the very same measurement. */
  bool matching_pose_tf(const Status & status) const {
    try {
      const auto transform = tf_buffer_->lookupTransform(frame_, base_frame_, rclcpp::Time(status.pose_stamp));
      Clock::time_point ignored;
      if (stamp_ns(transform.header.stamp) != stamp_ns(status.pose_stamp) ||
        !timing(transform.header.stamp, quality_->limit("max_tf_age"), ignored)) { return false; }
      const auto & measured = transform.transform;
      const auto & pose = status.pose.pose;
      const double norm = std::sqrt(measured.rotation.x * measured.rotation.x + measured.rotation.y * measured.rotation.y +
        measured.rotation.z * measured.rotation.z + measured.rotation.w * measured.rotation.w);
      if (!std::isfinite(norm) || norm < 1e-9 || std::abs(measured.rotation.x / norm) > 1e-4 ||
          std::abs(measured.rotation.y / norm) > 1e-4) { return false; }
      return std::abs(measured.translation.x - pose.position.x) <= 1e-6 &&
        std::abs(measured.translation.y - pose.position.y) <= 1e-6 &&
        std::abs(measured.translation.z - pose.position.z) <= 1e-6 &&
        std::abs(measured.rotation.z * pose.orientation.w - measured.rotation.w * pose.orientation.z) / norm <= 1e-6;
    } catch (const tf2::TransformException &) { return false; }
  }
  /** @brief Resolve each sensor extrinsic at its actual measurement timestamp. */
  bool sensor_tf(const std::string & sensor_frame, const builtin_interfaces::msg::Time & stamp) const {
    if (sensor_frame.empty()) { return false; }
    try {
      tf_buffer_->lookupTransform(base_frame_, sensor_frame, rclcpp::Time(stamp));
      return true;
    } catch (const tf2::TransformException &) { return false; }
  }
  std::unique_ptr<ReadyQuality> quality_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string frame_, base_frame_, cloud_frame_, imu_frame_;
  uint64_t monitor_session_{};
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

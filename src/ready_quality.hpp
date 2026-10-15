// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <array>
#include "gn10_pointcloud_localization/msg/localization_status.hpp"

namespace gn10_pointcloud_localization {
/** @brief Approved synthetic profile; real quality remains gated on E2 evidence. */
class ReadyQuality {
public:
  /** @brief Load explicit limits without inventing an approved hardware profile. */
  explicit ReadyQuality(const std::string & path) {
    if (path.empty()) { return; }
    std::ifstream stream(path);
    if (!stream) { throw std::invalid_argument("Quality profile missing"); }
    stream >> profile_;
    if (profile_.at("schema").get<int>() != 1) { throw std::invalid_argument("Quality schema unsupported"); }
    if (!profile_.at("approved").get<bool>()) { return; }
    const char * localhost = std::getenv("ROS_LOCALHOST_ONLY");
    const char * domain = std::getenv("ROS_DOMAIN_ID");
    if (profile_.at("mode") != "simulation" || localhost == nullptr ||
      std::string(localhost) != "1" || domain == nullptr || std::stoi(domain) < 100 || std::stoi(domain) > 232)
    {
      throw std::invalid_argument("Only dedicated localhost simulation quality is available; E2 not approved");
    }
    for (const auto key : {"quality_id", "evidence_id", "map_id", "team", "extrinsic_id",
        "implementation_id", "configuration_id"})
    {
      if (profile_.at(key).get<std::string>().empty()) { throw std::invalid_argument("Quality identity missing"); }
    }
    if (profile_.at("evidence_id").get<std::string>().rfind("synthetic-", 0) != 0) {
      throw std::invalid_argument("Synthetic evidence required");
    }
    for (const auto key : {"max_best_cost", "max_inlier_cost", "max_cloud_age", "max_imu_age",
        "max_result_age", "max_pose_age", "max_tf_age", "max_position_error", "max_yaw_error",
        "max_pose_jump", "max_yaw_jump"})
    {
      const double value = profile_.at(key).get<double>();
      const bool positive_required = std::string(key) != "max_best_cost" && std::string(key) != "max_inlier_cost";
      if (!std::isfinite(value) || value < 0 || (positive_required && value == 0)) {
        throw std::invalid_argument("Invalid quality threshold");
      }
    }
    if (!profile_.at("min_inliers").is_number_integer() || profile_.at("min_inliers").get<int>() <= 0) {
      throw std::invalid_argument("Invalid inlier threshold");
    }
    approved_ = true;
  }
  /** @brief Preserve a one-second continuous interval and count distinct actual outcomes. */
  bool evaluate(msg::LocalizationStatus & status, bool pose_tf_valid, double steady_s) {
    status.quality_id = profile_.value("quality_id", std::string{});
    status.evidence_id = profile_.value("evidence_id", std::string{});
    status.simulation_profile = approved_;
    status.pose_tf_valid = pose_tf_valid;
    bool valid = approved_ && status.processing_fresh && status.cloud_fresh && status.imu_fresh &&
      status.cloud_tf_available && status.imu_tf_available && status.pose_valid && pose_tf_valid &&
      status.last_match_success && !status.lost && status.map_loaded && !status.using_fallback_map &&
      status.match_metrics_valid && status.inlier_metrics_valid;
    if (valid) {
      valid = status.best_cost >= 0 && status.best_cost <= limit("max_best_cost") &&
        status.inlier_count >= profile_.at("min_inliers").get<int>() &&
        status.inlier_cost >= 0 && status.inlier_cost <= limit("max_inlier_cost") &&
        status.map_id == profile_.at("map_id") && status.team == profile_.at("team") &&
        status.extrinsic_id == profile_.at("extrinsic_id") &&
        status.implementation_id == profile_.at("implementation_id") &&
        status.configuration_id == profile_.at("configuration_id");
    }
    const bool new_result = status.processor_session != session_ || status.result_sequence != sequence_;
    const auto & position = status.pose.pose.position;
    const auto & orientation = status.pose.pose.orientation;
    const double yaw = std::atan2(2 * orientation.w * orientation.z,
      1 - 2 * orientation.z * orientation.z);
    if (valid && new_result && count_ > 0 && session_ == status.processor_session) {
      valid = std::hypot(position.x - previous_x_, position.y - previous_y_) <= limit("max_pose_jump") &&
        std::abs(std::atan2(std::sin(yaw - previous_yaw_), std::cos(yaw - previous_yaw_))) <= limit("max_yaw_jump");
    }
    if (!valid || (session_ != 0 && status.processor_session != session_)) { reset(); }
    if (!valid) { return false; }
    if (count_ == 0) { started_s_ = steady_s; }
    if (new_result || count_ == 0) {
      ++count_;
      session_ = status.processor_session;
      sequence_ = status.result_sequence;
      previous_x_ = position.x;
      previous_y_ = position.y;
      previous_yaw_ = yaw;
    }
    return count_ >= 10 && steady_s - started_s_ >= 1.0;
  }
  /** @brief Reset stability after failed inputs, a new session or a clock rollback. */
  void reset() { count_ = 0; started_s_ = 0; }
  /** @brief Read a validated profile limit or retain the original unapproved input deadline. */
  double limit(const std::string & key, double fallback = 0.5) const {
    if (!approved_) { return fallback; }
    return profile_.at(key).get<double>();
  }
private:
  nlohmann::json profile_ = nlohmann::json::object();
  bool approved_{};
  unsigned count_{};
  uint64_t session_{}, sequence_{};
  double started_s_{}, previous_x_{}, previous_y_{}, previous_yaw_{};
};
}  // namespace gn10_pointcloud_localization

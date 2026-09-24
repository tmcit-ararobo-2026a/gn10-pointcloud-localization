#include "gn10_pointcloud_localization/pose_fusion_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <Eigen/Cholesky>

namespace gn10 {

double wrapYaw(double yaw) { return std::atan2(std::sin(yaw), std::cos(yaw)); }

namespace {
bool finite(const Pose2d& pose)
{
    return std::isfinite(pose.x) && std::isfinite(pose.y) && std::isfinite(pose.yaw);
}
}  // namespace

PoseFusionFilter::PoseFusionFilter(FusionConfig config) : config_(config) {}

void PoseFusionFilter::reset() { history_.clear(); }

bool PoseFusionFilter::hasPose() const { return !history_.empty() && history_.back().valid; }

double PoseFusionFilter::latestStamp() const
{
    return history_.empty() ? -std::numeric_limits<double>::infinity() : history_.back().stamp;
}

Pose2d PoseFusionFilter::pose() const { return history_.back().map; }

Eigen::Matrix3d PoseFusionFilter::covariance() const { return history_.back().covariance; }

void PoseFusionFilter::predict(size_t index)
{
    const auto& previous = history_[index - 1];
    auto& current = history_[index];
    current.valid = previous.valid;
    if (!current.valid) return;

    const double dt = current.stamp - previous.stamp;
    const double odx = current.odom.x - previous.odom.x;
    const double ody = current.odom.y - previous.odom.y;
    const double co = std::cos(previous.odom.yaw);
    const double so = std::sin(previous.odom.yaw);
    const double local_x = co * odx + so * ody;
    const double local_y = -so * odx + co * ody;
    const double cm = std::cos(previous.map.yaw);
    const double sm = std::sin(previous.map.yaw);
    current.map.x = previous.map.x + cm * local_x - sm * local_y;
    current.map.y = previous.map.y + sm * local_x + cm * local_y;
    current.map.yaw = wrapYaw(
        previous.map.yaw + wrapYaw(current.odom.yaw - previous.odom.yaw)
    );

    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0, 2) = -sm * local_x - cm * local_y;
    F(1, 2) = cm * local_x - sm * local_y;
    const double distance = std::hypot(local_x, local_y);
    const double rotation = std::abs(wrapYaw(current.odom.yaw - previous.odom.yaw));
    const double variance_xy =
        std::pow(config_.process_translation_per_m, 2) * distance +
        std::pow(config_.process_translation_per_s, 2) * dt;
    const double variance_yaw =
        std::pow(config_.process_yaw_per_rad, 2) * rotation +
        std::pow(config_.process_yaw_per_s, 2) * dt;
    Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
    Q(0, 0) = Q(1, 1) = variance_xy;
    Q(2, 2) = variance_yaw;
    current.covariance = F * previous.covariance * F.transpose() + Q;
}

bool PoseFusionFilter::correct(size_t index)
{
    auto& sample = history_[index];
    if (!sample.match) return true;
    const auto& measurement = sample.match->pose;
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    R(0, 0) = R(1, 1) = std::pow(config_.match_xy_stddev, 2);
    R(2, 2) = std::pow(config_.match_yaw_stddev, 2);
    if (!sample.valid) {
        sample.map = measurement;
        sample.map.yaw = wrapYaw(sample.map.yaw);
        sample.covariance = R;
        sample.valid = true;
        return true;
    }

    Eigen::Vector3d innovation(
        measurement.x - sample.map.x, measurement.y - sample.map.y,
        wrapYaw(measurement.yaw - sample.map.yaw)
    );
    const Eigen::Matrix3d S = sample.covariance + R;
    const auto solver = S.ldlt();
    if (solver.info() != Eigen::Success ||
        innovation.dot(solver.solve(innovation)) > config_.innovation_gate) {
        sample.match.reset();
        return false;
    }
    const Eigen::Matrix3d K = sample.covariance * solver.solve(Eigen::Matrix3d::Identity());
    const Eigen::Vector3d change = K * innovation;
    sample.map.x += change.x();
    sample.map.y += change.y();
    sample.map.yaw = wrapYaw(sample.map.yaw + change.z());
    const Eigen::Matrix3d I_K = Eigen::Matrix3d::Identity() - K;
    sample.covariance = I_K * sample.covariance * I_K.transpose() + K * R * K.transpose();
    return true;
}

void PoseFusionFilter::replayFrom(size_t index)
{
    for (size_t i = index; i < history_.size(); ++i) {
        if (i > 0) predict(i);
        correct(i);
    }
}

void PoseFusionFilter::addOdometry(double stamp, Pose2d pose)
{
    if (!std::isfinite(stamp) || !finite(pose)) return;
    pose.yaw = wrapYaw(pose.yaw);
    if (!history_.empty()) {
        const double dt = stamp - history_.back().stamp;
        const double step = std::hypot(
            pose.x - history_.back().odom.x, pose.y - history_.back().odom.y
        );
        const double yaw_step = std::abs(wrapYaw(pose.yaw - history_.back().odom.yaw));
        if (dt <= 0.0 || dt > config_.max_odom_gap_s) {
            if (dt < -0.5 || dt > config_.max_odom_gap_s) reset();
            else return;
        } else if (step > config_.max_odom_step_m ||
                   yaw_step > config_.max_odom_step_yaw_rad) {
            reset();
        }
    }
    history_.push_back(Sample{stamp, pose, {}, Eigen::Matrix3d::Zero(), false, {}});
    if (history_.size() > 1) predict(history_.size() - 1);
    while (history_.size() > 1 &&
           history_.back().stamp - history_.front().stamp > config_.history_s) {
        history_.pop_front();
        history_.front().match.reset();  // Keep its computed state as the replay anchor.
    }
}

bool PoseFusionFilter::addMatch(double stamp, Pose2d pose)
{
    if (history_.empty() || !std::isfinite(stamp) || !finite(pose)) return false;
    auto nearest = history_.begin();
    double best = std::abs(nearest->stamp - stamp);
    for (auto it = history_.begin() + 1; it != history_.end(); ++it) {
        const double skew = std::abs(it->stamp - stamp);
        if (skew < best) {
            best = skew;
            nearest = it;
        }
    }
    if (best > config_.max_match_skew_s || nearest->match) return false;
    const size_t index = static_cast<size_t>(nearest - history_.begin());
    nearest->match = Match{pose};
    if (index == 0) {
        if (!correct(0)) return false;
        replayFrom(1);
        return true;
    }
    replayFrom(index);
    return history_[index].match.has_value();
}

}  // namespace gn10

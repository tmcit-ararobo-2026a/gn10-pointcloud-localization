#pragma once

#include <Eigen/Core>

#include <deque>
#include <optional>

namespace gn10 {

struct Pose2d {
    double x{0.0};
    double y{0.0};
    double yaw{0.0};
};

struct FusionConfig {
    double history_s{5.0};
    double max_odom_gap_s{3.0};
    double max_odom_step_m{2.0};
    double max_odom_step_yaw_rad{1.5};
    double max_match_skew_s{0.15};
    double process_translation_per_m{0.03};
    double process_translation_per_s{0.05};
    double process_yaw_per_rad{0.03};
    double process_yaw_per_s{0.03};
    double match_xy_stddev{0.08};
    double match_yaw_stddev{0.08};
    double max_match_translation_m{0.5};
    double max_match_yaw_rad{0.35};
    double max_recovery_translation_m{0.75};
    double max_recovery_yaw_rad{0.75};
    double recovery_step_m{0.10};
    double recovery_step_yaw_rad{0.10};
    double innovation_gate{16.27};  // chi-square 3 DoF, 99.9%
};

enum class MatchRejection { None, NoOdometry, Invalid, Timestamp, Duplicate, Innovation };

class PoseFusionFilter
{
public:
    explicit PoseFusionFilter(FusionConfig config);

    void addOdometry(double stamp, Pose2d pose);
    bool addMatch(double stamp, Pose2d pose, bool confirmed_recovery = false);
    std::optional<Pose2d> predictionAt(double stamp) const;
    MatchRejection lastMatchRejection() const;
    bool hasPose() const;
    double latestStamp() const;
    Pose2d pose() const;
    Eigen::Matrix3d covariance() const;
    void reset();
    bool addMapPrediction(double stamp); // No observed control: hold pose, increase uncertainty.
    bool resynchronizeOdometry(double stamp, Pose2d odom); // Start a new control segment, no unknown increment.
    bool setMapPose(Pose2d pose); // Explicit operator initialization at latest odometry.

private:
    struct Match { Pose2d pose; bool confirmed_recovery{false}; double timing_skew_s{0.0}; };
    struct Sample {
        double stamp;
        Pose2d odom;
        Pose2d map;
        Eigen::Matrix3d covariance{Eigen::Matrix3d::Zero()};
        bool valid{false};
        std::optional<Match> match;
    };

    void predict(size_t index);
    bool correct(size_t index);
    void replayFrom(size_t index);

    FusionConfig config_;
    std::deque<Sample> history_;
    MatchRejection last_match_rejection_{MatchRejection::None};
};

double wrapYaw(double yaw);

}  // namespace gn10

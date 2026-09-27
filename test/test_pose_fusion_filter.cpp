#include "gn10_pointcloud_localization/pose_fusion_filter.hpp"

#include <cmath>
#include <stdexcept>

namespace {
void require(bool condition) { if (!condition) throw std::runtime_error("fusion check failed"); }
bool near(double actual, double expected, double tolerance = 1e-6)
{
    return std::abs(actual - expected) < tolerance;
}
}  // namespace

int main()
{
    {
        gn10::PoseFusionFilter map_only({});
        require(!map_only.addMapPrediction(1.0));
        map_only.addOdometry(1.0,{0,0,0});require(map_only.addMatch(1.0,{10,0,0}));
        const auto covariance=map_only.covariance();
        require(map_only.addMapPrediction(1.2));require(near(map_only.pose().x,10));
        require(map_only.covariance()(0,0)>covariance(0,0));
        require(!map_only.addMapPrediction(1.1));
        require(!map_only.addMatch(1.2,{12,0,0})); // unknown control is not permission for a jump
        require(map_only.addMatch(1.2,{10.1,0,0}));
        const double observed=map_only.pose().x;
        require(!map_only.resynchronizeOdometry(1.1,{20,0,0}));
        require(map_only.resynchronizeOdometry(1.3,{20,0,0}));require(near(map_only.pose().x,observed));
        map_only.addOdometry(1.4,{20.1,0,0});require(near(map_only.pose().x,observed+.1));
    }

    {
        gn10::PoseFusionFilter timed({});timed.addOdometry(0,{0,0,0});
        require(timed.addMatch(0,{10,0,0}));timed.addOdometry(.1,{.2,0,0});
        require(timed.addMatch(.04,{10.08,0,0}));
        require(near(timed.pose().x,10.2)); // No 4 cm correction caused by the nearest endpoint.
        require(!timed.addMatch(.04,{10.08,0,0}));
        require(timed.lastMatchRejection()==gn10::MatchRejection::Duplicate);
    }
    gn10::PoseFusionFilter filter({});
    filter.addOdometry(0.0, {0.0, 0.0, 3.13});
    require(!filter.hasPose());
    require(filter.addMatch(0.0, {5.0, 2.0, 3.13}));
    filter.addOdometry(0.1, {-1.0, 0.0, -3.13});
    require(filter.hasPose());
    require(near(filter.pose().x, 4.0, 0.03));
    require(std::abs(gn10::wrapYaw(filter.pose().yaw - (-3.13))) < 0.03);

    filter.addOdometry(0.2, {-2.0, 0.0, -3.13});
    const double before = filter.pose().x;
    require(filter.addMatch(0.1, {4.1, 2.0, -3.13}));
    require(filter.pose().x > before);
    require(filter.pose().x < before + 0.1);
    require(!filter.addMatch(0.2, {100.0, 100.0, 0.0}));
    require(filter.lastMatchRejection() == gn10::MatchRejection::Innovation);
    require(filter.pose().x < 4.0);

    require(!filter.addMatch(10.0, {1.0, 1.0, 0.0}));
    require(filter.lastMatchRejection() == gn10::MatchRejection::Timestamp);

    const auto held = filter.pose();
    filter.addOdometry(4.0, {0.0, 0.0, 0.0});
    require(filter.hasPose());
    require(near(filter.pose().x, held.x));
    require(!filter.predictionAt(0.2));
    filter.reset();
    filter.addOdometry(4.0, {0.0, 0.0, 0.0});
    require(!filter.hasPose());
    require(filter.addMatch(4.0, {1.0, 1.0, 0.0}));
    require(near(filter.pose().x, 1.0));
    filter.addOdometry(4.1, {10.0, 0.0, 0.0});
    require(!filter.hasPose());
    gn10::PoseFusionFilter gap_filter({});
    gap_filter.addOdometry(0.0, {});
    require(gap_filter.addMatch(0.0, {1, 2, 0}));
    gap_filter.addOdometry(2.1, {0.2, 0, 0});
    require(gap_filter.hasPose());
    require(near(gap_filter.pose().x, 1.2));
    // Even large covariance cannot authorize a distant map correction.
    gn10::FusionConfig noisy;
    noisy.process_translation_per_s = 10;
    gn10::PoseFusionFilter bounded(noisy);
    bounded.addOdometry(0.0, {});
    require(bounded.addMatch(0.0, {}));
    bounded.addOdometry(0.1, {});
    require(!bounded.addMatch(0.1, {2, 0, 0}));
    require(bounded.lastMatchRejection() == gn10::MatchRejection::Innovation);
    require(near(bounded.pose().x, 0));
    require(!bounded.addMatch(0.1, {0, 0, 0.7}));
    require(bounded.setMapPose({3, 4, 0.2}));
    require(near(bounded.pose().x, 3));
    bounded.addOdometry(0.2, {0.1, 0, 0});
    require(near(bounded.pose().x, 3+0.1*std::cos(0.2)));
    require(near(bounded.pose().y, 4+0.1*std::sin(0.2)));
    require(!bounded.addMatch(0.0, {})); // old epoch/history cannot undo initialization
    bounded.reset();
    require(!bounded.setMapPose({}));
    gn10::PoseFusionFilter recovery(noisy);
    recovery.addOdometry(0.0, {});
    require(recovery.addMatch(0.0, {}));
    recovery.addOdometry(0.1, {});
    require(!recovery.addMatch(0.1, {0.65, 0, 0.65}));
    require(recovery.predictionAt(0.1).has_value());
    require(recovery.addMatch(0.1, {0.65, 0, 0.65}, true));
    require(recovery.pose().x > 0 && recovery.pose().x <= 0.100001);
    require(recovery.pose().yaw > 0 && recovery.pose().yaw <= 0.100001);
    require(recovery.covariance().diagonal().minCoeff() > 0);
    recovery.addOdometry(0.2, {});
    require(!recovery.addMatch(0.2, {5, 0, 0}, true));
    require(!recovery.predictionAt(10).has_value());
    return 0;
}

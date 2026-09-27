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
    return 0;
}

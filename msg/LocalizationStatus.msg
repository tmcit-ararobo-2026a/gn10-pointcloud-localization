std_msgs/Header header
# Measurement and pose timestamps are preserved across periodic monitor reports.
builtin_interfaces/Time measurement_stamp
builtin_interfaces/Time pose_stamp
uint64 processor_session
uint64 result_sequence
uint64 report_sequence
bool processing_fresh
bool cloud_fresh
bool imu_fresh
bool cloud_tf_available
bool imu_tf_available
bool last_match_success
bool lost
bool map_loaded
bool using_fallback_map
bool match_metrics_valid
float64 best_cost
bool inlier_metrics_valid
int32 inlier_count
float64 inlier_cost
# The existing fixed covariance is not a measured uncertainty estimate.
bool covariance_valid
# Remains false until a Ready-quality contract has been approved and implemented.
bool ready_quality_valid
string reason

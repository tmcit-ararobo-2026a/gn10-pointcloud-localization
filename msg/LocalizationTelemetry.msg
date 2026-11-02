# Telemetry and metrics for gn10_pointcloud_localization parameter tuning and diagnostics.
# Published at matching frame rate (~50Hz) or logged for offline analysis.

std_msgs/Header header

# Matching costs
float32 best_cost              # Total composite matching cost (3D + 2D weighted)
float32 cost_3d                # Normalized 3D LiDAR ESDF residual cost
float32 cost_2d                # Normalized 2D LiDAR ESDF residual cost
float32 weight_2d              # Effective weight applied to 2D LiDAR residual [0.0, 1.0]
float32 inlier_cost            # Average ESDF distance of inlier obstacle points [m]

# Point counts and ratios
int32 inlier_count             # Number of inlier obstacle points
float32 inlier_ratio           # Ratio of inliers to total obstacles (inlier_count / obstacle_count_total)
int32 obstacle_count_3d        # Number of 3D obstacle points evaluated
int32 obstacle_count_2d        # Number of 2D obstacle points evaluated
int32 obstacle_count_total     # Total obstacle points evaluated (3D + 2D)

# Motion, velocity, and search windows
float32 linear_speed           # Estimated linear speed magnitude [m/s]
float32 velocity_x             # Estimated velocity along map X [m/s]
float32 velocity_y             # Estimated velocity along map Y [m/s]
float32 search_range_xy        # Active local XY search half-width [m]
float32 search_range_yaw       # Active local Yaw search half-width [rad]

# Localization tracking status
bool match_accepted            # Whether current match satisfied acceptance criteria
bool is_lost                   # Whether localization is currently in lost state
int32 lost_frame_count         # Consecutive lost/failed frame counter
bool global_recovered          # Whether current frame was recovered via global search

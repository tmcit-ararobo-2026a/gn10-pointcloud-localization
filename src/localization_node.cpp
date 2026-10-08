#include "gn10_pointcloud_localization/localization_node.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_ros/create_timer_ros.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

LocalizationNode::LocalizationNode() : Node("gn10_localization_node")
{
    declareAndGetParameters();

    solver_ =
        std::make_unique<PoseSolver>(this->get_parameter("filter_params.max_points").as_int());

    GlobalSearchConfig search_config;
    search_config.range_min_x       = global_range_min_x_;
    search_config.range_max_x       = global_range_max_x_;
    search_config.range_min_y       = global_range_min_y_;
    search_config.range_max_y       = global_range_max_y_;
    search_config.step_xy           = global_step_xy_;
    search_config.step_yaw          = global_step_yaw_;
    search_config.downsample_stride = global_downsample_stride_;

    global_searcher_ = std::make_unique<GlobalSearcher>(search_config, this->get_logger());

    setupMapData();
    setupROSInterfaces();

    // マップ読み込み完了後に一度だけ ESDF マップをパブリッシュ
    if (publish_esdf_map_) {
        publishESDFMap();
    }
}

void LocalizationNode::declareAndGetParameters()
{
    this->declare_parameter("map_source_type", "json");
    this->declare_parameter("map_file_path", "");
    this->declare_parameter("map_objects", std::vector<std::string>{});
    this->declare_parameter("esdf.resolution", 0.05);
    this->declare_parameter("esdf.max_dist", 0.50);
    this->declare_parameter("esdf.publish_map", true);
    this->declare_parameter("esdf.publish_max_distance", -1.0);
    this->declare_parameter("esdf.publish_stride", 1);

    this->declare_parameter("frames.map_frame", "map");
    this->declare_parameter("frames.base_frame", "base_link");

    this->declare_parameter("topics.input_cloud", "/livox/lidar");
    this->declare_parameter("topics.input_imu", "/livox/imu");
    this->declare_parameter("topics.output_dynamic", "/dynamic_cloud");
    this->declare_parameter("topics.output_ground", "/ground_cloud");
    this->declare_parameter("topics.output_obstacle", "/obstacle_cloud");
    this->declare_parameter("topics.output_pose", "/platform_constraint");
    this->declare_parameter("topics.output_markers", "/field_map_markers");
    this->declare_parameter("topics.output_esdf_map", "/esdf_map");
    this->declare_parameter("topics.fused_prior", "/platform_constraint");
    this->declare_parameter("publish_tf", true);
    this->declare_parameter("fusion.use_prior", false);
    this->declare_parameter("fusion.prior_max_age_s", 0.25);

    this->declare_parameter("filter_params.max_range", 12.0);
    this->declare_parameter("filter_params.robot_radius", 0.6);
    this->declare_parameter("filter_params.robot_height_min", 0.0);
    this->declare_parameter("filter_params.robot_height_max", 1.2);
    this->declare_parameter("filter_params.ground_z_thresh", 0.08);
    this->declare_parameter("filter_params.max_points", 200000);
    this->declare_parameter("scan_accumulation.window_s", 0.10);
    this->declare_parameter("scan_accumulation.timestamp_field", "timestamp");
    this->declare_parameter("scan_accumulation.timestamp_scale", 1e-9);
    this->declare_parameter("scan_accumulation.timestamp_relative", false);
    const double window_s = this->get_parameter("scan_accumulation.window_s").as_double();
    if (!std::isfinite(window_s) || window_s < 0.0 || window_s > 0.5) {
        throw std::invalid_argument("scan_accumulation.window_s must be in [0, 0.5]");
    }
    scan_accumulator_   = ScanAccumulator(window_s);
    timestamp_field_    = this->get_parameter("scan_accumulation.timestamp_field").as_string();
    timestamp_scale_    = this->get_parameter("scan_accumulation.timestamp_scale").as_double();
    timestamp_relative_ = this->get_parameter("scan_accumulation.timestamp_relative").as_bool();
    if (!std::isfinite(timestamp_scale_) || timestamp_scale_ <= 0) {
        throw std::invalid_argument("scan_accumulation.timestamp_scale must be positive");
    }

    this->declare_parameter("matching_params.search_range_xy", 0.30);
    this->declare_parameter("matching_params.search_step_xy", 0.03);
    this->declare_parameter("matching_params.search_range_yaw", 0.15);
    this->declare_parameter("matching_params.search_step_yaw", 0.02);
    this->declare_parameter("matching_params.fine_refine", true);
    this->declare_parameter("matching_params.fine_refine_levels", 2);
    this->declare_parameter("matching_params.max_dist_thresh", 0.20);
    this->declare_parameter("matching_params.cost_threshold", 0.165);
    this->declare_parameter("matching_params.dynamic_dist_thresh", 0.15);
    this->declare_parameter("matching_params.field_min_x", -5.5);
    this->declare_parameter("matching_params.field_max_x", 5.5);
    this->declare_parameter("matching_params.field_min_y", -6.0);
    this->declare_parameter("matching_params.field_max_y", 6.0);
    this->declare_parameter("matching_params.use_map_bounds", true);
    this->declare_parameter("matching_params.inlier_dist_thresh", 0.08);
    this->declare_parameter("matching_params.min_inliers", 60);
    this->declare_parameter("matching_params.inlier_cost_thresh", 0.05);

    this->declare_parameter("lidar_2d.enable", false);
    this->declare_parameter("lidar_2d.topic", "/lakibeam/scan");
    this->declare_parameter("lidar_2d.frame_id", "lakibeam_frame");

    this->declare_parameter("global_search.range_min_x", -5.25);
    this->declare_parameter("global_search.range_max_x", 5.25);
    this->declare_parameter("global_search.range_min_y", -5.70);
    this->declare_parameter("global_search.range_max_y", 5.70);
    this->declare_parameter("global_search.range_yaw_diff", 0.7854);
    this->declare_parameter("global_search.step_xy", 0.30);
    this->declare_parameter("global_search.step_yaw", 0.2618);
    this->declare_parameter("global_search.downsample_stride", 2);
    this->declare_parameter("global_search.lost_count_thresh", 5);

    this->declare_parameter("initial_pose.x", -4.0);
    this->declare_parameter("initial_pose.y", -4.0);
    this->declare_parameter("initial_pose.yaw", -1.5708);
    this->declare_parameter("initial_pose.use_for_local_search", false);

    map_frame_  = this->get_parameter("frames.map_frame").as_string();
    base_frame_ = this->get_parameter("frames.base_frame").as_string();

    filter_params_.range_max =
        static_cast<float>(this->get_parameter("filter_params.max_range").as_double());
    filter_params_.robot_radius =
        static_cast<float>(this->get_parameter("filter_params.robot_radius").as_double());
    filter_params_.robot_height_min =
        static_cast<float>(this->get_parameter("filter_params.robot_height_min").as_double());
    filter_params_.robot_height_max =
        static_cast<float>(this->get_parameter("filter_params.robot_height_max").as_double());
    filter_params_.ground_z_thresh =
        static_cast<float>(this->get_parameter("filter_params.ground_z_thresh").as_double());

    match_params_.range_xy =
        static_cast<float>(this->get_parameter("matching_params.search_range_xy").as_double());
    match_params_.step_xy =
        static_cast<float>(this->get_parameter("matching_params.search_step_xy").as_double());
    match_params_.range_yaw =
        static_cast<float>(this->get_parameter("matching_params.search_range_yaw").as_double());
    match_params_.step_yaw =
        static_cast<float>(this->get_parameter("matching_params.search_step_yaw").as_double());
    match_params_.fine_refine = this->get_parameter("matching_params.fine_refine").as_bool();
    match_params_.fine_refine_levels =
        int(this->get_parameter("matching_params.fine_refine_levels").as_int());
    if (match_params_.fine_refine_levels < 1 || match_params_.fine_refine_levels > 3) {
        throw std::invalid_argument("matching_params.fine_refine_levels must be in [1, 3]");
    }
    match_params_.max_dist_thresh =
        static_cast<float>(this->get_parameter("matching_params.max_dist_thresh").as_double());
    match_params_.cost_threshold =
        static_cast<float>(this->get_parameter("matching_params.cost_threshold").as_double());
    match_params_.dynamic_dist_thresh =
        static_cast<float>(this->get_parameter("matching_params.dynamic_dist_thresh").as_double());
    match_params_.field_min_x =
        static_cast<float>(this->get_parameter("matching_params.field_min_x").as_double());
    match_params_.field_max_x =
        static_cast<float>(this->get_parameter("matching_params.field_max_x").as_double());
    match_params_.field_min_y =
        static_cast<float>(this->get_parameter("matching_params.field_min_y").as_double());
    match_params_.field_max_y =
        static_cast<float>(this->get_parameter("matching_params.field_max_y").as_double());
    match_params_.inlier_dist_thresh =
        static_cast<float>(this->get_parameter("matching_params.inlier_dist_thresh").as_double());
    match_params_.min_inliers = this->get_parameter("matching_params.min_inliers").as_int();
    match_params_.inlier_cost_thresh =
        static_cast<float>(this->get_parameter("matching_params.inlier_cost_thresh").as_double());

    use_2d_lidar_   = this->get_parameter("lidar_2d.enable").as_bool();
    topic_2d_lidar_ = this->get_parameter("lidar_2d.topic").as_string();
    lidar_2d_frame_ = this->get_parameter("lidar_2d.frame_id").as_string();

    global_range_min_x_ =
        static_cast<float>(this->get_parameter("global_search.range_min_x").as_double());
    global_range_max_x_ =
        static_cast<float>(this->get_parameter("global_search.range_max_x").as_double());
    global_range_min_y_ =
        static_cast<float>(this->get_parameter("global_search.range_min_y").as_double());
    global_range_max_y_ =
        static_cast<float>(this->get_parameter("global_search.range_max_y").as_double());
    global_range_yaw_diff_ =
        static_cast<float>(this->get_parameter("global_search.range_yaw_diff").as_double());
    global_step_xy_ = static_cast<float>(this->get_parameter("global_search.step_xy").as_double());
    global_step_yaw_ =
        static_cast<float>(this->get_parameter("global_search.step_yaw").as_double());
    global_downsample_stride_ = std::max(
        1, static_cast<int>(this->get_parameter("global_search.downsample_stride").as_int())
    );
    lost_threshold_count_ = this->get_parameter("global_search.lost_count_thresh").as_int();
    publish_tf_           = this->get_parameter("publish_tf").as_bool();
    use_fused_prior_      = this->get_parameter("fusion.use_prior").as_bool();
    prior_max_age_s_      = this->get_parameter("fusion.prior_max_age_s").as_double();

    last_known_pose_.x   = static_cast<float>(this->get_parameter("initial_pose.x").as_double());
    last_known_pose_.y   = static_cast<float>(this->get_parameter("initial_pose.y").as_double());
    last_known_pose_.yaw = static_cast<float>(this->get_parameter("initial_pose.yaw").as_double());
    predicted_pose_      = last_known_pose_;
    // A known start pose resolves the field's near-symmetric global matches.
    // Do not publish it as a measurement; first require a successful cloud match.
    is_lost_ = !this->get_parameter("initial_pose.use_for_local_search").as_bool();

    topic_esdf_map_   = this->get_parameter("topics.output_esdf_map").as_string();
    publish_esdf_map_ = this->get_parameter("esdf.publish_map").as_bool();
    esdf_publish_max_distance_m_ =
        static_cast<float>(this->get_parameter("esdf.publish_max_distance").as_double());
    esdf_publish_stride_ =
        std::max(1, static_cast<int>(this->get_parameter("esdf.publish_stride").as_int()));
}

void LocalizationNode::setupMapData()
{
    std::string map_source = this->get_parameter("map_source_type").as_string();
    std::string file_path  = this->get_parameter("map_file_path").as_string();
    float esdf_res         = static_cast<float>(this->get_parameter("esdf.resolution").as_double());
    float esdf_max_dist    = static_cast<float>(this->get_parameter("esdf.max_dist").as_double());

    // 相対パスの場合、パッケージの map/ ディレクトリを基準に探索
    if (!file_path.empty()) {
        std::filesystem::path p(file_path);
        if (p.is_relative()) {
            std::string package_share =
                ament_index_cpp::get_package_share_directory("gn10_pointcloud_localization");
            std::filesystem::path map_dir_path = std::filesystem::path(package_share) / "map" / p;
            if (std::filesystem::exists(map_dir_path)) {
                file_path = map_dir_path.string();
            } else {
                // share側になければソース側の map
                // ディレクトリもチェック（シンボリックリンクや開発時対応）
                file_path = map_dir_path.string();
            }
        }
    }

    if (map_source == "esdf") {
        if (file_path.empty()) {
            RCLCPP_ERROR(
                this->get_logger(), "map_source_type is 'esdf' but map_file_path is empty!"
            );
        } else if (!esdf_map_.loadBinary(file_path)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load ESDF file: %s", file_path.c_str());
        } else {
            RCLCPP_INFO(
                this->get_logger(),
                "Loaded ESDF map: %dx%dx%d, res=%.3f",
                esdf_map_.header().size_x,
                esdf_map_.header().size_y,
                esdf_map_.header().size_z,
                esdf_map_.header().resolution
            );
            configureESDFBounds();
            solver_->setESDFMap(esdf_map_);
            return;
        }
    } else if (map_source == "pcd") {
        if (file_path.empty()) {
            RCLCPP_ERROR(
                this->get_logger(), "map_source_type is 'pcd' but map_file_path is empty!"
            );
        } else {
            // 自動キャッシュチェック (.pcd -> .pcd.esdf または .esdf)
            std::string cache_path = file_path + ".esdf";
            bool loaded_cache      = esdf_map_.loadBinary(cache_path);
            if (loaded_cache) {
                RCLCPP_INFO(
                    this->get_logger(), "Found ESDF cache: %s. Loaded directly.", cache_path.c_str()
                );
            } else {
                RCLCPP_INFO(
                    this->get_logger(),
                    "Building ESDF map from PCD: %s (res=%.3f, max_dist=%.3f)...",
                    file_path.c_str(),
                    esdf_res,
                    esdf_max_dist
                );
                if (esdf_map_.buildFromPCD(file_path, esdf_res, esdf_max_dist)) {
                    esdf_map_.saveBinary(cache_path);
                    RCLCPP_INFO(
                        this->get_logger(), "ESDF built and cached to: %s", cache_path.c_str()
                    );
                } else {
                    RCLCPP_ERROR(
                        this->get_logger(), "Failed to build ESDF from PCD: %s", file_path.c_str()
                    );
                }
            }

            if (!esdf_map_.empty()) {
                configureESDFBounds();
                solver_->setESDFMap(esdf_map_);
                return;
            }
        }
    }

    if (map_source == "json") {
        std::string json_path = file_path;
        if (json_path.empty()) {
            json_path =
                ament_index_cpp::get_package_share_directory("gn10_pointcloud_localization") +
                "/map/nhk2026_map.json";
        }
        map_objects_ = MapLoader::loadFromJSON(json_path);
    } else if (map_source == "ros2_param") {
        auto param_list = this->get_parameter("map_objects").as_string_array();
        map_objects_    = MapLoader::loadFromParams(param_list);
    }

    if (map_objects_.empty()) {
        RCLCPP_WARN(
            this->get_logger(), "Map is empty or failed to load. Falling back to default map."
        );
        map_objects_ = MapLoader::createNHK2026FieldMap();
    }

    // JSON / パラメータ読み込み時も ESDF 3D テクスチャを自動生成して高速化
    RCLCPP_INFO(this->get_logger(), "Building 3D ESDF texture from field objects...");
    if (esdf_map_.buildFromFieldObjects(
            map_objects_,
            esdf_res,
            match_params_.field_min_x,
            match_params_.field_max_x,
            match_params_.field_min_y,
            match_params_.field_max_y,
            -0.2f,
            2.0f,
            esdf_max_dist
        )) {
        RCLCPP_INFO(
            this->get_logger(),
            "ESDF texture ready: %dx%dx%d",
            esdf_map_.header().size_x,
            esdf_map_.header().size_y,
            esdf_map_.header().size_z
        );
        solver_->setESDFMap(esdf_map_);
    } else {
        RCLCPP_WARN(this->get_logger(), "Falling back to geometric object SDF solver.");
        solver_->setMap(map_objects_);
    }
}

void LocalizationNode::configureESDFBounds()
{
    // A PCD/ESDF map can cover a different room than the default NHK field.
    // Keep an explicitly requested crop only when use_map_bounds is disabled.
    if (!this->get_parameter("matching_params.use_map_bounds").as_bool()) return;
    match_params_.useESDFBounds(esdf_map_.header());
    this->set_parameters(
        {rclcpp::Parameter("matching_params.field_min_x", double(match_params_.field_min_x)),
         rclcpp::Parameter("matching_params.field_max_x", double(match_params_.field_max_x)),
         rclcpp::Parameter("matching_params.field_min_y", double(match_params_.field_min_y)),
         rclcpp::Parameter("matching_params.field_max_y", double(match_params_.field_max_y))}
    );
    RCLCPP_INFO(
        this->get_logger(),
        "ESDF matching bounds: x=[%.3f, %.3f], y=[%.3f, %.3f]",
        match_params_.field_min_x,
        match_params_.field_max_x,
        match_params_.field_min_y,
        match_params_.field_max_y
    );
}

void LocalizationNode::setupROSInterfaces()
{
    auto clock           = this->get_clock();
    tf_buffer_           = std::make_unique<tf2_ros::Buffer>(clock);
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(), this->get_node_timers_interface()
    );
    tf_buffer_->setCreateTimerInterface(timer_interface);
    tf_listener_    = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    std::string topic_cloud = this->get_parameter("topics.input_cloud").as_string();
    sub_cloud_filter_.subscribe(this, topic_cloud, rmw_qos_profile_sensor_data);
    tf_filter_ = std::make_shared<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>>(
        sub_cloud_filter_,
        *tf_buffer_,
        base_frame_,
        10,
        this->get_node_logging_interface(),
        this->get_node_clock_interface(),
        std::chrono::milliseconds(100)
    );
    tf_filter_->registerCallback(&LocalizationNode::cloudCallback, this);

    std::string topic_imu = this->get_parameter("topics.input_imu").as_string();
    sub_imu_              = this->create_subscription<sensor_msgs::msg::Imu>(
        topic_imu,
        rclcpp::SensorDataQoS().keep_last(200),
        std::bind(&LocalizationNode::imuCallback, this, std::placeholders::_1)
    );
    if (use_fused_prior_) {
        sub_fused_prior_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            this->get_parameter("topics.fused_prior").as_string(),
            20,
            std::bind(&LocalizationNode::fusedPriorCallback, this, std::placeholders::_1)
        );
    }

    pub_dynamic_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        this->get_parameter("topics.output_dynamic").as_string(), 10
    );
    pub_obstacle_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        this->get_parameter("topics.output_obstacle").as_string(), 10
    );
    pub_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        this->get_parameter("topics.output_ground").as_string(), 10
    );
    pub_platform_pose_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        this->get_parameter("topics.output_pose").as_string(), 10
    );
    pub_map_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        this->get_parameter("topics.output_markers").as_string(), rclcpp::QoS(1).transient_local()
    );
    if (publish_esdf_map_) {
        pub_esdf_map_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            topic_esdf_map_, rclcpp::QoS(1).transient_local().reliable()
        );
    }

    if (use_2d_lidar_) {
        sub_2d_lidar_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            topic_2d_lidar_,
            rclcpp::SensorDataQoS(),
            std::bind(&LocalizationNode::scan2dCallback, this, std::placeholders::_1)
        );
        RCLCPP_INFO(
            this->get_logger(), "Subscribed to 2D LiDAR on topic: %s", topic_2d_lidar_.c_str()
        );
    }

    using namespace std::chrono_literals;
    map_timer_ =
        this->create_wall_timer(1s, std::bind(&LocalizationNode::publishFieldMapMarkers, this));
}

void LocalizationNode::scan2dCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    geometry_msgs::msg::TransformStamped tf_stamped;
    try {
        tf_stamped = tf_buffer_->lookupTransform(
            base_frame_,
            msg->header.frame_id,
            msg->header.stamp,
            rclcpp::Duration::from_seconds(0.05)
        );
    } catch (const tf2::TransformException&) {
        try {
            tf_stamped =
                tf_buffer_->lookupTransform(base_frame_, msg->header.frame_id, tf2::TimePointZero);
        } catch (const tf2::TransformException&) {
            return;
        }
    }

    const double tx = tf_stamped.transform.translation.x;
    const double ty = tf_stamped.transform.translation.y;
    const double tz = tf_stamped.transform.translation.z;
    const double qx = tf_stamped.transform.rotation.x;
    const double qy = tf_stamped.transform.rotation.y;
    const double qz = tf_stamped.transform.rotation.z;
    const double qw = tf_stamped.transform.rotation.w;

    const double siny_cosp = 2.0 * (qw * qz + qx * qy);
    const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    const double yaw       = std::atan2(siny_cosp, cosy_cosp);
    const float cos_y      = static_cast<float>(std::cos(yaw));
    const float sin_y      = static_cast<float>(std::sin(yaw));

    std::vector<float> points;
    points.reserve(msg->ranges.size() * 3);

    const float angle_min = msg->angle_min;
    const float angle_inc = msg->angle_increment;
    const float range_min = msg->range_min;
    const float range_max = std::min(msg->range_max, filter_params_.range_max);

    for (size_t i = 0; i < msg->ranges.size(); ++i) {
        const float r = msg->ranges[i];
        if (!std::isfinite(r) || r < range_min || r > range_max) continue;

        const float angle = angle_min + static_cast<float>(i) * angle_inc;
        const float xl    = r * std::cos(angle);
        const float yl    = r * std::sin(angle);

        const float xb = cos_y * xl - sin_y * yl + static_cast<float>(tx);
        const float yb = sin_y * xl + cos_y * yl + static_cast<float>(ty);
        const float zb = static_cast<float>(tz);

        if (xb * xb + yb * yb <= filter_params_.robot_radius * filter_params_.robot_radius) {
            continue;
        }

        points.push_back(xb);
        points.push_back(yb);
        points.push_back(zb);
    }

    {
        std::lock_guard<std::mutex> lock(lidar_2d_mutex_);
        latest_2d_points_ = std::move(points);
        latest_2d_stamp_  = msg->header.stamp;
    }
}

void LocalizationNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    float h_transform[12];
    if (!getTransformAsArray(msg->header.frame_id, msg->header.stamp, h_transform)) {
        return;
    }

    int64_t reference_ns;
    auto timed_points = extractTimedBasePoints(*msg, h_transform, reference_ns);
    if (timed_points.empty()) return;
    const rclcpp::Time reference_stamp(reference_ns, this->get_clock()->get_clock_type());
    if (scan_accumulator_.backwards(reference_ns)) {
        scan_accumulator_.clear();
        yaw_history_.clear();
        velocity_history_.clear();
        imu_initialized_ = false;
        has_velocity_    = false;
        is_lost_         = true;
    }
    if (!scan_accumulator_.append(timed_points, reference_ns)) return;
    // Points are already in base_link, at their individual acquisition times.
    std::vector<float> h_raw_cloud;
    const float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    std::vector<float> ground_pts, obstacle_pts, dynamic_pts;
    PoseCandidate best_pose;
    float best_cost = 0.0f;
    bool matched    = false;

    // 2D LiDAR (Lakibeam 1) の点群を取得
    std::vector<float> extra_2d_points;
    if (use_2d_lidar_) {
        std::lock_guard<std::mutex> lock(lidar_2d_mutex_);
        if (!latest_2d_points_.empty()) {
            const double age = (reference_stamp - latest_2d_stamp_).seconds();
            if (std::abs(age) < 0.20) {
                extra_2d_points = latest_2d_points_;
            }
        }
    }

    // 速度予測モデル + IMU積分による探索中心 (search_base_pose) の計算
    PoseCandidate search_base_pose;
    bool has_fresh_prior = false;
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        search_base_pose = last_known_pose_;
        double rotation  = 0;
        if (imu_initialized_ &&
            yaw_history_.delta(last_match_stamp_.nanoseconds(), reference_ns, rotation)) {
            search_base_pose.yaw += float(rotation);
        }
        if (has_velocity_ && !is_lost_) {
            const double dt = (reference_stamp - last_match_stamp_).seconds();
            if (dt > 0.0 && dt < 0.5) {
                search_base_pose.x = last_known_pose_.x + velocity_x_ * static_cast<float>(dt);
                search_base_pose.y = last_known_pose_.y + velocity_y_ * static_cast<float>(dt);
            }
        }
        if (use_fused_prior_ && prior_received_) {
            const double age = (reference_stamp - prior_stamp_).seconds();
            if (std::abs(age) <= prior_max_age_s_) {
                search_base_pose = fused_prior_;
                has_fresh_prior  = true;
            }
        }
    }

    bool fully_deskewed = false;
    h_raw_cloud         = scan_accumulator_.cloud(
        reference_ns,
        yaw_history_,
        search_base_pose.yaw,
        has_velocity_ ? velocity_x_ : 0.0,
        has_velocity_ ? velocity_y_ : 0.0,
        size_t(this->get_parameter("filter_params.max_points").as_int()),
        &fully_deskewed
    );
    if (!fully_deskewed) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            10000,
            "IMU does not cover the accumulation window; using only the newest scan"
        );
    }
    if (h_raw_cloud.size() < 150) return;

    int inlier_count  = 0;
    float inlier_cost = 1.0f;

    // A recent fused pose can reacquire locally even after the map matcher was lost.
    if (is_lost_ && has_fresh_prior) {
        matched = solver_->processPointCloud(
            h_raw_cloud,
            identity,
            filter_params_,
            match_params_,
            search_base_pose,
            dynamic_pts,
            best_pose,
            best_cost,
            &inlier_count,
            &inlier_cost,
            extra_2d_points.empty() ? nullptr : &extra_2d_points
        );
        const bool accepted = matched && std::isfinite(best_cost) &&
                              ((best_cost < match_params_.cost_threshold) ||
                               (inlier_count >= match_params_.min_inliers &&
                                inlier_cost < match_params_.inlier_cost_thresh));
        if (accepted) {
            is_lost_          = false;
            lost_frame_count_ = 0;
            matched           = true;
            last_match_stamp_ = reference_stamp;
        } else {
            matched = false;
            dynamic_pts.clear();
        }
    }

    if (is_lost_ && !matched) {
        float current_prior_yaw = 0.0f;
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            current_prior_yaw = predicted_pose_.yaw;
        }

        best_pose = global_searcher_->search(
            *solver_,
            h_raw_cloud,
            identity,
            filter_params_,
            match_params_,
            current_prior_yaw,
            global_range_yaw_diff_,
            best_cost
        );

        if (best_cost < match_params_.cost_threshold) {
            is_lost_          = false;
            lost_frame_count_ = 0;
            matched           = true;
            has_velocity_     = false;
            velocity_history_.clear();
            velocity_history_.push_back({reference_stamp, best_pose});
            last_match_stamp_ = reference_stamp;
            RCLCPP_INFO(
                this->get_logger(), "[GlobalSearch] Successfully recovered from lost state."
            );
        }
    } else if (!matched) {
        matched = solver_->processPointCloud(
            h_raw_cloud,
            identity,
            filter_params_,
            match_params_,
            search_base_pose,
            dynamic_pts,
            best_pose,
            best_cost,
            &inlier_count,
            &inlier_cost,
            extra_2d_points.empty() ? nullptr : &extra_2d_points
        );

        // 動的障害物に対するロバスト合否判定:
        // 通常の全点平均コストが閾値未満、または
        // 静止壁/オブジェクトの Inlier 点が十分多く、かつその平均残差が極小であれば合格
        const bool match_accepted = matched && std::isfinite(best_cost) &&
                                    ((best_cost < match_params_.cost_threshold) ||
                                     (inlier_count >= match_params_.min_inliers &&
                                      inlier_cost < match_params_.inlier_cost_thresh));

        if (match_accepted) {
            matched           = true;
            lost_frame_count_ = 0;
            is_lost_          = false;

            // Estimate translation over a fixed time interval rather than
            // differentiating alternating 50 ms scan noise into fake velocity.
            velocity_history_.push_back({reference_stamp, best_pose});
            while (velocity_history_.size() > 2 &&
                   (reference_stamp - velocity_history_[1].stamp).seconds() > 0.20) {
                velocity_history_.pop_front();
            }
            const auto& oldest = velocity_history_.front();
            const double dt    = (reference_stamp - oldest.stamp).seconds();
            if (dt >= 0.15 && dt < 0.5) {
                const float vx = (best_pose.x - oldest.pose.x) / float(dt);
                const float vy = (best_pose.y - oldest.pose.y) / float(dt);
                if (std::hypot(vx, vy) < 4.0f) {
                    velocity_x_   = vx;
                    velocity_y_   = vy;
                    has_velocity_ = true;
                } else {
                    has_velocity_ = false;
                }
            }
            last_match_stamp_ = reference_stamp;
        } else {
            matched = false;
            lost_frame_count_++;
            if (lost_frame_count_ >= lost_threshold_count_) {
                is_lost_      = true;
                has_velocity_ = false;
                velocity_history_.clear();
                RCLCPP_WARN(
                    this->get_logger(),
                    "Localization lost! (best_cost: %.4f, inliers: %d, inlier_cost: %.4f)",
                    best_cost,
                    inlier_count,
                    inlier_cost
                );
            }
        }
    }

    if (matched) {
        publishPoseAndTransform(reference_stamp, best_pose);
    }

    std_msgs::msg::Header out_header = msg->header;
    out_header.frame_id              = base_frame_;
    out_header.stamp                 = reference_stamp;
    const bool need_ground           = pub_ground_->get_subscription_count() > 0;
    const bool need_obstacle         = pub_obstacle_->get_subscription_count() > 0;
    if (need_ground || need_obstacle) {
        solver_->copyFilteredClouds(
            need_ground ? &ground_pts : nullptr, need_obstacle ? &obstacle_pts : nullptr
        );
        if (need_ground) publishCloud(pub_ground_, out_header, ground_pts);
        if (need_obstacle) publishCloud(pub_obstacle_, out_header, obstacle_pts);
    }
    publishCloud(pub_dynamic_, out_header, dynamic_pts);
}

void LocalizationNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    static Eigen::Matrix3d R_base_imu = Eigen::Matrix3d::Identity();
    static bool tf_initialized        = false;

    if (!tf_initialized) {
        try {
            geometry_msgs::msg::TransformStamped transform_stamped =
                tf_buffer_->lookupTransform(base_frame_, msg->header.frame_id, tf2::TimePointZero);
            const Eigen::Affine3d eigen_tf = tf2::transformToEigen(transform_stamped);
            R_base_imu                     = eigen_tf.rotation();
            tf_initialized                 = true;
        } catch (const tf2::TransformException&) {
            return;
        }
    }

    const Eigen::Vector3d omega_imu(
        msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z
    );
    const Eigen::Vector3d omega_base = R_base_imu * omega_imu;
    const rclcpp::Time stamp(msg->header.stamp);
    if (imu_initialized_ && stamp < last_imu_stamp_) {
        yaw_history_.clear();
        scan_accumulator_.clear();
        velocity_history_.clear();
        has_velocity_ = false;
        is_lost_      = true;
    }
    yaw_history_.add(stamp.nanoseconds(), omega_base.z());
    last_imu_stamp_  = stamp;
    imu_initialized_ = true;
}

void LocalizationNode::fusedPriorCallback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg
)
{
    if (msg->header.frame_id != map_frame_) return;
    const auto& pose = msg->pose.pose;
    if (!std::isfinite(pose.position.x) || !std::isfinite(pose.position.y)) return;
    const double yaw = tf2::getYaw(pose.orientation);
    if (!std::isfinite(yaw)) return;
    std::lock_guard<std::mutex> lock(pose_mutex_);
    fused_prior_ = {
        static_cast<float>(pose.position.x),
        static_cast<float>(pose.position.y),
        static_cast<float>(yaw)
    };
    prior_stamp_    = msg->header.stamp;
    prior_received_ = true;
}

bool LocalizationNode::getTransformAsArray(
    const std::string& frame_id, const rclcpp::Time& stamp, float out_transform[12]
)
{
    try {
        auto tf                        = tf_buffer_->lookupTransform(base_frame_, frame_id, stamp);
        const Eigen::Affine3d eigen_tf = tf2::transformToEigen(tf);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                out_transform[r * 4 + c] = static_cast<float>(eigen_tf.matrix()(r, c));
            }
        }
        return true;
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
        return false;
    }
}

std::vector<TimedScanPoint> LocalizationNode::extractTimedBasePoints(
    const sensor_msgs::msg::PointCloud2& msg, const float transform[12], int64_t& end_ns
)
{
    const int64_t header_ns                = rclcpp::Time(msg.header.stamp).nanoseconds();
    end_ns                                 = header_ns;
    const sensor_msgs::msg::PointField *xf = nullptr, *yf = nullptr, *zf = nullptr, *tf = nullptr;
    for (const auto& field : msg.fields) {
        if (field.name == "x") xf = &field;
        if (field.name == "y") yf = &field;
        if (field.name == "z") zf = &field;
        if (field.name == timestamp_field_) tf = &field;
    }
    using Field = sensor_msgs::msg::PointField;
    if (!xf || !yf || !zf || xf->datatype != Field::FLOAT32 || yf->datatype != Field::FLOAT32 ||
        zf->datatype != Field::FLOAT32 || xf->offset + 4 > msg.point_step ||
        yf->offset + 4 > msg.point_step || zf->offset + 4 > msg.point_step)
        return {};
    const size_t time_size =
        !tf ? 0
        : tf->datatype == Field::FLOAT64
            ? 8
            : (tf->datatype == Field::FLOAT32 || tf->datatype == Field::UINT32 ? 4 : 0);
    const bool has_time = tf && time_size && tf->offset + time_size <= msg.point_step;
    if (!has_time) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            10000,
            "No usable per-point '%s' timestamp; using cloud stamps for accumulation",
            timestamp_field_.c_str()
        );
    }
    const auto read = [&](const uint8_t* source, auto& value) {
        std::memcpy(&value, source, sizeof(value));
        if (msg.is_bigendian) {
            auto* bytes = reinterpret_cast<uint8_t*>(&value);
            std::reverse(bytes, bytes + sizeof(value));
        }
    };
    std::vector<TimedScanPoint> result;
    result.reserve(size_t(msg.width) * msg.height);
    for (size_t row = 0; row < msg.height; ++row) {
        for (size_t col = 0; col < msg.width; ++col) {
            const size_t offset = row * msg.row_step + col * msg.point_step;
            if (offset + msg.point_step > msg.data.size()) return {};
            const auto* p = msg.data.data() + offset;
            float x, y, z;
            read(p + xf->offset, x);
            read(p + yf->offset, y);
            read(p + zf->offset, z);
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
            int64_t ns = header_ns;
            if (has_time) {
                double timestamp = 0;
                if (tf->datatype == Field::FLOAT64)
                    read(p + tf->offset, timestamp);
                else if (tf->datatype == Field::FLOAT32) {
                    float value;
                    read(p + tf->offset, value);
                    timestamp = value;
                } else {
                    uint32_t value;
                    read(p + tf->offset, value);
                    timestamp = value;
                }
                const long double value_ns =
                    static_cast<long double>(timestamp) * timestamp_scale_ * 1e9L +
                    (timestamp_relative_ ? header_ns : 0);
                // Reject mismatched time units/clocks before conversion to int64.
                if (std::isfinite(timestamp) && value_ns >= header_ns - 5000000 &&
                    value_ns <= header_ns + 500000000)
                    ns = int64_t(std::llround(value_ns));
            }
            end_ns = std::max(end_ns, ns);
            result.push_back(
                {transform[0] * x + transform[1] * y + transform[2] * z + transform[3],
                 transform[4] * x + transform[5] * y + transform[6] * z + transform[7],
                 transform[8] * x + transform[9] * y + transform[10] * z + transform[11],
                 ns}
            );
        }
    }
    return result;
}

void LocalizationNode::updateLostState(bool matched, float best_cost)
{
    if (!matched || std::isnan(best_cost) || best_cost > match_params_.cost_threshold) {
        lost_frame_count_++;
        RCLCPP_WARN(
            this->get_logger(),
            "Matching failed or high cost (cost: %.4f). Lost count: %d/%d",
            best_cost,
            lost_frame_count_,
            lost_threshold_count_
        );
        if (lost_frame_count_ >= lost_threshold_count_) {
            RCLCPP_ERROR(this->get_logger(), "Localization lost! Transitioning to Global Search.");
            is_lost_ = true;
        }
    } else {
        lost_frame_count_ = 0;
    }
}

void LocalizationNode::publishPoseAndTransform(const rclcpp::Time& stamp, const PoseCandidate& pose)
{
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        last_known_pose_ = pose;
        predicted_pose_  = pose;
    }

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, pose.yaw);

    auto pose_msg             = std::make_unique<geometry_msgs::msg::PoseWithCovarianceStamped>();
    pose_msg->header.stamp    = stamp;
    pose_msg->header.frame_id = map_frame_;
    pose_msg->pose.pose.position.x  = pose.x;
    pose_msg->pose.pose.position.y  = pose.y;
    pose_msg->pose.pose.position.z  = 0.0f;
    pose_msg->pose.pose.orientation = tf2::toMsg(q);
    pose_msg->pose.covariance[0]    = 0.005;
    pose_msg->pose.covariance[7]    = 0.005;
    pose_msg->pose.covariance[35]   = 0.002;
    pub_platform_pose_->publish(std::move(pose_msg));

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp            = stamp;
    tf_msg.header.frame_id         = map_frame_;
    tf_msg.child_frame_id          = base_frame_;
    tf_msg.transform.translation.x = pose.x;
    tf_msg.transform.translation.y = pose.y;
    tf_msg.transform.translation.z = 0.0f;
    tf_msg.transform.rotation      = tf2::toMsg(q);
    if (publish_tf_) tf_broadcaster_->sendTransform(tf_msg);
}

void LocalizationNode::publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
    const std_msgs::msg::Header& header,
    const std::vector<float>& pts
)
{
    int count             = static_cast<int>(pts.size() / 3);
    auto out_msg          = std::make_unique<sensor_msgs::msg::PointCloud2>();
    out_msg->header       = header;
    out_msg->height       = 1;
    out_msg->width        = count;
    out_msg->is_dense     = true;
    out_msg->is_bigendian = false;

    sensor_msgs::PointCloud2Modifier modifier(*out_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(count);

    sensor_msgs::PointCloud2Iterator<float> iter_x(*out_msg, "x"), iter_y(*out_msg, "y"),
        iter_z(*out_msg, "z");
    for (int i = 0; i < count; ++i, ++iter_x, ++iter_y, ++iter_z) {
        *iter_x = pts[i * 3 + 0];
        *iter_y = pts[i * 3 + 1];
        *iter_z = pts[i * 3 + 2];
    }
    pub->publish(std::move(out_msg));
}

void LocalizationNode::publishFieldMapMarkers()
{
    visualization_msgs::msg::MarkerArray marker_array;
    int id = 0;
    for (const auto& obj : map_objects_) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = map_frame_;
        marker.header.stamp    = this->now();
        marker.ns              = "field_objects";
        marker.id              = id++;
        marker.action          = visualization_msgs::msg::Marker::ADD;

        marker.color.r = 0.1f;
        marker.color.g = 0.8f;
        marker.color.b = 0.4f;
        marker.color.a = 0.6f;

        const float height        = obj.z_max - obj.z_min;
        marker.pose.position.x    = obj.center_x;
        marker.pose.position.y    = obj.center_y;
        marker.pose.position.z    = obj.z_min + height / 2.0f;
        marker.pose.orientation.w = 1.0;

        if (obj.type == BOX || obj.type == VISUAL_BOX) {
            marker.type    = visualization_msgs::msg::Marker::CUBE;
            marker.scale.x = obj.param1 * 2.0f;
            marker.scale.y = obj.param2 * 2.0f;
            marker.scale.z = height;
        } else if (obj.type == CYLINDER) {
            marker.type    = visualization_msgs::msg::Marker::CYLINDER;
            marker.scale.x = obj.param1 * 2.0f;
            marker.scale.y = obj.param1 * 2.0f;
            marker.scale.z = height;
            marker.color.r = 0.9f;
            marker.color.g = 0.3f;
            marker.color.b = 0.1f;
        }
        marker_array.markers.push_back(marker);
    }
    pub_map_markers_->publish(marker_array);
}

/**
 * @brief 読み込み済みの ESDF マップを RViz2 表示用 PointCloud2 として 1 回だけパブリッシュする
 */
void LocalizationNode::publishESDFMap()
{
    if (!pub_esdf_map_ || esdf_map_.empty()) {
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Extracting ESDF map voxel points for RViz2 visualization...");
    const std::vector<ESDFMap::ESDFVoxelPoint> voxel_points =
        esdf_map_.extractVoxelPoints(esdf_publish_max_distance_m_, esdf_publish_stride_);

    if (voxel_points.empty()) {
        RCLCPP_WARN(this->get_logger(), "No ESDF voxel points found within threshold.");
        return;
    }

    const size_t total_points = voxel_points.size();
    auto out_msg              = std::make_unique<sensor_msgs::msg::PointCloud2>();
    out_msg->header.stamp     = this->now();
    out_msg->header.frame_id  = map_frame_;
    out_msg->height           = 1;
    out_msg->width            = static_cast<uint32_t>(total_points);
    out_msg->is_dense         = true;
    out_msg->is_bigendian     = false;

    // フィールドの設定: x, y, z, intensity (距離を格納)
    sensor_msgs::PointCloud2Modifier modifier(*out_msg);
    modifier.setPointCloud2Fields(
        4,
        "x",
        1,
        sensor_msgs::msg::PointField::FLOAT32,
        "y",
        1,
        sensor_msgs::msg::PointField::FLOAT32,
        "z",
        1,
        sensor_msgs::msg::PointField::FLOAT32,
        "intensity",
        1,
        sensor_msgs::msg::PointField::FLOAT32
    );
    modifier.resize(total_points);

    // 点群データの書き込み (intensity フィールドに距離[m]を代入)
    sensor_msgs::PointCloud2Iterator<float> iter_x(*out_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*out_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*out_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_intensity(*out_msg, "intensity");

    for (size_t point_index = 0; point_index < total_points;
         ++point_index, ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
        const auto& point_data = voxel_points[point_index];
        *iter_x                = point_data.x;
        *iter_y                = point_data.y;
        *iter_z                = point_data.z;
        *iter_intensity        = point_data.distance_m;
    }

    // 1 回だけパブリッシュ (QoS transient_local により後から起動した RViz2 にも自動配信)
    pub_esdf_map_->publish(std::move(out_msg));
    RCLCPP_INFO(
        this->get_logger(),
        "Published ESDF map to '%s' (%zu points, transient_local QoS)",
        topic_esdf_map_.c_str(),
        total_points
    );
}

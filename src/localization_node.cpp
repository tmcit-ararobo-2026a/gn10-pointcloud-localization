#include "gn10_pointcloud_localization/localization_node.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_ros/create_timer_ros.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include "gn10_pointcloud_localization/scan_points.hpp"
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
}

void LocalizationNode::declareAndGetParameters()
{
    this->declare_parameter("map_source_type", "json");
    this->declare_parameter("map_file_path", "");
    this->declare_parameter("map_objects", std::vector<std::string>{});

    this->declare_parameter("frames.map_frame", "map");
    this->declare_parameter("frames.base_frame", "base_link");

    this->declare_parameter("topics.input_cloud", "/livox/lidar");
    rcl_interfaces::msg::ParameterDescriptor input_type_descriptor;
    input_type_descriptor.read_only = true;
    input_type_descriptor.description = "LiDAR message type selected at startup: pointcloud2 or custom_msg";
    this->declare_parameter("input_cloud_type", "pointcloud2", input_type_descriptor);
    this->declare_parameter("topics.input_imu", "/livox/imu");
    this->declare_parameter("topics.output_dynamic", "/dynamic_cloud");
    this->declare_parameter("topics.output_ground", "/ground_cloud");
    this->declare_parameter("topics.output_obstacle", "/obstacle_cloud");
    this->declare_parameter("topics.output_pose", "/platform_constraint");
    this->declare_parameter("topics.output_markers", "/field_map_markers");
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

    this->declare_parameter("matching_params.search_range_xy", 0.30);
    this->declare_parameter("matching_params.search_step_xy", 0.03);
    this->declare_parameter("matching_params.search_range_yaw", 0.15);
    this->declare_parameter("matching_params.search_step_yaw", 0.02);
    this->declare_parameter("matching_params.fine_refine", true);
    this->declare_parameter("matching_params.robust_local", true);
    this->declare_parameter("matching_params.robust_distance", 0.08);
    this->declare_parameter("matching_params.robust_cost_threshold", 0.045);
    this->declare_parameter("matching_params.min_support_ratio", 0.10);
    this->declare_parameter("matching_params.min_support_count", 100);
    this->declare_parameter("matching_params.min_support_sectors", 3);
    this->declare_parameter("matching_params.min_axis_support", 10);
    this->declare_parameter("motion.use_odom", false);
    this->declare_parameter("motion.scan_wait_s", 2.0);
    this->declare_parameter("motion.scan_queue_size", 30);
    this->declare_parameter("topics.motion_odom", "/gn10/odom_base");
    this->declare_parameter("matching_params.max_dist_thresh", 0.20);
    this->declare_parameter("matching_params.cost_threshold", 0.165);
    this->declare_parameter("matching_params.dynamic_dist_thresh", 0.15);
    this->declare_parameter("matching_params.field_min_x", -5.5);
    this->declare_parameter("matching_params.field_max_x", 5.5);
    this->declare_parameter("matching_params.field_min_y", -6.0);
    this->declare_parameter("matching_params.field_max_y", 6.0);

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
    use_motion_ = get_parameter("motion.use_odom").as_bool();
    scan_wait_s_ = get_parameter("motion.scan_wait_s").as_double();
    const auto queue_size = get_parameter("motion.scan_queue_size").as_int();
    if (!std::isfinite(scan_wait_s_) || scan_wait_s_ <= 0 || scan_wait_s_ > 4.0 ||
        queue_size < 1 || queue_size > 100)
        throw std::invalid_argument("Invalid scan wait/queue parameters");
    scan_queue_size_ = static_cast<size_t>(queue_size);
    match_params_.robust_local = get_parameter("matching_params.robust_local").as_bool();
    match_params_.robust_distance = get_parameter("matching_params.robust_distance").as_double();
    match_params_.robust_cost_threshold = get_parameter("matching_params.robust_cost_threshold").as_double();
    match_params_.min_support_ratio = get_parameter("matching_params.min_support_ratio").as_double();
    match_params_.min_support_count = get_parameter("matching_params.min_support_count").as_int();
    match_params_.min_axis_support = get_parameter("matching_params.min_axis_support").as_int();
    match_params_.min_support_sectors = get_parameter("matching_params.min_support_sectors").as_int();
    match_params_.fine_refine = this->get_parameter("matching_params.fine_refine").as_bool();
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

    if (match_params_.robust_distance <= 0 || match_params_.robust_distance >= match_params_.max_dist_thresh ||
        match_params_.robust_cost_threshold <= 0 || match_params_.robust_cost_threshold >= match_params_.robust_distance ||
        match_params_.min_axis_support < 1 || match_params_.step_xy <= 0 || match_params_.step_yaw <= 0 ||
        match_params_.range_xy < 0 || match_params_.range_yaw < 0 || match_params_.min_support_count < 1 || match_params_.min_support_ratio <= 0 ||
        match_params_.min_support_ratio > 1 || match_params_.min_support_sectors < 1 || match_params_.min_support_sectors > 8)
        throw std::invalid_argument("Invalid robust matching parameters");

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
    publish_tf_ = this->get_parameter("publish_tf").as_bool();
    use_fused_prior_ = this->get_parameter("fusion.use_prior").as_bool();
    prior_max_age_s_ = this->get_parameter("fusion.prior_max_age_s").as_double();

    last_known_pose_.x   = static_cast<float>(this->get_parameter("initial_pose.x").as_double());
    last_known_pose_.y   = static_cast<float>(this->get_parameter("initial_pose.y").as_double());
    last_known_pose_.yaw = static_cast<float>(this->get_parameter("initial_pose.yaw").as_double());
    predicted_pose_      = last_known_pose_;
    // A known start pose resolves the field's near-symmetric global matches.
    // Do not publish it as a measurement; first require a successful cloud match.
    is_lost_ = !this->get_parameter("initial_pose.use_for_local_search").as_bool();
}

void LocalizationNode::setupMapData()
{
    std::string map_source = this->get_parameter("map_source_type").as_string();
    if (map_source == "json") {
        std::string json_path = this->get_parameter("map_file_path").as_string();
        if (json_path.empty()) {
            json_path =
                ament_index_cpp::get_package_share_directory("gn10_pointcloud_localization") +
                "/config/nhk2026_map.json";
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

    solver_->setMap(map_objects_);
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
    const auto input_type = get_parameter("input_cloud_type").as_string();
    if (input_type == "custom_msg") {
        sub_custom_cloud_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
            topic_cloud, rclcpp::SensorDataQoS().keep_last(50),
            std::bind(&LocalizationNode::customCloudCallback, this, std::placeholders::_1));
    } else if (input_type == "pointcloud2") {
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
    } else {
        throw std::invalid_argument("input_cloud_type must be pointcloud2 or custom_msg");
    }
    RCLCPP_INFO(get_logger(), "LiDAR input: %s (%s)", topic_cloud.c_str(), input_type.c_str());

    std::string topic_imu = this->get_parameter("topics.input_imu").as_string();
    sub_imu_              = this->create_subscription<sensor_msgs::msg::Imu>(
        topic_imu,
        rclcpp::SensorDataQoS(),
        std::bind(&LocalizationNode::imuCallback, this, std::placeholders::_1)
    );
    if (use_fused_prior_) {
        sub_fused_prior_ =
            this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
                this->get_parameter("topics.fused_prior").as_string(), 20,
                std::bind(&LocalizationNode::fusedPriorCallback, this, std::placeholders::_1)
            );
    }

    pub_match_diagnostics_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "/gn10_matcher/diagnostics", 10);
    if (use_motion_) {
        sub_motion_ = create_subscription<nav_msgs::msg::Odometry>(
            get_parameter("topics.motion_odom").as_string(), 50,
            std::bind(&LocalizationNode::odometryCallback, this, std::placeholders::_1));
    }
    motion_timer_ = create_wall_timer(std::chrono::milliseconds(10),
        std::bind(&LocalizationNode::drainClouds, this));
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

    using namespace std::chrono_literals;
    map_timer_ =
        this->create_wall_timer(1s, std::bind(&LocalizationNode::publishFieldMapMarkers, this));
}

void LocalizationNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    enqueueScan(msg->header, gn10::decodeScan(*msg));
}

void LocalizationNode::customCloudCallback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
{
    enqueueScan(msg->header, gn10::decodeScan(*msg));
}

void LocalizationNode::enqueueScan(const std_msgs::msg::Header& header, gn10::ScanPoints scan)
{
    if (!scan.valid || (use_motion_ && !scan.timed)) {
        ++timing_drops_;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "Dropping invalid scan (motion mode requires consistent per-point timestamps)");
        return;
    }
    if (pending_clouds_.size() >= scan_queue_size_) { pending_clouds_.pop_front(); ++timing_drops_; }
    pending_clouds_.push_back({header, std::chrono::steady_clock::now(), std::move(scan)});
}

void LocalizationNode::drainClouds()
{
    while (!pending_clouds_.empty()) {
        const auto& scan = pending_clouds_.front().scan;
        const auto& header = pending_clouds_.front().header;
        const bool tf_ready = tf_buffer_->canTransform(base_frame_, header.frame_id, rclcpp::Time(header.stamp));
        const bool motion_ready = !use_motion_ || (motion_.at(scan.start) && motion_.at(scan.end));
        const bool known_motion_gap = use_motion_ && !motion_ready && !motion_.samples.empty() &&
            (scan.start < motion_.samples.front().stamp - 1e-4 ||
             scan.end <= motion_.samples.back().stamp + 1e-4);
        if (tf_ready && motion_ready) {
            auto pending = std::move(pending_clouds_.front());
            pending_clouds_.pop_front();
            processScan(pending.header, std::move(pending.scan));
            return; // Let odometry and fused-prior callbacks run between scans.
        } else if (known_motion_gap ||
            std::chrono::steady_clock::now() - pending_clouds_.front().received > std::chrono::duration<double>(scan_wait_s_)) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Dropping scan: TF=%d odom_start=%d odom_end=%d scan=%.6f..%.6f history=%.6f..%.6f",
                tf_ready, static_cast<bool>(motion_.at(scan.start)), static_cast<bool>(motion_.at(scan.end)),
                scan.start, scan.end, motion_.samples.empty() ? 0.0 : motion_.samples.front().stamp,
                motion_.samples.empty() ? 0.0 : motion_.samples.back().stamp);
            pending_clouds_.pop_front(); ++timing_drops_;
        } else break;
    }
}

void LocalizationNode::odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    if (msg->child_frame_id != base_frame_ || msg->header.frame_id.empty()) return;
    const auto& p = msg->pose.pose;
    const auto& q = p.orientation;
    const double norm = q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
    if (!std::isfinite(norm) || std::abs(norm-1.0)>0.01) return;
    const double stamp = rclcpp::Time(msg->header.stamp).seconds();
    if ((!motion_.samples.empty() && stamp < motion_.samples.back().stamp - 0.5) ||
        (!motion_frame_.empty() && motion_frame_ != msg->header.frame_id)) {
        motion_.samples.clear(); prior_history_.clear(); pending_clouds_.clear();
        prior_received_ = false; last_cloud_stamp_ = -1.0;
    }
    if (!motion_.samples.empty() && stamp > motion_.samples.back().stamp) {
        const auto& prev=motion_.samples.back();
        if (std::hypot(p.position.x-prev.x,p.position.y-prev.y)>2.0 ||
            std::abs(gn10::angleDifference(tf2::getYaw(q),prev.yaw))>1.5) {
            motion_.samples.clear(); prior_history_.clear();
        }
    }
    motion_frame_ = msg->header.frame_id;
    motion_.add({stamp,p.position.x,p.position.y,tf2::getYaw(q)});
}

void LocalizationNode::processScan(const std_msgs::msg::Header& header, gn10::ScanPoints scan)
{
    const auto processing_started = std::chrono::steady_clock::now();
    float h_transform[12];
    if (!getTransformAsArray(header.frame_id, header.stamp, h_transform)) {
        return;
    }

    if (!scan.valid) return;
    const double stamp = use_motion_ ? scan.end : scan.start;
    if (stamp <= last_cloud_stamp_) return; // duplicates/out-of-order clouds cannot rewind state
    std::vector<float> h_raw_cloud = std::move(scan.xyz);
    if (use_motion_) {
        const auto end_pose = motion_.at(stamp);
        if (!end_pose) return;
        // Convert LiDAR -> base at each acquisition time, then deskew to scan end.
        for (size_t i=0; i<scan.times.size(); ++i) {
            const auto at = motion_.at(scan.times[i]);
            if (!at) return; // gap in motion history: do not manufacture a measurement
            const float x=h_raw_cloud[3*i],y=h_raw_cloud[3*i+1],z=h_raw_cloud[3*i+2];
            float bx=h_transform[0]*x+h_transform[1]*y+h_transform[2]*z+h_transform[3];
            float by=h_transform[4]*x+h_transform[5]*y+h_transform[6]*z+h_transform[7];
            float bz=h_transform[8]*x+h_transform[9]*y+h_transform[10]*z+h_transform[11];
            gn10::MotionHistory::transformPoint(*at,*end_pose,bx,by);
            h_raw_cloud[3*i]=bx; h_raw_cloud[3*i+1]=by; h_raw_cloud[3*i+2]=bz;
        }
        const float identity[12]={1,0,0,0,0,1,0,0,0,0,1,0};
        std::copy(identity,identity+12,h_transform);
    }
    last_cloud_stamp_ = stamp;
    const rclcpp::Time match_stamp(static_cast<int64_t>(std::llround(stamp * 1e9)));
    std::vector<float> ground_pts, obstacle_pts, dynamic_pts;
    PoseCandidate best_pose;
    float best_cost = std::numeric_limits<float>::max();
    bool matched    = false;
    bool attempted = false;

    PoseCandidate search_base_pose;
    bool has_fresh_prior = false;
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        search_base_pose = predicted_pose_;
        if (!use_fused_prior_ && !imu_history_.empty()) {
            const auto hi=std::upper_bound(imu_history_.begin(),imu_history_.end(),stamp,
                [](double t,const auto& p){return t<p.stamp;});
            if (hi!=imu_history_.begin()) {
                const auto lo=std::prev(hi);
                if (stamp-lo->stamp<=0.1) search_base_pose.yaw = last_known_pose_.yaw +
                    lo->yaw - match_integrated_yaw_;
            }
        }
        if (use_fused_prior_) {
            // Use a past map prior propagated by continuous odometry to the exact scan end.
            for (auto it=prior_history_.rbegin();it!=prior_history_.rend();++it) {
                if (it->stamp > stamp + 1e-6) continue;
                if (stamp-it->stamp > prior_max_age_s_) break;
                auto from=motion_.at(it->stamp),to=motion_.at(stamp);
                if (use_motion_ && from && to) {
                    const auto p=gn10::MotionHistory::advance(*it,*from,*to);
                    search_base_pose={static_cast<float>(p.x),static_cast<float>(p.y),static_cast<float>(p.yaw)};
                    has_fresh_prior=true;
                } else if (!use_motion_) {
                    search_base_pose={static_cast<float>(it->x),static_cast<float>(it->y),static_cast<float>(it->yaw)};
                    has_fresh_prior=true;
                }
                break;
            }
        }
    }

    const float local_threshold = match_params_.robust_local ?
        match_params_.robust_cost_threshold : match_params_.cost_threshold;
    float acceptance_threshold = local_threshold;
    // A recent fused pose can reacquire locally even after the map matcher was lost.
    if (is_lost_ && has_fresh_prior) {
        attempted = true;
        matched = solver_->processPointCloud(
            h_raw_cloud, h_transform, filter_params_, match_params_, search_base_pose,
            dynamic_pts, best_pose, best_cost
        );
        if (matched && std::isfinite(best_cost) &&
            best_cost < local_threshold) {
            is_lost_ = false;
            lost_frame_count_ = 0;
        } else {
            matched = false;
            dynamic_pts.clear();
            ++lost_frame_count_;
        }
    }

    if (is_lost_ && !matched && !has_fresh_prior && !(use_fused_prior_ && prior_received_)) {
        acceptance_threshold = match_params_.cost_threshold;
        float current_prior_yaw = 0.0f;
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            current_prior_yaw = search_base_pose.yaw;
        }

        attempted = true;
        best_pose = global_searcher_->search(
            *solver_,
            h_raw_cloud,
            h_transform,
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
            RCLCPP_INFO(
                this->get_logger(), "[GlobalSearch] Successfully recovered from lost state."
            );
        }
    } else if (!matched && !is_lost_ && (!use_fused_prior_ || !prior_received_ || has_fresh_prior)) {
        attempted = true;
        matched = solver_->processPointCloud(
            h_raw_cloud,
            h_transform,
            filter_params_,
            match_params_,
            search_base_pose,
            dynamic_pts,
            best_pose,
            best_cost
        );

        updateLostState(matched, best_cost, local_threshold);
        matched = matched && std::isfinite(best_cost) && best_cost < local_threshold;
    }

    const bool accepted = matched && std::isfinite(best_cost) && best_cost < acceptance_threshold;
    if (accepted) ++match_accepted_; else ++match_rejected_;
    diagnostic_msgs::msg::DiagnosticArray diagnostics;
    diagnostics.header.stamp=match_stamp;
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name="gn10_matcher"; status.hardware_id="field_sdf";
    status.level=accepted ? status.OK : status.WARN;
    status.message=accepted ? "Map support accepted" : "Insufficient map support; no pose measurement";
    const auto quality=attempted ? solver_->lastMatchStats() : FieldMatchStats{};
    auto value=[&](const std::string& key, auto v) {
        diagnostic_msgs::msg::KeyValue kv; kv.key=key;kv.value=std::to_string(v);status.values.push_back(kv);
    };
    value("match_attempted",attempted);
    value("matches_accepted",match_accepted_);value("matches_rejected",match_rejected_);
    value("cost",best_cost);value("threshold",acceptance_threshold);
    value("support_count",quality.support_count);value("support_ratio",quality.support_ratio);
    value("support_sectors",quality.sectors);
    value("axis_x_support",quality.axis_x);value("axis_y_support",quality.axis_y);value("has_motion_prior",has_fresh_prior);
    value("processing_ms",std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-processing_started).count());
    value("global_recovery_locked",use_fused_prior_ && prior_received_);
    value("timing_drops",timing_drops_);value("deskewed",use_motion_);
    diagnostics.status.push_back(status);pub_match_diagnostics_->publish(diagnostics);
    if (accepted) {
        publishPoseAndTransform(match_stamp, best_pose);
    }

    std_msgs::msg::Header out_header = header;
    out_header.frame_id              = base_frame_;
    out_header.stamp = match_stamp;
    const bool need_ground = pub_ground_->get_subscription_count() > 0;
    const bool need_obstacle = pub_obstacle_->get_subscription_count() > 0;
    if (need_ground || need_obstacle) {
        solver_->copyFilteredClouds(
            need_ground ? &ground_pts : nullptr,
            need_obstacle ? &obstacle_pts : nullptr
        );
        if (need_ground) publishCloud(pub_ground_, out_header, ground_pts);
        if (need_obstacle) publishCloud(pub_obstacle_, out_header, obstacle_pts);
    }
    publishCloud(pub_dynamic_, out_header, dynamic_pts);
}

void LocalizationNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    if (use_fused_prior_ || use_motion_) return;
    if (imu_frame_ != msg->header.frame_id) {
        try {
            geometry_msgs::msg::TransformStamped transform_stamped =
                tf_buffer_->lookupTransform(base_frame_, msg->header.frame_id, tf2::TimePointZero);
            const Eigen::Affine3d eigen_tf = tf2::transformToEigen(transform_stamped);
            imu_rotation_ = eigen_tf.rotation();
            imu_frame_ = msg->header.frame_id;
            imu_initialized_ = false;
        } catch (const tf2::TransformException&) {
            return;
        }
    }

    if (!imu_initialized_) {
        last_imu_stamp_  = msg->header.stamp;
        imu_initialized_ = true;
        return;
    }

    const double dt = (rclcpp::Time(msg->header.stamp) - last_imu_stamp_).seconds();
    if (dt <= 0.0) return;
    last_imu_stamp_ = msg->header.stamp;
    if (dt > 0.5) return;

    const Eigen::Vector3d omega_imu(
        msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z
    );
    const Eigen::Vector3d omega_base = imu_rotation_ * omega_imu;
    if (!omega_base.allFinite()) return;
    integrated_yaw_ += omega_base.z() * dt;
    imu_history_.push_back({rclcpp::Time(msg->header.stamp).seconds(),integrated_yaw_});
    while (imu_history_.size()>2 && imu_history_.back().stamp-imu_history_.front().stamp>2.0)
        imu_history_.pop_front();
}

void LocalizationNode::fusedPriorCallback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg
)
{
    if (msg->header.frame_id != map_frame_) return;
    const auto& pose = msg->pose.pose;
    if (!std::isfinite(pose.position.x) || !std::isfinite(pose.position.y)) return;
    const auto& q=pose.orientation;
    const double norm=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
    if (!std::isfinite(norm) || std::abs(norm-1.0)>0.01) return;
    const double yaw = tf2::getYaw(q);
    if (!std::isfinite(yaw)) return;
    std::lock_guard<std::mutex> lock(pose_mutex_);
    fused_prior_ = {static_cast<float>(pose.position.x),
                    static_cast<float>(pose.position.y), static_cast<float>(yaw)};
    prior_stamp_ = msg->header.stamp;
    prior_received_ = true;
    const double t=prior_stamp_.seconds();
    if (!prior_history_.empty() && std::abs(t-prior_history_.back().stamp) < 1e-6) {
        prior_history_.back() = {t,pose.position.x,pose.position.y,yaw};
    } else if (prior_history_.empty() || t>prior_history_.back().stamp) {
        prior_history_.push_back({t,pose.position.x,pose.position.y,yaw});
        while(prior_history_.size()>2 && t-prior_history_.front().stamp>5.0) prior_history_.pop_front();
    }
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

void LocalizationNode::updateLostState(bool matched, float best_cost, float threshold)
{
    if (!matched || std::isnan(best_cost) || best_cost > threshold) {
        lost_frame_count_++;
        RCLCPP_WARN(
            this->get_logger(),
            "Matching failed or high cost (cost: %.4f). Lost count: %d/%d",
            best_cost,
            lost_frame_count_,
            lost_threshold_count_
        );
        if (lost_frame_count_ >= lost_threshold_count_) {
            RCLCPP_ERROR(this->get_logger(), "%s", use_fused_prior_ && prior_received_ ?
                "Localization lost; continuing local matching with internal prior" :
                "Localization lost! Transitioning to Global Search.");
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
        const auto hi=std::upper_bound(imu_history_.begin(),imu_history_.end(),stamp.seconds(),
            [](double t,const auto& p){return t<p.stamp;});
        if (hi!=imu_history_.begin()) match_integrated_yaw_=std::prev(hi)->yaw;
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

        const float height     = obj.z_max - obj.z_min;
        marker.pose.position.x = obj.center_x;
        marker.pose.position.y = obj.center_y;
        marker.pose.position.z = obj.z_min + height / 2.0f;
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

#include "gn10_pointcloud_localization/localization_node.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/create_timer_ros.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

LocalizationNode::LocalizationNode() : Node("gn10_localization_node")
{
    declareAndGetParameters();

    solver_ =
        std::make_unique<PoseSolver>(this->get_parameter("filter_params.max_points").as_int());

    // --- Map Data Loading ---
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

    // --- TF & MessageFilters ---
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
        rclcpp::SensorDataQoS(),
        std::bind(&LocalizationNode::imuCallback, this, std::placeholders::_1)
    );

    // --- Publishers ---
    pub_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        this->get_parameter("topics.output_ground").as_string(), 10
    );
    pub_obstacle_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        this->get_parameter("topics.output_obstacle").as_string(), 10
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

void LocalizationNode::declareAndGetParameters()
{
    this->declare_parameter("map_source_type", "json");
    this->declare_parameter("map_file_path", "");
    this->declare_parameter("map_objects", std::vector<std::string>{});

    this->declare_parameter("frames.map_frame", "map");
    this->declare_parameter("frames.base_frame", "base_link");

    this->declare_parameter("topics.input_cloud", "/livox/lidar");
    this->declare_parameter("topics.input_imu", "/livox/imu");
    this->declare_parameter("topics.output_ground", "/ground_cloud");
    this->declare_parameter("topics.output_obstacle", "/obstacle_cloud");
    this->declare_parameter("topics.output_pose", "/platform_constraint");
    this->declare_parameter("topics.output_markers", "/field_map_markers");

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
    this->declare_parameter("matching_params.max_dist_thresh", 0.20);
    this->declare_parameter("matching_params.cost_threshold", 0.20);

    this->declare_parameter("initial_pose.x", -4.0);
    this->declare_parameter("initial_pose.y", -4.0);
    this->declare_parameter("initial_pose.yaw", -1.5708);

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
    match_params_.max_dist_thresh =
        static_cast<float>(this->get_parameter("matching_params.max_dist_thresh").as_double());
    match_params_.cost_threshold =
        static_cast<float>(this->get_parameter("matching_params.cost_threshold").as_double());

    last_known_pose_.x   = static_cast<float>(this->get_parameter("initial_pose.x").as_double());
    last_known_pose_.y   = static_cast<float>(this->get_parameter("initial_pose.y").as_double());
    last_known_pose_.yaw = static_cast<float>(this->get_parameter("initial_pose.yaw").as_double());
    predicted_pose_      = last_known_pose_;
}

void LocalizationNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    geometry_msgs::msg::TransformStamped transform_stamped;
    try {
        transform_stamped =
            tf_buffer_->lookupTransform(base_frame_, msg->header.frame_id, msg->header.stamp);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
        return;
    }

    const Eigen::Affine3d eigen_tf = tf2::transformToEigen(transform_stamped);
    float h_transform[12];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            h_transform[r * 4 + c] = static_cast<float>(eigen_tf.matrix()(r, c));
        }
    }

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x"), iter_y(*msg, "y"),
        iter_z(*msg, "z");
    std::vector<float> h_raw_cloud;
    h_raw_cloud.reserve(msg->width * msg->height * 3);

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        if (std::isnan(*iter_x)) continue;
        h_raw_cloud.push_back(*iter_x);
        h_raw_cloud.push_back(*iter_y);
        h_raw_cloud.push_back(*iter_z);
    }

    PoseCandidate search_base_pose;
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        search_base_pose = predicted_pose_;
    }

    std::vector<float> ground_pts, obstacle_pts;
    PoseCandidate best_pose;
    float best_cost = 0.0f;

    bool matched = solver_->processPointCloud(
        h_raw_cloud,
        h_transform,
        filter_params_,
        match_params_,
        search_base_pose,
        ground_pts,
        obstacle_pts,
        best_pose,
        best_cost
    );

    if (matched && best_cost < match_params_.cost_threshold) {
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            last_known_pose_ = best_pose;
            predicted_pose_  = best_pose;
        }

        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, best_pose.yaw);

        auto pose_msg          = std::make_unique<geometry_msgs::msg::PoseWithCovarianceStamped>();
        pose_msg->header.stamp = msg->header.stamp;
        pose_msg->header.frame_id       = map_frame_;
        pose_msg->pose.pose.position.x  = best_pose.x;
        pose_msg->pose.pose.position.y  = best_pose.y;
        pose_msg->pose.pose.position.z  = 0.0f;
        pose_msg->pose.pose.orientation = tf2::toMsg(q);
        pose_msg->pose.covariance[0]    = 0.005;
        pose_msg->pose.covariance[7]    = 0.005;
        pose_msg->pose.covariance[35]   = 0.002;
        pub_platform_pose_->publish(std::move(pose_msg));

        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp            = msg->header.stamp;
        tf_msg.header.frame_id         = map_frame_;
        tf_msg.child_frame_id          = base_frame_;
        tf_msg.transform.translation.x = best_pose.x;
        tf_msg.transform.translation.y = best_pose.y;
        tf_msg.transform.translation.z = 0.0f;
        tf_msg.transform.rotation      = tf2::toMsg(q);
        tf_broadcaster_->sendTransform(tf_msg);
    }

    std_msgs::msg::Header out_header = msg->header;
    out_header.frame_id              = base_frame_;
    publishCloud(pub_ground_, out_header, ground_pts);
    publishCloud(pub_obstacle_, out_header, obstacle_pts);
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

    if (!imu_initialized_) {
        last_imu_stamp_  = msg->header.stamp;
        imu_initialized_ = true;
        return;
    }

    const double dt = (rclcpp::Time(msg->header.stamp) - last_imu_stamp_).seconds();
    last_imu_stamp_ = msg->header.stamp;

    if (dt <= 0.0 || dt > 0.5) {
        imu_initialized_ = false;
        return;
    }

    const Eigen::Vector3d omega_imu(
        msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z
    );
    const Eigen::Vector3d omega_base = R_base_imu * omega_imu;

    predicted_pose_.yaw += static_cast<float>(omega_base.z() * dt);
    predicted_pose_.yaw = std::atan2(std::sin(predicted_pose_.yaw), std::cos(predicted_pose_.yaw));
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

        if (obj.type == BOX) {
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
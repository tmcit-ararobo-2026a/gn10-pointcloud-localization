#include "gn10_pointcloud_localization/localization_node.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/create_timer_ros.h>

#include <cmath>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <vector>

#include "gn10_pointcloud_localization/cuda/ground_filter.cuh"

// CUDA 関数の前方宣言
void uploadFieldMapToGPU(const std::vector<FieldObject>& host_map);
bool launchFieldSDFMatcher(
    const float* d_obstacle_cloud,
    int num_points,
    const PoseCandidate& base_pose,
    float range_xy,
    float step_xy,
    float range_yaw,
    float step_yaw,
    float max_dist_thresh,
    PoseCandidate& out_best_pose,
    float& out_best_cost
);

// NHK2026 フィールドマップ構築関数
std::vector<FieldObject> createNHK2026FieldMap()
{
    std::vector<FieldObject> map;

    // 1. 外壁 (10.5m x 11.4m, H=0.15m) -> X: [-5.25, 5.25], Y: [-5.7, 5.7]
    map.push_back({BOX, 0.000f, 5.700f, 0.000f, 0.150f, 5.250f, 0.050f});   // 上壁
    map.push_back({BOX, 0.000f, -5.700f, 0.000f, 0.150f, 5.250f, 0.050f});  // 下壁
    map.push_back({BOX, -5.250f, 0.000f, 0.000f, 0.150f, 0.050f, 5.700f});  // 左壁
    map.push_back({BOX, 5.250f, 0.000f, 0.000f, 0.150f, 0.050f, 5.700f});   // 右壁

    // 2. 教壇 (X: -5.25~5.25m, Y: -0.3~0.3m, H: 0.2m)
    map.push_back({BOX, 0.000f, 0.000f, 0.000f, 0.200f, 5.250f, 0.300f});

    struct ObjectSpec {
        ObjectType type;
        float x, y;
        float param1, param2;  // CYL: radius, 0 / BOX: half_w, half_d
        float z_min, z_max;
    };

    constexpr float bucket_radius = 0.273f / 2.0f;  // バケツ半径 0.1365m
    std::vector<ObjectSpec> base_specs;

    // バケツ① (φ0.273 x H0.255)
    base_specs.push_back({CYLINDER, 0.550f, 0.870f, bucket_radius, 0.000f, 0.000f, 0.255f});

    // バケツ② (台座 0.3x0.3xH0.6 + バケツ①)
    base_specs.push_back({BOX, -1.270f, 1.480f, 0.150f, 0.150f, 0.000f, 0.600f});
    base_specs.push_back({CYLINDER, -1.270f, 1.480f, bucket_radius, 0.000f, 0.600f, 0.855f});

    // バケツ③ (台座 0.3x0.3xH0.3 + バケツ①)
    base_specs.push_back({BOX, 2.370f, 1.480f, 0.150f, 0.150f, 0.000f, 0.300f});
    base_specs.push_back({CYLINDER, 2.370f, 1.480f, bucket_radius, 0.000f, 0.300f, 0.555f});

    // 椅子 (W0.36 x D0.40, H0.807)
    base_specs.push_back({BOX, 0.550f, 4.980f, 0.180f, 0.200f, 0.000f, 0.807f});

    // 机 (W0.65 x D0.45 x H0.76) - 4台
    constexpr float desk_coords[4][2] = {
        {-2.295f, 3.855f},
        { 3.395f, 3.855f},
        {-4.895f, 5.445f},
        {-4.750f, 1.105f}
    };
    for (const auto& coord : desk_coords) {
        base_specs.push_back({BOX, coord[0], coord[1], 0.325f, 0.225f, 0.000f, 0.760f});
    }

    // 旗 (土台 0.39x0.39xH0.18 + 支柱 φ0.06 x H3.0)
    base_specs.push_back({BOX, 0.550f, 3.025f, 0.195f, 0.195f, 0.000f, 0.180f});
    base_specs.push_back({CYLINDER, 0.550f, 3.025f, 0.030f, 0.000f, 0.180f, 3.000f});

    // 領域A (+Y) と 領域B (-Y) に対称展開
    for (const auto& spec : base_specs) {
        for (const float y_sign : {1.0f, -1.0f}) {
            map.push_back(
                {spec.type,
                 spec.x,
                 spec.y * y_sign,
                 spec.z_min,
                 spec.z_max,
                 spec.param1,
                 spec.param2}
            );
        }
    }

    return map;
}

LocalizationNode::LocalizationNode() : Node("gn10_localization_node"), max_points_(200000)
{
    // 1. TF 関連の初期化
    auto clock = this->get_clock();
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(clock);
    // タイマーインターフェースを登録する
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(), this->get_node_timers_interface()
    );
    tf_buffer_->setCreateTimerInterface(timer_interface);
    tf_listener_    = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // 2. PointCloud2 Subscriber (MessageFilter 経由)
    sub_cloud_filter_.subscribe(this, "/livox/lidar", rmw_qos_profile_sensor_data);
    tf_filter_ = std::make_shared<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>>(
        sub_cloud_filter_,
        *tf_buffer_,
        "base_link",
        10,
        this->get_node_logging_interface(),
        this->get_node_clock_interface(),
        std::chrono::milliseconds(100)
    );
    tf_filter_->registerCallback(&LocalizationNode::cloudCallback, this);

    // 3. IMU Subscriber
    sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/livox/imu",
        rclcpp::SensorDataQoS(),
        std::bind(&LocalizationNode::imuCallback, this, std::placeholders::_1)
    );

    // 4. Publishers
    pub_ground_   = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_cloud", 10);
    pub_obstacle_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/obstacle_cloud", 10);
    pub_platform_pose_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/platform_constraint", 10
    );
    pub_map_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/field_map_markers", rclcpp::QoS(1).transient_local()
    );

    // 5. CUDA デバイスメモリの確保
    cudaMalloc(&d_in_, max_points_ * 3 * sizeof(float));
    cudaMalloc(&d_ground_, max_points_ * 3 * sizeof(float));
    cudaMalloc(&d_obstacle_, max_points_ * 3 * sizeof(float));
    cudaMalloc(&d_ground_count_, sizeof(int));
    cudaMalloc(&d_obstacle_count_, sizeof(int));
    cudaMalloc(&d_transform_, 12 * sizeof(float));

    // 6. ホスト側 Pinned Memory の確保（転送速度の最適化）
    cudaMallocHost(&h_in_, max_points_ * 3 * sizeof(float));
    cudaMallocHost(&h_out_ground_, max_points_ * 3 * sizeof(float));
    cudaMallocHost(&h_out_obstacle_, max_points_ * 3 * sizeof(float));

    // 7. 初期姿勢およびマップの初期化
    last_known_pose_ = {-4.0f, -4.0f, -1.5708f};
    predicted_pose_  = last_known_pose_;
    is_initialized_  = false;

    map_objects_ = createNHK2026FieldMap();
    uploadFieldMapToGPU(map_objects_);
    publishFieldMapMarkers();

    // 8. 1秒周期で静的マップマーカーを配信するタイマーの設定
    using namespace std::chrono_literals;
    map_timer_ =
        this->create_wall_timer(1s, std::bind(&LocalizationNode::publishFieldMapMarkers, this));
}

LocalizationNode::~LocalizationNode()
{
    cudaFree(d_in_);
    cudaFree(d_ground_);
    cudaFree(d_obstacle_);
    cudaFree(d_ground_count_);
    cudaFree(d_obstacle_count_);
    cudaFree(d_transform_);

    cudaFreeHost(h_in_);
    cudaFreeHost(h_out_ground_);
    cudaFreeHost(h_out_obstacle_);
}

void LocalizationNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    const int num_points = msg->width * msg->height;
    if (num_points == 0 || num_points > max_points_) return;

    // MessageFilter 経由で呼び出されるため、ブロッキングなしで TF 取得可能
    geometry_msgs::msg::TransformStamped transform_stamped;
    try {
        transform_stamped =
            tf_buffer_->lookupTransform("base_link", msg->header.frame_id, msg->header.stamp);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "TF lookup failed despite MessageFilter: %s", ex.what());
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
    int valid_pts = 0;
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        if (std::isnan(*iter_x)) continue;
        h_in_[valid_pts * 3 + 0] = *iter_x;
        h_in_[valid_pts * 3 + 1] = *iter_y;
        h_in_[valid_pts * 3 + 2] = *iter_z;
        valid_pts++;
    }

    cudaMemcpy(d_in_, h_in_, valid_pts * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transform_, h_transform, 12 * sizeof(float), cudaMemcpyHostToDevice);

    int h_ground_count   = 0;
    int h_obstacle_count = 0;
    launchGroundFilter(
        d_in_,
        d_ground_,
        d_obstacle_,
        d_transform_,
        valid_pts,
        12.0f,
        0.6f,
        0.0f,
        1.2f,
        0.08f,
        d_ground_count_,
        d_obstacle_count_,
        &h_ground_count,
        &h_obstacle_count
    );

    if (h_obstacle_count > 50) {
        PoseCandidate search_base_pose;
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            search_base_pose = predicted_pose_;
        }

        PoseCandidate best_pose;
        float best_cost = 0.0f;

        const bool matched = launchFieldSDFMatcher(
            d_obstacle_,
            h_obstacle_count,
            search_base_pose,
            0.30f,
            0.03f,
            0.15f,
            0.02f,
            0.20f,
            best_pose,
            best_cost
        );

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "[SDF Match] points: %d, best_cost: %.4f, pose: (%.2f, %.2f, %.2f)",
            h_obstacle_count,
            best_cost,
            best_pose.x,
            best_pose.y,
            best_pose.yaw
        );

        if (matched && best_cost < 0.20f) {
            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                last_known_pose_ = best_pose;
                predicted_pose_  = best_pose;
            }

            // Quat 計算
            tf2::Quaternion q;
            q.setRPY(0.0, 0.0, best_pose.yaw);

            // 1. Pose With Covariance Message Publish
            auto pose_msg = std::make_unique<geometry_msgs::msg::PoseWithCovarianceStamped>();
            pose_msg->header.stamp          = msg->header.stamp;
            pose_msg->header.frame_id       = "map";
            pose_msg->pose.pose.position.x  = best_pose.x;
            pose_msg->pose.pose.position.y  = best_pose.y;
            pose_msg->pose.pose.position.z  = 0.0f;
            pose_msg->pose.pose.orientation = tf2::toMsg(q);
            pose_msg->pose.covariance[0]    = 0.005;
            pose_msg->pose.covariance[7]    = 0.005;
            pose_msg->pose.covariance[35]   = 0.002;
            pub_platform_pose_->publish(std::move(pose_msg));

            // 2. TF Broadcast (map -> base_link)
            geometry_msgs::msg::TransformStamped tf_msg;
            tf_msg.header.stamp            = msg->header.stamp;
            tf_msg.header.frame_id         = "map";
            tf_msg.child_frame_id          = "base_link";
            tf_msg.transform.translation.x = best_pose.x;
            tf_msg.transform.translation.y = best_pose.y;
            tf_msg.transform.translation.z = 0.0f;
            tf_msg.transform.rotation      = tf2::toMsg(q);
            tf_broadcaster_->sendTransform(tf_msg);
        }
    }

    cudaMemcpy(
        h_out_ground_, d_ground_, h_ground_count * 3 * sizeof(float), cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        h_out_obstacle_, d_obstacle_, h_obstacle_count * 3 * sizeof(float), cudaMemcpyDeviceToHost
    );

    std_msgs::msg::Header out_header = msg->header;
    out_header.frame_id              = "base_link";
    publishCloud(pub_ground_, out_header, h_out_ground_, h_ground_count);
    publishCloud(pub_obstacle_, out_header, h_out_obstacle_, h_obstacle_count);
}

void LocalizationNode::publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
    const std_msgs::msg::Header& header,
    const float* data,
    const int count
)
{
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
        *iter_x = data[i * 3 + 0];
        *iter_y = data[i * 3 + 1];
        *iter_z = data[i * 3 + 2];
    }
    pub->publish(std::move(out_msg));
}

void LocalizationNode::publishFieldMapMarkers()
{
    visualization_msgs::msg::MarkerArray marker_array;

    int id = 0;
    for (const auto& obj : map_objects_) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
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

            // 円柱オブジェクトの色指定
            marker.color.r = 0.9f;
            marker.color.g = 0.3f;
            marker.color.b = 0.1f;
        }

        marker_array.markers.push_back(marker);
    }

    pub_map_markers_->publish(marker_array);
}

void LocalizationNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);

    // 1. base_link -> IMU の回転行列を取得（初回またはキャッシュ更新）
    static Eigen::Matrix3d R_base_imu = Eigen::Matrix3d::Identity();
    static bool tf_initialized        = false;

    if (!tf_initialized) {
        try {
            geometry_msgs::msg::TransformStamped transform_stamped =
                tf_buffer_->lookupTransform("base_link", msg->header.frame_id, tf2::TimePointZero);
            const Eigen::Affine3d eigen_tf = tf2::transformToEigen(transform_stamped);
            R_base_imu                     = eigen_tf.rotation();
            tf_initialized                 = true;
        } catch (const tf2::TransformException& ex) {
            // TFがまだ利用可能でない場合はスキップ
            return;
        }
    }

    // 2. 時刻の初期化チェック
    if (!imu_initialized_) {
        last_imu_stamp_  = msg->header.stamp;
        imu_initialized_ = true;
        return;
    }

    const double dt = (rclcpp::Time(msg->header.stamp) - last_imu_stamp_).seconds();
    last_imu_stamp_ = msg->header.stamp;

    if (dt <= 0.0 || dt > 0.5) return;

    // 3. 角速度を base_link 座標系へ変換して Yaw 積分
    const Eigen::Vector3d omega_imu(
        msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z
    );
    const Eigen::Vector3d omega_base = R_base_imu * omega_imu;

    const float gz_base = static_cast<float>(omega_base.z());
    predicted_pose_.yaw += gz_base * static_cast<float>(dt);

    // 4. 角度の正規化 [-PI, PI]
    predicted_pose_.yaw = std::atan2(std::sin(predicted_pose_.yaw), std::cos(predicted_pose_.yaw));
}
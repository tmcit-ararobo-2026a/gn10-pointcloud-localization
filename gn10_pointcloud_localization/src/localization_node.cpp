#include "gn10_pointcloud_localization/localization_node.hpp"

#include <tf2_ros/transform_broadcaster.h>

#include <cmath>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <vector>

#include "gn10_pointcloud_localization/cuda/ground_filter.cuh"

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

// Python スクリプトと完全に一致させたフィールドマップ構築関数
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

    // --- 領域A (+Y) および 領域B (-Y) のオブジェクト定義 ---
    struct ObjectSpec {
        ObjectType type;
        float x, y;
        float param1, param2;  // CYL: radius, 0 / BOX: half_w, half_d
        float z_min, z_max;
    };

    const float b1_r = 0.273f / 2.0f;  // バケツ半径 0.1365m
    std::vector<ObjectSpec> base_specs;

    // バケツ① (φ0.273 x H0.255)
    base_specs.push_back({CYLINDER, 0.550f, 0.870f, b1_r, 0.000f, 0.000f, 0.255f});

    // バケツ② (台座 0.3x0.3xH0.6 + バケツ①)
    base_specs.push_back({BOX, -1.270f, 1.480f, 0.150f, 0.150f, 0.000f, 0.600f});
    base_specs.push_back({CYLINDER, -1.270f, 1.480f, b1_r, 0.000f, 0.600f, 0.855f});

    // バケツ③ (台座 0.3x0.3xH0.3 + バケツ①)
    base_specs.push_back({BOX, 2.370f, 1.480f, 0.150f, 0.150f, 0.000f, 0.300f});
    base_specs.push_back({CYLINDER, 2.370f, 1.480f, b1_r, 0.000f, 0.300f, 0.555f});

    // 椅子 (W0.36 x D0.40, H0.807)
    base_specs.push_back({BOX, 0.550f, 4.980f, 0.180f, 0.200f, 0.000f, 0.807f});

    // 机 (W0.65 x D0.45 x H0.76) - 4台
    float desk_coords[4][2] = {
        {-2.295f, 3.855f},
        { 3.395f, 3.855f},
        {-4.895f, 5.445f},
        {-4.750f, 1.105f}
    };
    for (int i = 0; i < 4; ++i) {
        base_specs.push_back(
            {BOX, desk_coords[i][0], desk_coords[i][1], 0.325f, 0.225f, 0.000f, 0.760f}
        );
    }

    // 旗 (土台 0.39x0.39xH0.18 + 支柱 φ0.06 x H3.0)
    base_specs.push_back({BOX, 0.550f, 3.025f, 0.195f, 0.195f, 0.000f, 0.180f});
    base_specs.push_back({CYLINDER, 0.550f, 3.025f, 0.030f, 0.000f, 0.180f, 3.000f});

    // 領域A (+Y) と 領域B (-Y) に対称展開
    for (const auto& spec : base_specs) {
        for (float y_sign : {1.0f, -1.0f}) {
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
    // TF
    tf_buffer_      = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_    = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    // サブスクライバ
    sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar",
        rclcpp::SensorDataQoS(),
        std::bind(&LocalizationNode::cloudCallback, this, std::placeholders::_1)
    );
    sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/livox/imu",
        rclcpp::SensorDataQoS(),
        std::bind(&LocalizationNode::imuCallback, this, std::placeholders::_1)
    );
    // パブリッシャ
    pub_ground_   = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_cloud", 10);
    pub_obstacle_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/obstacle_cloud", 10);
    pub_platform_pose_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/platform_constraint", 10
    );
    pub_map_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/field_map_markers", rclcpp::QoS(1).transient_local()
    );
    // 1秒周期でマップマーカーを配信
    marker_timer_ = this->create_wall_timer(
        std::chrono::seconds(1), std::bind(&LocalizationNode::publishFieldMapMarkers, this)
    );
    // CUDA メモリ確保
    cudaMalloc(&d_in_, max_points_ * 3 * sizeof(float));
    cudaMalloc(&d_ground_, max_points_ * 3 * sizeof(float));
    cudaMalloc(&d_obstacle_, max_points_ * 3 * sizeof(float));
    cudaMalloc(&d_ground_count_, sizeof(int));
    cudaMalloc(&d_obstacle_count_, sizeof(int));
    cudaMalloc(&d_transform_, 12 * sizeof(float));
    // Pinned Memory の使用で転送を高速化
    cudaMallocHost(&h_in_, max_points_ * 3 * sizeof(float));
    cudaMallocHost(&h_out_ground_, max_points_ * 3 * sizeof(float));
    cudaMallocHost(&h_out_obstacle_, max_points_ * 3 * sizeof(float));

    // マップ保持と転送（メンバ変数に保存して再構築を防ぐ）
    map_objects_ = createNHK2026FieldMap();
    uploadFieldMapToGPU(map_objects_);

    last_known_pose_ = {-4.0f, -4.0f, -1.5708f};  // 初期位置を左下隅に設定
    predicted_pose_  = last_known_pose_;          // ★ IMU予測姿勢も同じ初期位置で同期！

    is_initialized_ = false;

    // 初回にマーカーを出力
    publishFieldMapMarkers();
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
    int num_points = msg->width * msg->height;
    if (num_points == 0 || num_points > max_points_) return;

    geometry_msgs::msg::TransformStamped transform_stamped;
    try {
        transform_stamped = tf_buffer_->lookupTransform(
            "base_link",
            msg->header.frame_id,
            msg->header.stamp,
            rclcpp::Duration::from_seconds(0.03)
        );
    } catch (const tf2::TransformException& ex) {
        return;
    }

    Eigen::Affine3d eigen_tf = tf2::transformToEigen(transform_stamped);
    float h_transform[12];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            h_transform[r * 4 + c] = static_cast<float>(eigen_tf.matrix()(r, c));

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

    int h_ground_count = 0, h_obstacle_count = 0;
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
            // IMU で先回りして予測した姿勢を探索の原点にする
            search_base_pose = predicted_pose_;
        }

        PoseCandidate best_pose;
        float best_cost = 0.0f;

        // IMU予測により起点精度が高いため、探索範囲(range)を小さく絞り込んで計算速度を向上させる
        bool matched = launchFieldSDFMatcher(
            d_obstacle_,
            h_obstacle_count,
            search_base_pose,
            0.30f,
            0.03f,
            0.15f,
            0.02f,
            0.20f,  // max_dist_thresh
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
                // 正確な SDF Match 結果で IMU 予測姿勢のドリフトを補正
                last_known_pose_ = best_pose;
                predicted_pose_  = best_pose;
            }

            // 1. Topic Publish
            auto pose_msg = std::make_unique<geometry_msgs::msg::PoseWithCovarianceStamped>();
            pose_msg->header.stamp            = msg->header.stamp;
            pose_msg->header.frame_id         = "map";
            pose_msg->pose.pose.position.x    = best_pose.x;
            pose_msg->pose.pose.position.y    = best_pose.y;
            pose_msg->pose.pose.position.z    = 0.0f;
            pose_msg->pose.pose.orientation.z = std::sin(best_pose.yaw / 2.0f);
            pose_msg->pose.pose.orientation.w = std::cos(best_pose.yaw / 2.0f);
            pose_msg->pose.covariance[0]      = 0.005;
            pose_msg->pose.covariance[7]      = 0.005;
            pose_msg->pose.covariance[35]     = 0.002;
            pub_platform_pose_->publish(std::move(pose_msg));

            // 2. TF (map -> base_link) ブロードキャスト
            geometry_msgs::msg::TransformStamped tf_msg;
            tf_msg.header.stamp            = msg->header.stamp;
            tf_msg.header.frame_id         = "map";
            tf_msg.child_frame_id          = "base_link";
            tf_msg.transform.translation.x = best_pose.x;
            tf_msg.transform.translation.y = best_pose.y;
            tf_msg.transform.translation.z = 0.0f;
            tf_msg.transform.rotation.z    = std::sin(best_pose.yaw / 2.0f);
            tf_msg.transform.rotation.w    = std::cos(best_pose.yaw / 2.0f);
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
    int count
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

        // 色・透過度の共通設定 (緑系の半透明)
        marker.color.r = 0.1f;
        marker.color.g = 0.8f;
        marker.color.b = 0.4f;
        marker.color.a = 0.6f;

        if (obj.type == BOX) {
            marker.type  = visualization_msgs::msg::Marker::CUBE;
            float width  = obj.param1 * 2.0f;
            float depth  = obj.param2 * 2.0f;
            float height = obj.z_max - obj.z_min;

            // obj.x -> obj.center_x, obj.y -> obj.center_y に修正
            marker.pose.position.x = obj.center_x;
            marker.pose.position.y = obj.center_y;
            marker.pose.position.z = obj.z_min + height / 2.0f;

            marker.scale.x = width;
            marker.scale.y = depth;
            marker.scale.z = height;

        } else if (obj.type == CYLINDER) {
            marker.type  = visualization_msgs::msg::Marker::CYLINDER;
            float radius = obj.param1;
            float height = obj.z_max - obj.z_min;

            // obj.x -> obj.center_x, obj.y -> obj.center_y に修正
            marker.pose.position.x = obj.center_x;
            marker.pose.position.y = obj.center_y;
            marker.pose.position.z = obj.z_min + height / 2.0f;

            marker.scale.x = radius * 2.0f;
            marker.scale.y = radius * 2.0f;
            marker.scale.z = height;

            // 円柱（バケツ・ポール等）はオレンジ色で区別
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

    // 1. base_link -> livox_frame (または msg->header.frame_id) の回転TFを取得
    geometry_msgs::msg::TransformStamped transform_stamped;
    try {
        transform_stamped = tf_buffer_->lookupTransform(
            "base_link",
            msg->header.frame_id,
            msg->header.stamp,
            rclcpp::Duration::from_seconds(0.01)
        );
    } catch (const tf2::TransformException& ex) {
        return;  // TF取得未完了時は処理スキップ
    }

    Eigen::Affine3d eigen_tf   = tf2::transformToEigen(transform_stamped);
    Eigen::Matrix3d R_base_imu = eigen_tf.rotation();

    if (!imu_initialized_) {
        last_imu_stamp_  = msg->header.stamp;
        imu_initialized_ = true;
        return;
    }

    double dt       = (rclcpp::Time(msg->header.stamp) - last_imu_stamp_).seconds();
    last_imu_stamp_ = msg->header.stamp;

    if (dt <= 0.0 || dt > 0.5) return;

    // 2. IMUローカル角速度を base_link 座標系へ回転変換
    Eigen::Vector3d omega_imu(
        msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z
    );
    Eigen::Vector3d omega_base = R_base_imu * omega_imu;

    // 3. base_link の Z 軸角速度 (Yaw レート) で積分
    float gz_base = static_cast<float>(omega_base.z());
    predicted_pose_.yaw += gz_base * static_cast<float>(dt);

    // 正規化 [-PI, PI]
    while (predicted_pose_.yaw > M_PI) predicted_pose_.yaw -= 2.0f * M_PI;
    while (predicted_pose_.yaw < -M_PI) predicted_pose_.yaw += 2.0f * M_PI;
}
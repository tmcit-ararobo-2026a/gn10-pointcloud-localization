#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <cuda_runtime.h>
#include <Eigen/Geometry>
#include "gn10_pointcloud_localization/cuda/ground_filter.cuh"

class GroundFilterNode : public rclcpp::Node {
public:
    GroundFilterNode() : Node("gn10_ground_filter_node") {
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar", rclcpp::SensorDataQoS(),
            std::bind(&GroundFilterNode::cloudCallback, this, std::placeholders::_1));

        pub_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_cloud", 10);
        pub_obstacle_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/obstacle_cloud", 10);

        max_points_ = 200000;
        cudaMalloc(&d_in_, max_points_ * 3 * sizeof(float));
        cudaMalloc(&d_ground_, max_points_ * 3 * sizeof(float));
        cudaMalloc(&d_obstacle_, max_points_ * 3 * sizeof(float));
        cudaMalloc(&d_ground_count_, sizeof(int));
        cudaMalloc(&d_obstacle_count_, sizeof(int));
        cudaMalloc(&d_transform_, 12 * sizeof(float)); // 3x4 変換行列

        h_in_ = new float[max_points_ * 3];
        h_out_ground_ = new float[max_points_ * 3];
        h_out_obstacle_ = new float[max_points_ * 3];
    }

    ~GroundFilterNode() {
        cudaFree(d_in_);
        cudaFree(d_ground_);
        cudaFree(d_obstacle_);
        cudaFree(d_ground_count_);
        cudaFree(d_obstacle_count_);
        cudaFree(d_transform_);

        delete[] h_in_;
        delete[] h_out_ground_;
        delete[] h_out_obstacle_;
    }

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        int num_points = msg->width * msg->height;
        if (num_points == 0 || num_points > max_points_) return;

        // base_link -> livox_frame への TF を取得
        geometry_msgs::msg::TransformStamped transform_stamped;
        try {
            transform_stamped = tf_buffer_->lookupTransform(
                "base_link", msg->header.frame_id,
                msg->header.stamp, rclcpp::Duration::from_seconds(0.05));
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "TF Lookup failed: %s", ex.what());
            return;
        }

        // Eigen 経由で 3x4 行列 (R|t) に変換
        Eigen::Affine3d eigen_transform = tf2::transformToEigen(transform_stamped);
        float h_transform[12];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                h_transform[r * 4 + c] = static_cast<float>(eigen_transform.matrix()(r, c));
            }
        }

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        int valid_pts = 0;
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            if (std::isnan(*iter_x) || std::isnan(*iter_y) || std::isnan(*iter_z)) continue;
            h_in_[valid_pts * 3 + 0] = *iter_x;
            h_in_[valid_pts * 3 + 1] = *iter_y;
            h_in_[valid_pts * 3 + 2] = *iter_z;
            valid_pts++;
        }

        // HtoD 転送
        cudaMemcpy(d_in_, h_in_, valid_pts * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_transform_, h_transform, 12 * sizeof(float), cudaMemcpyHostToDevice);

        // CUDA 実行: フィルタパラメータの設定
        // robot_radius: 0.6m, robot_height_min: 0.0m, robot_height_max: 1.2m
        // ground_z_thresh: 0.08m (base_link 原点基準の地面高差)
        int h_ground_count = 0;
        int h_obstacle_count = 0;
        launchGroundFilter(
            d_in_, d_ground_, d_obstacle_, d_transform_, valid_pts,
            12.0f, 0.6f, 0.0f, 1.2f, 0.08f,
            d_ground_count_, d_obstacle_count_, &h_ground_count, &h_obstacle_count
        );

        // DtoH 転送
        cudaMemcpy(h_out_ground_, d_ground_, h_ground_count * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_obstacle_, d_obstacle_, h_obstacle_count * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        // base_link 基準のヘッダーを作成して Publish
        std_msgs::msg::Header out_header = msg->header;
        out_header.frame_id = "base_link";

        publishCloud(pub_ground_, out_header, h_out_ground_, h_ground_count);
        publishCloud(pub_obstacle_, out_header, h_out_obstacle_, h_obstacle_count);
    }

    void publishCloud(
        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
        const std_msgs::msg::Header& header, const float* data, int count) 
    {
        auto out_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
        out_msg->header = header;
        out_msg->height = 1;
        out_msg->width = count;
        out_msg->is_dense = true;
        out_msg->is_bigendian = false;

        sensor_msgs::PointCloud2Modifier modifier(*out_msg);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(count);

        sensor_msgs::PointCloud2Iterator<float> iter_x(*out_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(*out_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(*out_msg, "z");

        for (int i = 0; i < count; ++i, ++iter_x, ++iter_y, ++iter_z) {
            *iter_x = data[i * 3 + 0];
            *iter_y = data[i * 3 + 1];
            *iter_z = data[i * 3 + 2];
        }

        pub->publish(std::move(out_msg));
    }

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obstacle_;

    int max_points_;
    float *d_in_, *d_ground_, *d_obstacle_, *d_transform_;
    int *d_ground_count_, *d_obstacle_count_;
    float *h_in_, *h_out_ground_, *h_out_obstacle_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GroundFilterNode>());
    rclcpp::shutdown();
    return 0;
}
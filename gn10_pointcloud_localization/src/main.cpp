#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cuda_runtime.h>
#include "gn10_pointcloud_localization/cuda/ground_filter.cuh"

class GroundFilterNode : public rclcpp::Node {
public:
    GroundFilterNode() : Node("gn10_ground_filter_node") {
        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar", rclcpp::SensorDataQoS(),
            std::bind(&GroundFilterNode::cloudCallback, this, std::placeholders::_1));

        pub_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_cloud", 10);
        pub_obstacle_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/obstacle_cloud", 10);

        // CUDA メモリ事前確保 (最大 200,000 点分)
        max_points_ = 200000;
        cudaMalloc(&d_in_, max_points_ * 3 * sizeof(float));
        cudaMalloc(&d_ground_, max_points_ * 3 * sizeof(float));
        cudaMalloc(&d_obstacle_, max_points_ * 3 * sizeof(float));
        cudaMalloc(&d_ground_count_, sizeof(int));
        cudaMalloc(&d_obstacle_count_, sizeof(int));

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

        delete[] h_in_;
        delete[] h_out_ground_;
        delete[] h_out_obstacle_;
    }

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        int num_points = msg->width * msg->height;
        if (num_points == 0 || num_points > max_points_) return;

        // PointCloud2 -> 連続 float 配列に抽出
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

        // カーネル実行 (半径 12.0m, Z閾値 -0.2m)
        int h_ground_count = 0;
        int h_obstacle_count = 0;
        launchGroundFilter(
            d_in_, d_ground_, d_obstacle_, valid_pts, 12.0f, -0.2f,
            d_ground_count_, d_obstacle_count_, &h_ground_count, &h_obstacle_count
        );

        // DtoH 転送
        cudaMemcpy(h_out_ground_, d_ground_, h_ground_count * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_obstacle_, d_obstacle_, h_obstacle_count * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        // ROS2 メッセージとして Publish
        publishCloud(pub_ground_, msg->header, h_out_ground_, h_ground_count);
        publishCloud(pub_obstacle_, msg->header, h_out_obstacle_, h_obstacle_count);
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

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obstacle_;

    int max_points_;
    float *d_in_, *d_ground_, *d_obstacle_;
    int *d_ground_count_, *d_obstacle_count_;
    float *h_in_, *h_out_ground_, *h_out_obstacle_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GroundFilterNode>());
    rclcpp::shutdown();
    return 0;
}
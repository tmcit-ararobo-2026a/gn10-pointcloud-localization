"""Verify sensor and processor freshness independently of CUDA matching."""

import os
import subprocess
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2

from gn10_pointcloud_localization.msg import LocalizationStatus


def test_localization_status_expiry():
    """Report input versus processing loss and never approve fixed-covariance Ready."""
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    os.environ['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 100)
    context = rclpy.context.Context()
    rclpy.init(context=context)
    node = Node('localization_status_test', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    clouds = node.create_publisher(PointCloud2, '/livox/lidar', 10)
    imus = node.create_publisher(Imu, '/livox/imu', 10)
    results = node.create_publisher(LocalizationStatus, '/localization/status_raw', 10)
    outputs = []
    node.create_subscription(LocalizationStatus, '/localization/status', outputs.append, 10)
    process = subprocess.Popen([os.environ['STATUS_MONITOR_EXECUTABLE'], '--ros-args',
                                '-p', 'status_monitor.input_timeout_s:=0.1',
                                '-p', 'status_monitor.processing_timeout_s:=0.1'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sequence = 0

    def send(cloud=True, imu=True, result=True, frozen=None, matched=True):
        """Feed synthetic progressing inputs or an intentionally frozen raw outcome."""
        nonlocal sequence
        stamp = node.get_clock().now().to_msg()
        if cloud:
            message = PointCloud2()
            message.header.stamp = stamp
            clouds.publish(message)
        if imu:
            message = Imu()
            message.header.stamp = stamp
            imus.publish(message)
        if result:
            sequence += 1
            message = LocalizationStatus()
            message.header.stamp = stamp
            message.header.frame_id = 'map'
            message.measurement_stamp = stamp
            message.pose_stamp = stamp
            message.processor_session = 123
            message.result_sequence = sequence
            message.last_match_success = matched
            message.lost = not matched
            message.map_loaded = True
            message.cloud_tf_available = True
            message.imu_tf_available = True
            message.match_metrics_valid = True
            message.best_cost = 0.1
            # Even a raw publisher claiming these flags must not approve a missing contract.
            message.covariance_valid = True
            message.ready_quality_valid = True
            message.reason = 'Synthetic matching outcome'
            results.publish(frozen or message)
            return frozen or message

    def wait_for(predicate, repeated=None):
        """Wait for a status observation while processing isolated DDS callbacks."""
        end = time.monotonic() + 5
        while time.monotonic() < end:
            assert process.poll() is None
            if repeated:
                repeated()
            executor.spin_once(timeout_sec=0.005)
            if predicate():
                return
        raise AssertionError('localization status outcome timed out')

    def fresh():
        """Require both input streams and a completed current processing outcome."""
        return outputs[-1].processing_fresh and outputs[-1].cloud_fresh and outputs[-1].imu_fresh

    try:
        wait_for(lambda: outputs and outputs[-1].lost and not outputs[-1].processing_fresh)
        wait_for(fresh, repeated=send)
        assert outputs[-1].last_match_success and outputs[-1].match_metrics_valid
        assert not outputs[-1].covariance_valid and not outputs[-1].ready_quality_valid
        wait_for(lambda: not outputs[-1].cloud_fresh and outputs[-1].imu_fresh
                 and outputs[-1].processing_fresh, repeated=lambda: send(cloud=False))
        wait_for(fresh, repeated=send)
        wait_for(lambda: not outputs[-1].imu_fresh and outputs[-1].cloud_fresh
                 and outputs[-1].processing_fresh, repeated=lambda: send(imu=False))
        wait_for(fresh, repeated=send)
        wait_for(lambda: not outputs[-1].last_match_success and outputs[-1].lost
                 and outputs[-1].processing_fresh, repeated=lambda: send(matched=False))
        wait_for(fresh, repeated=send)
        frozen = send()
        wait_for(lambda: not outputs[-1].processing_fresh and outputs[-1].lost,
                 repeated=lambda: send(frozen=frozen))
        assert not outputs[-1].last_match_success
        assert not outputs[-1].match_metrics_valid
        wait_for(fresh, repeated=send)
        wait_for(lambda: not outputs[-1].processing_fresh and outputs[-1].cloud_fresh
                 and outputs[-1].imu_fresh, repeated=lambda: send(result=False))
        assert all(not status.ready_quality_valid and not status.covariance_valid for status in outputs)
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=context)

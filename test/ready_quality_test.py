"""Exercise actual quality monitor approval, measurement TF and stability rules."""
import json
import os
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Imu, PointCloud2
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time
from tf2_ros import TransformBroadcaster
from gn10_pointcloud_localization.msg import LocalizationStatus


def test_ready_quality(tmp_path):
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    os.environ['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 100)
    profile = dict(schema=1, approved=True, mode='simulation', quality_id='q',
                   evidence_id='synthetic-qa', map_id='map-hash', team='red', extrinsic_id='ext',
                   implementation_id='software', configuration_id='settings', max_best_cost=.165,
                   min_inliers=60, max_inlier_cost=.05, max_cloud_age=.15, max_imu_age=.15,
                   max_result_age=.4, max_pose_age=.4, max_tf_age=.4,
                   max_position_error=.05, max_yaw_error=.05, max_pose_jump=.2, max_yaw_jump=.2)
    filename = tmp_path / 'quality.json'
    filename.write_text(json.dumps(profile))
    context = rclpy.context.Context()
    rclpy.init(context=context)
    node = Node('quality_contract_test', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    cloud_pub = node.create_publisher(PointCloud2, '/livox/lidar', 10)
    imu_pub = node.create_publisher(Imu, '/livox/imu', 10)
    raw_pub = node.create_publisher(LocalizationStatus, '/localization/status_raw', 10)
    broadcaster = TransformBroadcaster(node)
    reports = []
    node.create_subscription(LocalizationStatus, '/localization/status', reports.append, 10)
    process = subprocess.Popen([os.environ['STATUS_MONITOR_EXECUTABLE'], '--ros-args',
                                '-p', f'status_monitor.quality_profile:={filename}'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sequence = 0
    session = 4
    frozen = None

    def feed(change=None, result=True):
        nonlocal sequence, frozen
        stamp = node.get_clock().now().to_msg()
        cloud = PointCloud2()
        cloud.header.stamp, cloud.header.frame_id = stamp, 'base_link'
        cloud_pub.publish(cloud)
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = stamp, 'base_link'
        imu_pub.publish(imu)
        if not result:
            raw_pub.publish(frozen)
            return
        transform = TransformStamped()
        transform.header.stamp, transform.header.frame_id = stamp, 'map'
        transform.child_frame_id = 'base_link'
        transform.transform.rotation.w = 1.
        broadcaster.sendTransform(transform)
        sequence += 1
        raw = LocalizationStatus()
        raw.header.stamp, raw.header.frame_id = stamp, 'map'
        raw.measurement_stamp, raw.pose_stamp = stamp, stamp
        raw.processor_session, raw.result_sequence = session, sequence
        raw.last_match_success, raw.map_loaded = True, True
        raw.cloud_tf_available, raw.imu_tf_available = True, True
        raw.match_metrics_valid, raw.inlier_metrics_valid = True, True
        raw.best_cost, raw.inlier_count, raw.inlier_cost = .1, 100, .01
        raw.pose.header.stamp, raw.pose.header.frame_id = stamp, 'map'
        raw.pose.pose.orientation.w, raw.pose_valid = 1., True
        for key in ('map_id', 'team', 'extrinsic_id', 'implementation_id', 'configuration_id'):
            setattr(raw, key, profile[key])
        if change:
            change(raw)
        raw_pub.publish(raw)
        frozen = raw

    def wait(predicate, supplier=feed, timeout=4):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            assert process.poll() is None
            supplier()
            executor.spin_once(timeout_sec=.02)
            if reports and predicate(reports[-1]):
                return
        raise AssertionError('Quality outcome timeout: ' + (reports[-1].reason if reports else 'no reports'))

    try:
        wait(lambda s: s.processing_fresh)
        # Repeated reports of the same raw result cannot manufacture ten results.
        start = time.monotonic()
        wait(lambda s: time.monotonic() - start > 1.05, supplier=lambda: feed(result=False))
        assert not reports[-1].ready_quality_valid
        start = time.monotonic()
        wait(lambda s: s.ready_quality_valid)
        assert time.monotonic() - start >= .98
        assert reports[-1].simulation_profile and not reports[-1].covariance_valid
        assert reports[-1].pose_tf_valid and reports[-1].quality_id == 'q'
        for change in (lambda s: setattr(s, 'using_fallback_map', True),
                       lambda s: setattr(s, 'best_cost', .2),
                       lambda s: setattr(s, 'map_id', 'wrong-map'),
                       lambda s: setattr(s, 'inlier_count', 59),
                       lambda s: setattr(s, 'best_cost', float('nan')),
                       lambda s: setattr(s, 'measurement_stamp', Time(sec=s.measurement_stamp.sec + 1, nanosec=s.measurement_stamp.nanosec)),
                       lambda s: setattr(s.pose.pose.position, 'x', 1.)):
            wait(lambda s: not s.ready_quality_valid, supplier=lambda: feed(change))
            wait(lambda s: s.ready_quality_valid)
        session = 5
        wait(lambda s: s.processor_session == session and not s.ready_quality_valid)
        wait(lambda s: s.ready_quality_valid)
        assert reports[-1].monitor_session != 0
    finally:
        process.terminate()
        output = process.communicate(timeout=5)[0]
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=context)
        if process.returncode not in (0, -15):
            print(output.decode(errors='replace'))

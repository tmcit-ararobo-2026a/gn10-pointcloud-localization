"""Regression: withholding map TF must not stop the matching prior/recovery loop."""
import math
import os
import subprocess
import sys
import tempfile
import time

os.environ['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 100)
os.environ['ROS_LOCALHOST_ONLY'] = '1'
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


def main():
    rclpy.init(args=[])
    node = rclpy.create_node('fusion_feedback_check')
    poses, priors = [], []
    pose_sub = node.create_subscription(PoseWithCovarianceStamped, '/platform_constraint', poses.append, 100)
    prior_sub = node.create_subscription(PoseWithCovarianceStamped, '/gn10/matching_prior', priors.append, 100)
    odom = node.create_publisher(Odometry, '/Odometry', 20)
    init = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
    match = node.create_publisher(PoseWithCovarianceStamped, '/platform_constraint_raw', 10)
    log = tempfile.TemporaryFile(mode='w+')
    process = subprocess.Popen([sys.argv[1], '--ros-args', '-p', 'frames.body_frame:=base_link', '-p', 'fusion.map_only_updates:=true'],
                               stdout=log, stderr=subprocess.STDOUT)

    def spin(duration=0.12):
        end = time.monotonic() + duration
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)

    def wait_for(predicate, timeout=5):
        end = time.monotonic() + timeout
        while not predicate() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.02)
        assert predicate(), 'Timed out waiting for ROS output'

    def set_stamp(msg, stamp):
        ns = int(round(stamp * 1e9))
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 1_000_000_000)

    def motion(stamp, x):
        msg = Odometry()
        set_stamp(msg, stamp)
        msg.header.frame_id = 'camera_init'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.orientation.w = 1.0
        odom.publish(msg)
        spin()

    def map_pose(pub, stamp, x, yaw=0.0):
        msg = PoseWithCovarianceStamped()
        set_stamp(msg, stamp)
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.orientation.z = math.sin(yaw/2)
        msg.pose.pose.orientation.w = math.cos(yaw/2)
        pub.publish(msg)
        spin()

    def seconds(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec/1e9

    try:
        wait_for(lambda: all(pub.get_subscription_count() for pub in (odom, init, match)))
        spin(0.3)
        motion(100.0, 0.0)
        map_pose(init, 100.0, 10.0)
        wait_for(lambda: bool(poses) and bool(priors))
        assert abs(poses[-1].pose.pose.position.x - 10) < 1e-6
        for i in range(1, 16):
            motion(100+i*0.1, i*0.01)
        assert max(map(seconds, poses)) <= 101.000001, 'Stale prediction emitted as map pose'
        assert seconds(priors[-1]) >= 101.499999, 'Matching prior stopped with map TF'
        before = len(poses)
        motion(101.6, 100)
        assert len(poses) == before, 'Bad odometry emitted as map pose'
        for i in range(1, 4):
            motion(101.6+i*0.1, 100+i*0.01)
        assert len(poses) == before, 'Recovery emitted without map observation'
        map_pose(match, 101.9, 10.15)
        wait_for(lambda: len(poses) > before)
        assert seconds(poses[-1]) == seconds(priors[-1]), 'Correction changed sensor timestamp'
        motion(102.0, 100.04)
        assert abs(poses[-1].pose.pose.position.x - 10.16) < 0.05

        # Alternating near-but-inconsistent map hypotheses must not authorize recovery.
        for i in range(1, 4):
            motion(102+i*0.1, 100.04+i*0.01)
            predicted = priors[-1].pose.pose
            x_before = predicted.position.x
            yaw = 2*math.atan2(predicted.orientation.z, predicted.orientation.w)
            map_pose(match, 102+i*0.1, x_before+(0.65 if i % 2 else -0.65), yaw+0.65)
            assert abs(priors[-1].pose.pose.position.x-x_before) < 1e-6

        # Three consistent errors permit a bounded correction, without new odometry.
        recovered = False
        for i in range(4, 7):
            motion(102+i*0.1, 100.04+i*0.01)
            predicted = priors[-1].pose.pose
            x_before = predicted.position.x
            yaw = 2*math.atan2(predicted.orientation.z, predicted.orientation.w)
            map_pose(match, 102+i*0.1, x_before+0.65, yaw+0.65)
            change = priors[-1].pose.pose.position.x-x_before
            assert -1e-6 <= change <= 0.100001, 'Recovery exceeded bounded correction'
            recovered = recovered or change > 1e-5
        assert recovered, 'Consistent local observations never resumed correction'
        before = len(poses)
        motion(110.0, 100.20)
        for i in range(1,4):
            motion(110+i*0.1, 100.20+i*0.01)
        assert seconds(priors[-1]) >= 110.299999, 'Long motion gap erased the search anchor'
        assert len(poses) == before, 'Long motion gap emitted stale map pose'
        predicted = priors[-1].pose.pose
        yaw = 2*math.atan2(predicted.orientation.z, predicted.orientation.w)
        map_pose(match, 110.3, predicted.position.x, yaw)
        wait_for(lambda: len(poses) > before)
        # A map observation can update during an unhealthy odometry segment.
        observed_x=poses[-1].pose.pose.position.x
        observed_yaw=2*math.atan2(poses[-1].pose.pose.orientation.z,poses[-1].pose.pose.orientation.w)
        motion(110.4,500)
        before=len(poses)
        map_pose(match,110.5,observed_x+.02,observed_yaw)
        wait_for(lambda: len(poses)>before)
        assert abs(seconds(poses[-1])-110.5)<1e-6, 'Map-only update has the wrong observation timestamp'
        before=len(poses)
        map_pose(match,110.6,observed_x+2,observed_yaw)
        assert len(poses)==before, 'Unknown odometry authorized a map jump'
        assert abs(seconds(priors[-1])-110.6)<1e-6, 'Rejected map-only observation stopped the internal prior'
        for i in range(1,4):
            motion(110.6+i*.1,500+i*.01)
        assert abs(priors[-1].pose.pose.position.x-observed_x)<.1, 'Reconnected odometry counted unknown motion'
        map_pose(match,110.9,observed_x+.02,observed_yaw)
        wait_for(lambda: len(poses)>before)
        print('PASS: prior continuity, bounded recovery, map-only update and control resynchronization')
    except Exception:
        log.seek(0)
        print(log.read())
        raise
    finally:
        process.terminate()
        process.wait(timeout=5)
        log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

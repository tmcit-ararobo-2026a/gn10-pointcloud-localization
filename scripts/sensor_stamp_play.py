#!/usr/bin/env python3
"""Replay header-first LiDAR/IMU messages at their acquisition times (SQLite bags)."""
import argparse
from pathlib import Path
import sqlite3
import struct
import time

TYPES = {'sensor_msgs/msg/Imu', 'sensor_msgs/msg/PointCloud2',
         'livox_ros_driver2/msg/CustomMsg'}


def header_stamp_ns(prefix):
    if len(prefix) < 12 or prefix[:2] not in (b'\x00\x01', b'\x00\x00'):
        raise ValueError('Expected CDR with Header as the first message field')
    sec, nsec = struct.unpack_from('<iI' if prefix[1] else '>iI', prefix, 4)
    if sec < 0 or nsec >= 1_000_000_000:
        raise ValueError('Invalid sensor header stamp')
    return sec * 1_000_000_000 + nsec


def index_bag(path, topics):
    path = Path(path)
    files = sorted(path.glob('*.db3')) if path.is_dir() else [path]
    if not files:
        raise ValueError('No SQLite .db3 bag files found')
    connections, events, topic_types = [], [], {}
    try:
        for file in files:
            conn = sqlite3.connect(file.resolve().as_uri() + '?mode=ro', uri=True)
            connections.append(conn)
            for tid, name, typ in conn.execute('SELECT id,name,type FROM topics'):
                if name not in topics:
                    continue
                if typ not in TYPES or (name in topic_types and topic_types[name] != typ):
                    raise ValueError('Unsupported or inconsistent sensor type: ' + typ)
                topic_types[name] = typ
                for mid, prefix in conn.execute(
                        'SELECT id,substr(data,1,12) FROM messages WHERE topic_id=?', (tid,)):
                    stamp = header_stamp_ns(prefix)
                    # IMU precedes LiDAR when timestamps tie.
                    events.append((stamp, 0 if typ == 'sensor_msgs/msg/Imu' else 1,
                                   len(connections)-1, mid, name))
        if set(topics) != {event[4] for event in events} or not events:
            raise ValueError('Requested sensor topics are missing or empty')
        events.sort()
        return connections, events, topic_types
    except Exception:
        for conn in connections:
            conn.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag')
    parser.add_argument('--topics', nargs='+', default=['/livox/lidar', '/livox/imu'])
    parser.add_argument('--rate', type=float, default=1.0)
    args = parser.parse_args()
    if not 0 < args.rate <= 10:
        parser.error('--rate must be between 0 and 10')
    connections, events, topic_types = index_bag(args.bag, args.topics)
    import rclpy
    from rclpy.qos import QoSProfile
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosgraph_msgs.msg import Clock
    rclpy.init(args=[])
    node = rclpy.create_node('sensor_stamp_bag_player')
    types = {topic: get_message(typ) for topic, typ in topic_types.items()}
    publishers = {topic: node.create_publisher(cls, topic, QoSProfile(depth=200 if topic_types[topic] == 'sensor_msgs/msg/Imu' else 30))
                  for topic, cls in types.items()}
    clock = node.create_publisher(Clock, '/clock', 10)
    counts = dict.fromkeys(types, 0)
    first, last = events[0][0], events[-1][0]
    node.get_logger().info(f'{len(events)} sensor messages, {(last-first)/1e9:.3f}s; header-time replay')
    try:
        # Discovery before starting the sensor timeline.
        ready = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < ready:
            rclpy.spin_once(node, timeout_sec=0.02)
        origin = time.monotonic()
        current = first
        def publish_clock(stamp):
            msg = Clock()
            msg.clock.sec, msg.clock.nanosec = divmod(stamp, 1_000_000_000)
            clock.publish(msg)
        for stamp, _, db, mid, topic in events:
            # Fetch/decode before the due time to keep database IO off the timing path.
            data = connections[db].execute('SELECT data FROM messages WHERE id=?', (mid,)).fetchone()[0]
            msg = deserialize_message(data, types[topic])
            while rclpy.ok():
                due = origin + (stamp-first)/1e9/args.rate
                remaining = due-time.monotonic()
                if remaining <= 0:
                    break
                current = max(current, min(stamp, first+int((time.monotonic()-origin)*args.rate*1e9)))
                publish_clock(current)
                rclpy.spin_once(node, timeout_sec=min(remaining, 0.01))
            if not rclpy.ok():
                break
            current = max(current, stamp)
            publish_clock(current)
            publishers[topic].publish(msg)
            counts[topic] += 1
            rclpy.spin_once(node, timeout_sec=0)
        node.get_logger().info(f'Published: {counts}')
        finish = time.monotonic()+2.0
        while rclpy.ok() and time.monotonic() < finish:
            publish_clock(current)
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        for conn in connections:
            conn.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

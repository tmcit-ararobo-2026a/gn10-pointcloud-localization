import importlib.util
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('player', Path(__file__).parents[1]/'scripts/sensor_stamp_play.py')
player = importlib.util.module_from_spec(spec)
spec.loader.exec_module(player)

class SensorReplayTest(unittest.TestCase):
    def test_header_cdr(self):
        self.assertEqual(player.header_stamp_ns(b'\x00\x01\x00\x00'+struct.pack('<iI', 42, 123)), 42_000_000_123)
        self.assertEqual(player.header_stamp_ns(b'\x00\x00\x00\x00'+struct.pack('>iI', 42, 123)), 42_000_000_123)
        with self.assertRaises(ValueError):
            player.header_stamp_ns(b'\x00\x01\x00\x00'+struct.pack('<iI', 42, 1_000_000_000))

    def test_order_and_missing_interval(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder)/'bag.db3'
            conn = sqlite3.connect(db)
            conn.executescript('CREATE TABLE topics(id INTEGER,name TEXT,type TEXT); CREATE TABLE messages(id INTEGER,topic_id INTEGER,data BLOB);')
            conn.executemany('INSERT INTO topics VALUES(?,?,?)', [(1,'/lidar','livox_ros_driver2/msg/CustomMsg'),(2,'/imu','sensor_msgs/msg/Imu')])
            # Arrival order differs from sensor time; a 2 s missing interval must stay missing.
            for mid,tid,sec in [(1,1,12),(2,2,10),(3,1,10),(4,2,12)]:
                conn.execute('INSERT INTO messages VALUES(?,?,?)',(mid,tid,b'\x00\x01\x00\x00'+struct.pack('<iI',sec,0)))
            conn.commit();conn.close()
            conns, events, _ = player.index_bag(db, ['/lidar','/imu'])
            try:
                self.assertEqual([e[4] for e in events], ['/imu','/lidar','/imu','/lidar'])
                self.assertEqual(events[2][0]-events[1][0], 2_000_000_000)
                self.assertEqual(len(events),4)
            finally:
                for conn in conns: conn.close()

if __name__ == '__main__': unittest.main()

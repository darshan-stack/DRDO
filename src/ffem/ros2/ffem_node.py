"""ROS 2 subscriber/processor/publisher adapter for FFEM."""
from __future__ import annotations
import json
import time
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Float32MultiArray, String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from ffem.pipeline import FFEMPipeline, FFEMConfig
from ffem.ros2.pointcloud2_codec import decode_pointcloud2, encode_pointcloud2

if ROS_AVAILABLE:
    class FFEMNode(Node):
        def __init__(self):
            super().__init__('ffem_mapper')
            self.declare_parameter('input_topic', '/points')
            self.declare_parameter('map_topic', '~/map/elevation')
            self.declare_parameter('moving_topic', '~/map/moving_points')
            self.declare_parameter('metrics_topic', '~/metrics')
            self.declare_parameter('events_topic', '~/refinement_events')
            self.declare_parameter('map_frame', 'map')
            self.declare_parameter('queue_depth', 5)
            self.declare_parameter('max_points_per_frame', 150000)
            self.declare_parameter('use_input_motion', False)
            cfg = FFEMConfig(
                base_cell_size=float(self.declare_parameter('base_cell_size', 1.0).value),
                finest_cell_size=float(self.declare_parameter('finest_cell_size', 0.25).value),
                max_level=int(self.declare_parameter('max_level', 2).value),
                max_active_cells=int(self.declare_parameter('max_active_cells', 20000).value),
                max_topology_changes=int(self.declare_parameter('max_topology_changes', 32).value),
            )
            self.pipeline = FFEMPipeline(cfg)
            depth = int(self.get_parameter('queue_depth').value)
            qos = QoSProfile(depth=depth, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.sub = self.create_subscription(PointCloud2, str(self.get_parameter('input_topic').value), self.callback, qos)
            self.map_pub = self.create_publisher(PointCloud2, str(self.get_parameter('map_topic').value), 5)
            self.moving_pub = self.create_publisher(PointCloud2, str(self.get_parameter('moving_topic').value), 5)
            self.metrics_pub = self.create_publisher(Float32MultiArray, str(self.get_parameter('metrics_topic').value), 5)
            self.events_pub = self.create_publisher(String, str(self.get_parameter('events_topic').value), 5)
            self.map_frame = str(self.get_parameter('map_frame').value)
            self.max_points = int(self.get_parameter('max_points_per_frame').value)
            self.frame = 0
            self.last_log = time.monotonic()
            self.get_logger().info(f'FFEM ready: subscribed to {self.get_parameter("input_topic").value}')

        def callback(self, msg: PointCloud2) -> None:
            received = time.perf_counter()
            try:
                decoded = decode_pointcloud2(msg, remove_invalid=True)
                if len(decoded.points) == 0: return
                if len(decoded.points) > self.max_points:
                    idx = self._deterministic_downsample(len(decoded.points), self.max_points)
                    points = decoded.points[idx]; intensity = decoded.intensity[idx] if decoded.intensity is not None else None
                else:
                    points, intensity = decoded.points, decoded.intensity
                result = self.pipeline.process_points(points, intensity=intensity, frame=self.frame)
                self.publish_outputs(result, msg.header.stamp)
                self.frame += 1
                if time.monotonic() - self.last_log > 2.0:
                    s = result['stats']; self.get_logger().info(f'frame={self.frame} points={s["points"]} cells={s["active_cells"]} latency={s["total_ms"]:.1f}ms'); self.last_log = time.monotonic()
            except Exception as exc:
                self.get_logger().error(f'PointCloud2 processing failed: {exc}')

        @staticmethod
        def _deterministic_downsample(n: int, limit: int):
            return __import__('numpy').linspace(0, n - 1, limit, dtype=int)

        def publish_outputs(self, result: dict, stamp) -> None:
            import numpy as np
            map_points, map_colors, levels = self.pipeline.mapping.arrays()
            map_msg = encode_pointcloud2(map_points, frame_id=self.map_frame, stamp=stamp)
            self.map_pub.publish(map_msg)
            moving_points = result['points'][result['moving']]
            self.moving_pub.publish(encode_pointcloud2(moving_points, frame_id=self.map_frame, stamp=stamp))
            s = result['stats']; metrics = Float32MultiArray(); metrics.data = [float(s['total_ms']), float(s['map_ms']), float(s['active_cells']), float(s['topology_changes']), float(s['points']), float(s['moving_points'])]
            self.metrics_pub.publish(metrics)
            if self.pipeline.mapping.events:
                event = String(); event.data = json.dumps(self.pipeline.mapping.events[-1]); self.events_pub.publish(event)

    def main(args=None):
        rclpy.init(args=args); node = FFEMNode()
        try: rclpy.spin(node)
        except KeyboardInterrupt: pass
        finally: node.destroy_node(); rclpy.shutdown()
else:
    class FFEMNode:  # pragma: no cover
        def __init__(self): raise RuntimeError('ROS 2 is not installed. Install rclpy and sensor_msgs to run FFEMNode.')
    def main(args=None): raise RuntimeError('ROS 2 is not installed.')

if __name__ == '__main__': main()

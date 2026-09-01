"""ROS 2 adapter for FFEM.

The core pipeline remains ROS-independent. This node is importable only when
rclpy is installed, which lets development continue on non-ROS machines.
"""
from __future__ import annotations
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from ffem.pipeline import FFEMPipeline

if ROS_AVAILABLE:
    class FFEMNode(Node):
        def __init__(self):
            super().__init__('ffem_mapper')
            self.pipeline = FFEMPipeline()
            self.frame = 0
            self.subscription = self.create_subscription(PointCloud2, '/points', self.callback, 10)
            self.get_logger().info('FFEM ROS 2 node started; awaiting /points')

        def callback(self, _msg: PointCloud2) -> None:
            # PointCloud2 decoding is deliberately isolated for the next ROS milestone.
            # The core and synthetic replay are fully runnable without ROS.
            result = self.pipeline.step(self.frame)
            self.get_logger().debug(f"frame={self.frame} cells={result['stats']['active_cells']}")
            self.frame += 1

    def main(args=None):
        rclpy.init(args=args); node=FFEMNode()
        try: rclpy.spin(node)
        finally: node.destroy_node(); rclpy.shutdown()
else:
    class FFEMNode:  # pragma: no cover
        def __init__(self): raise RuntimeError('ROS 2 is not installed. Install rclpy and sensor_msgs to run FFEMNode.')
    def main(args=None): raise RuntimeError('ROS 2 is not installed.')

if __name__ == '__main__': main()

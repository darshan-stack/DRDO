#!/usr/bin/env python3
"""Live Open3D viewer for a ROS 2 PointCloud2 stream.

Subscribes to the real CARLA/FFEM PointCloud2 topics. This is intentionally
separate from scripts/run_replay.py so the viewer cannot accidentally display
synthetic replay data during the live CARLA demo.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import PointCloud2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ROS 2 Python packages are not available; source /opt/ros/humble/setup.bash") from exc

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Open3D is not installed; run python3 -m pip install open3d") from exc

from ffem.ros2.pointcloud2_codec import decode_pointcloud2


class LiveViewer(Node):
    def __init__(self, topic: str):
        super().__init__("ffem_open3d_viewer")
        self.topic = topic
        self.vis = o3d.visualization.Visualizer()
        if not self.vis.create_window("FFEM Live LiDAR", 1280, 720):
            raise RuntimeError("Open3D failed to create a window")
        self.cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.cloud)
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        self.vis.add_geometry(self.axis)
        self.ready = False
        self.frames = 0
        self.last_points = 0

        qos = QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.sub = self.create_subscription(PointCloud2, topic, self.callback, qos)
        self.get_logger().info(f"Open3D subscribing to {topic}")

    def callback(self, msg: PointCloud2) -> None:
        try:
            decoded = decode_pointcloud2(msg, remove_invalid=True)
            points = np.asarray(decoded.points, dtype=np.float64).reshape(-1, 3)
            if len(points) == 0:
                self.get_logger().warning("Received PointCloud2 with zero valid XYZ points", throttle_duration_sec=5.0)
                return

            self.cloud.points = o3d.utility.Vector3dVector(points)
            colors = np.tile(np.array([[0.30, 0.70, 1.00]]), (len(points), 1))
            self.cloud.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.cloud)

            if not self.ready:
                bbox = self.cloud.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                view = self.vis.get_view_control()
                view.set_lookat(center.tolist())
                view.set_front([0.0, 0.0, -1.0])
                view.set_up([0.0, -1.0, 0.0])
                view.set_zoom(0.25)
                self.ready = True

            self.frames += 1
            self.last_points = len(points)
            if self.frames == 1 or self.frames % 25 == 0:
                self.get_logger().info(f"frame={self.frames} points={len(points)}")
        except Exception as exc:
            self.get_logger().error(f"Open3D callback failed: {type(exc).__name__}: {exc}")

    def spin_view(self) -> None:
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_view(self) -> None:
        self.vis.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/carla/hero/lidar/point_cloud")
    args = parser.parse_args()

    rclpy.init()
    viewer = LiveViewer(args.topic)
    try:
        viewer.spin_view()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close_view()
        viewer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

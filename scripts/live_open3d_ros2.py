#!/usr/bin/env python3
"""Live Open3D viewer for the real CARLA/FFEM ROS 2 point clouds.

The viewer shows the semantic FFEM output when available, with the raw CARLA
LiDAR stream as a fallback and the moving-point stream as a second overlay.
"""
from __future__ import annotations

import argparse
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import PointCloud2
except ImportError as exc:
    raise SystemExit("ROS 2 Python packages are not available; source /opt/ros/humble/setup.bash") from exc

try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit("Open3D is not installed; install it in the active Python environment") from exc

from ffem.ros2.pointcloud2_codec import decode_pointcloud2

PALETTE = np.array(
    [
        [0.45, 0.45, 0.45],
        [0.20, 0.75, 0.35],
        [0.15, 0.80, 0.25],
        [0.85, 0.55, 0.20],
        [0.95, 0.15, 0.10],
        [0.95, 0.20, 0.75],
        [1.00, 0.75, 0.10],
    ], dtype=np.float64
)


class LiveViewer(Node):
    def __init__(self, raw_topic: str, semantic_topic: str, moving_topic: str):
        super().__init__("ffem_open3d_viewer")
        self.raw_topic = raw_topic
        self.semantic_topic = semantic_topic
        self.moving_topic = moving_topic

        self.vis = o3d.visualization.Visualizer()
        if not self.vis.create_window("FFEM Live LiDAR", 1280, 720):
            raise RuntimeError("Open3D failed to create a window")

        options = self.vis.get_render_option()
        options.background_color = np.array([0.02, 0.02, 0.02])
        options.point_size = 2.0

        self.raw_cloud = o3d.geometry.PointCloud()
        self.semantic_cloud = o3d.geometry.PointCloud()
        self.moving_cloud = o3d.geometry.PointCloud()
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)

        self.raw_added = False
        self.semantic_added = False
        self.moving_added = False
        self.axis_added = False
        self.ready = False
        self.raw_frames = 0
        self.semantic_frames = 0
        self.moving_frames = 0
        self.last_points = 0

        qos = QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.raw_sub = self.create_subscription(PointCloud2, raw_topic, self.raw_callback, qos)
        self.semantic_sub = self.create_subscription(PointCloud2, semantic_topic, self.semantic_callback, qos)
        self.moving_sub = self.create_subscription(PointCloud2, moving_topic, self.moving_callback, qos)

        self.get_logger().info(f"Raw LiDAR: {raw_topic}")
        self.get_logger().info(f"Semantic FFEM: {semantic_topic}")
        self.get_logger().info(f"Moving FFEM: {moving_topic}")

    @staticmethod
    def _decode(msg):
        decoded = decode_pointcloud2(msg, remove_invalid=True)
        points = np.asarray(decoded.points, dtype=np.float64).reshape(-1, 3)
        return decoded, points

    def _add_cloud_once(self, cloud, added_attr: str):
        if not getattr(self, added_attr):
            self.vis.add_geometry(cloud)
            setattr(self, added_attr, True)

    def _reset_camera(self, cloud):
        bbox = cloud.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = float(np.linalg.norm(bbox.get_extent()))
        if not np.isfinite(extent) or extent < 1.0:
            extent = 30.0
        view = self.vis.get_view_control()
        view.set_lookat(center.tolist())
        view.set_front([1.0, -1.0, 0.35])
        view.set_up([0.0, 0.0, 1.0])
        view.set_zoom(float(np.clip(28.0 / extent, 0.15, 1.0)))
        self.ready = True

    def raw_callback(self, msg: PointCloud2) -> None:
        try:
            _, points = self._decode(msg)
            if len(points) == 0:
                return
            self.raw_cloud.points = o3d.utility.Vector3dVector(points)
            self.raw_cloud.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[0.25, 0.65, 1.0]]), (len(points), 1))
            )
            self._add_cloud_once(self.raw_cloud, "raw_added")
            self.vis.update_geometry(self.raw_cloud)
            if not self.axis_added:
                self.vis.add_geometry(self.axis)
                self.axis_added = True
            if not self.ready:
                self._reset_camera(self.raw_cloud)
            self.raw_frames += 1
            self.last_points = len(points)
            if self.raw_frames == 1 or self.raw_frames % 25 == 0:
                self.get_logger().info(f"raw frame={self.raw_frames} points={len(points)}")
        except Exception as exc:
            self.get_logger().error(f"Raw LiDAR callback failed: {type(exc).__name__}: {exc}")

    def semantic_callback(self, msg: PointCloud2) -> None:
        try:
            decoded, points = self._decode(msg)
            if len(points) == 0:
                return
            labels = np.rint(np.asarray(decoded.intensity if decoded.intensity is not None else np.zeros(len(points))))
            labels = np.clip(labels.astype(np.int32), 0, len(PALETTE) - 1)
            self.semantic_cloud.points = o3d.utility.Vector3dVector(points)
            self.semantic_cloud.colors = o3d.utility.Vector3dVector(PALETTE[labels])
            self._add_cloud_once(self.semantic_cloud, "semantic_added")
            self.vis.update_geometry(self.semantic_cloud)
            self.semantic_frames += 1
            if self.semantic_frames == 1 or self.semantic_frames % 25 == 0:
                self.get_logger().info(f"semantic frame={self.semantic_frames} points={len(points)}")
        except Exception as exc:
            self.get_logger().error(f"Semantic callback failed: {type(exc).__name__}: {exc}")

    def moving_callback(self, msg: PointCloud2) -> None:
        try:
            _, points = self._decode(msg)
            if len(points) == 0:
                self.moving_cloud.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            else:
                self.moving_cloud.points = o3d.utility.Vector3dVector(points)
                self.moving_cloud.colors = o3d.utility.Vector3dVector(
                    np.tile(np.array([[1.0, 0.1, 0.1]]), (len(points), 1))
                )
            self._add_cloud_once(self.moving_cloud, "moving_added")
            self.vis.update_geometry(self.moving_cloud)
            self.moving_frames += 1
            if self.moving_frames == 1 or self.moving_frames % 25 == 0:
                self.get_logger().info(f"moving frame={self.moving_frames} points={len(points)}")
        except Exception as exc:
            self.get_logger().error(f"Moving callback failed: {type(exc).__name__}: {exc}")

    def spin_view(self) -> None:
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.005)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_view(self) -> None:
        self.vis.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/carla/hero/lidar/point_cloud")
    parser.add_argument("--semantic-topic", default="/ffem_mapper/map/semantic")
    parser.add_argument("--moving-topic", default="/ffem_mapper/map/moving_points")
    args = parser.parse_args()

    rclpy.init()
    viewer = LiveViewer(args.topic, args.semantic_topic, args.moving_topic)
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

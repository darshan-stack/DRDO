#!/usr/bin/env python3
"""Throttled, bounded live Open3D viewer for ROS 2 LiDAR/FFEM clouds."""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import PointCloud2
except ImportError as exc:
    raise SystemExit(
        "ROS 2 Python packages are not available; source /opt/ros/humble/setup.bash"
    ) from exc

if os.environ.get("FFEM_OPEN3D_SOFTWARE_RENDERING", "1") == "1":
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit(
        "Open3D is not installed; install it in the active Python environment"
    ) from exc

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
    ],
    dtype=np.float64,
)


def bounded_points(points: np.ndarray, limit: int) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(p) <= limit:
        return p
    idx = np.linspace(0, len(p) - 1, limit, dtype=np.int64)
    return p[idx]


class LiveViewer(Node):
    def __init__(
        self,
        raw_topic: str,
        semantic_topic: str,
        moving_topic: str,
        max_points: int,
        max_hz: float,
    ):
        super().__init__("ffem_open3d_viewer")
        if max_points <= 0 or max_hz <= 0:
            raise ValueError("max_points and max_hz must be positive")

        self.max_points = int(max_points)
        self.render_period = 1.0 / float(max_hz)
        self.next_render = time.monotonic()

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
        self.camera_ready = False

        self.pending_raw: np.ndarray | None = None
        self.pending_semantic: tuple[np.ndarray, np.ndarray] | None = None
        self.pending_moving: np.ndarray | None = None

        qos = QoSProfile(
            depth=2,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.raw_sub = self.create_subscription(
            PointCloud2, raw_topic, self.raw_callback, qos
        )
        self.semantic_sub = self.create_subscription(
            PointCloud2, semantic_topic, self.semantic_callback, qos
        )
        self.moving_sub = self.create_subscription(
            PointCloud2, moving_topic, self.moving_callback, qos
        )

        self.get_logger().info(
            f"Open3D viewer | raw={raw_topic} semantic={semantic_topic} "
            f"moving={moving_topic} max_points={max_points} max_hz={max_hz} "
            f"software_rendering={os.environ.get('LIBGL_ALWAYS_SOFTWARE', '0')}"
        )

    @staticmethod
    def _decode(msg: PointCloud2) -> tuple[object, np.ndarray]:
        decoded = decode_pointcloud2(msg, remove_invalid=True)
        points = np.asarray(decoded.points, dtype=np.float64).reshape(-1, 3)
        return decoded, points

    def raw_callback(self, msg: PointCloud2) -> None:
        try:
            _, points = self._decode(msg)
            self.pending_raw = bounded_points(points, self.max_points)
        except Exception as exc:
            self.get_logger().error(
                f"Raw LiDAR callback failed: {type(exc).__name__}: {exc}"
            )

    def semantic_callback(self, msg: PointCloud2) -> None:
        try:
            decoded, points = self._decode(msg)
            if len(points) == 0:
                return
            selected = bounded_points(points, self.max_points)
            intensity = (
                np.asarray(decoded.intensity, dtype=np.float64).reshape(-1)
                if decoded.intensity is not None
                else np.zeros(len(decoded.points), dtype=np.float64)
            )
            if len(intensity) != len(decoded.points):
                return
            if len(decoded.points) > self.max_points:
                idx = np.linspace(
                    0, len(decoded.points) - 1, self.max_points, dtype=np.int64
                )
                intensity = intensity[idx]
            labels = np.clip(np.rint(intensity).astype(np.int32), 0, len(PALETTE) - 1)
            self.pending_semantic = (selected, PALETTE[labels])
        except Exception as exc:
            self.get_logger().error(
                f"Semantic callback failed: {type(exc).__name__}: {exc}"
            )

    def moving_callback(self, msg: PointCloud2) -> None:
        try:
            _, points = self._decode(msg)
            self.pending_moving = bounded_points(points, self.max_points)
        except Exception as exc:
            self.get_logger().error(
                f"Moving callback failed: {type(exc).__name__}: {exc}"
            )

    def _add_once(self, cloud: object, flag: str) -> None:
        if not getattr(self, flag):
            self.vis.add_geometry(cloud)
            setattr(self, flag, True)

    def _reset_camera(self, cloud: object) -> None:
        if self.camera_ready or len(cloud.points) == 0:
            return
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
        self.camera_ready = True

    def _render(self) -> None:
        now = time.monotonic()
        if now < self.next_render:
            return
        self.next_render = now + self.render_period

        if self.pending_raw is not None:
            points = self.pending_raw
            self.raw_cloud.points = o3d.utility.Vector3dVector(points)
            self.raw_cloud.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[0.25, 0.65, 1.0]]), (len(points), 1))
            )
            self._add_once(self.raw_cloud, "raw_added")
            self.vis.update_geometry(self.raw_cloud)
            if not self.axis_added:
                self.vis.add_geometry(self.axis)
                self.axis_added = True
            self._reset_camera(self.raw_cloud)

        if self.pending_semantic is not None:
            points, colors = self.pending_semantic
            self.semantic_cloud.points = o3d.utility.Vector3dVector(points)
            self.semantic_cloud.colors = o3d.utility.Vector3dVector(colors)
            self._add_once(self.semantic_cloud, "semantic_added")
            self.vis.update_geometry(self.semantic_cloud)

        if self.pending_moving is not None:
            points = self.pending_moving
            self.moving_cloud.points = o3d.utility.Vector3dVector(points)
            self.moving_cloud.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[1.0, 0.1, 0.1]]), (len(points), 1))
            )
            self._add_once(self.moving_cloud, "moving_added")
            self.vis.update_geometry(self.moving_cloud)

        self.vis.poll_events()
        self.vis.update_renderer()

    def spin_view(self) -> None:
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.005)
            self._render()

    def close_view(self) -> None:
        self.vis.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/carla/hero/lidar/point_cloud")
    parser.add_argument("--semantic-topic", default="/ffem_mapper/map/semantic")
    parser.add_argument("--moving-topic", default="/ffem_mapper/map/moving_points")
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--max-hz", type=float, default=5.0)
    parser.add_argument(
        "--software-rendering",
        action="store_true",
        help="Force Mesa/llvmpipe software OpenGL before importing Open3D.",
    )
    args = parser.parse_args()
    if args.software_rendering:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

    rclpy.init()
    viewer = LiveViewer(
        args.topic,
        args.semantic_topic,
        args.moving_topic,
        args.max_points,
        args.max_hz,
    )
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

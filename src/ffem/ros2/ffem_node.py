"""Live ROS 2 adapter: PointCloud2 -> FFEM -> semantic/2.5D outputs/Rerun/RViz."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Float32MultiArray, String
    from visualization_msgs.msg import Marker, MarkerArray
    from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from ffem.pipeline import FFEMPipeline, FFEMConfig
from ffem.perception.factory import build_segmenter
from ffem.ros2.pointcloud2_codec import decode_pointcloud2, encode_pointcloud2
from ffem.ros2.transforms import transform_points, transform_from_ros_transform

try:
    import rerun as rr
except ImportError:  # pragma: no cover
    rr = None

SEMANTIC_PALETTE = np.array(
    [
        [90, 90, 90],
        [70, 140, 220],
        [70, 210, 100],
        [180, 120, 60],
        [230, 70, 60],
        [220, 80, 180],
        [245, 190, 40],
    ], dtype=np.uint8
)

if ROS_AVAILABLE:

    class FFEMNode(Node):
        def __init__(self):
            super().__init__("ffem_mapper")
            self.declare_parameter("input_topic", "/carla/hero/lidar/point_cloud")
            self.declare_parameter("map_frame", "lidar")
            self.declare_parameter("use_tf", False)
            self.declare_parameter("queue_depth", 5)
            self.declare_parameter("max_points_per_frame", 150000)
            self.declare_parameter("enable_rerun", True)
            self.declare_parameter("recording", "outputs/carla_ffem.rrd")
            self.declare_parameter("model_backend", "auto")
            self.declare_parameter("checkpoint", "")
            self.declare_parameter("range_height", 32)
            self.declare_parameter("range_width", 1024)
            self.declare_parameter("max_range", 80.0)
            self.declare_parameter("base_cell_size", 1.0)
            self.declare_parameter("finest_cell_size", 0.25)
            self.declare_parameter("max_level", 2)
            self.declare_parameter("max_active_cells", 20000)
            self.declare_parameter("max_topology_changes", 32)
            self.declare_parameter("queue_depth", 5)
            self.declare_parameter("map_topic", "~/map/elevation")
            self.declare_parameter("semantic_topic", "~/map/semantic")
            self.declare_parameter("moving_topic", "~/map/moving_points")
            self.declare_parameter("adaptive_topic", "~/map/adaptive_cells")
            self.declare_parameter("traversability_topic", "~/map/traversability")
            self.declare_parameter("uncertainty_topic", "~/map/uncertainty")
            self.declare_parameter("attention_topic", "~/map/attention")
            self.declare_parameter("metrics_topic", "~/metrics")
            self.declare_parameter("events_topic", "~/refinement_events")
            self.declare_parameter("tracks_topic", "~/tracks")
            self.declare_parameter("planning_risk_topic", "~/planning/risk")

            cfg = FFEMConfig(
                base_cell_size=float(self.get_parameter("base_cell_size").value),
                finest_cell_size=float(self.get_parameter("finest_cell_size").value),
                max_level=int(self.get_parameter("max_level").value),
                max_active_cells=int(self.get_parameter("max_active_cells").value),
                max_topology_changes=int(self.get_parameter("max_topology_changes").value),
            )

            backend = str(self.get_parameter("model_backend").value)
            checkpoint = str(self.get_parameter("checkpoint").value)
            segmenter, selected_checkpoint = build_segmenter(
                backend, checkpoint, cfg.num_classes,
                int(self.get_parameter("range_height").value),
                int(self.get_parameter("range_width").value),
                float(self.get_parameter("max_range").value),
            )
            self.pipeline = FFEMPipeline(cfg, segmenter=segmenter)
            self.frame = 0
            self.map_frame = str(self.get_parameter("map_frame").value)
            self.max_points = int(self.get_parameter("max_points_per_frame").value)
            self.use_tf = bool(self.get_parameter("use_tf").value)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

            qos = QoSProfile(
                depth=int(self.get_parameter("queue_depth").value),
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self.sub = self.create_subscription(
                PointCloud2,
                str(self.get_parameter("input_topic").value), self.callback, qos
            )
            self.map_pub = self.create_publisher(PointCloud2, str(self.get_parameter("map_topic").value), 5)
            self.semantic_pub = self.create_publisher(PointCloud2, str(self.get_parameter("semantic_topic").value), 5)
            self.moving_pub = self.create_publisher(PointCloud2, str(self.get_parameter("moving_topic").value), 5)
            self.adaptive_pub = self.create_publisher(PointCloud2, str(self.get_parameter("adaptive_topic").value), 5)
            self.traversability_pub = self.create_publisher(PointCloud2, str(self.get_parameter("traversability_topic").value), 5)
            self.uncertainty_pub = self.create_publisher(PointCloud2, str(self.get_parameter("uncertainty_topic").value), 5)
            self.attention_pub = self.create_publisher(PointCloud2, str(self.get_parameter("attention_topic").value), 5)
            self.metrics_pub = self.create_publisher(Float32MultiArray, str(self.get_parameter("metrics_topic").value), 5)
            self.events_pub = self.create_publisher(String, str(self.get_parameter("events_topic").value), 5)
            self.tracks_pub = self.create_publisher(MarkerArray, str(self.get_parameter("tracks_topic").value), 5)
            self.planning_risk_pub = self.create_publisher(Float32MultiArray, str(self.get_parameter("planning_risk_topic").value), 5)

            self.rerun_enabled = bool(self.get_parameter("enable_rerun").value) and rr is not None
            if self.rerun_enabled:
                rr.init("ffem-ros2", spawn=True)

            ckpt_text = selected_checkpoint if selected_checkpoint else "none (fallback)"
            self.get_logger().info(
                f"FFEM ready | input={self.get_parameter('input_topic').value} | backend={backend} | "
                f"checkpoint={ckpt_text} | map_frame={self.map_frame} | TF={self.use_tf} | Rerun={self.rerun_enabled}"
            )

        def _lookup(self, msg):
            if not self.use_tf or msg.header.frame_id == self.map_frame:
                return np.eye(4, dtype=np.float64)
            try:
                transform = self.tf_buffer.lookup_transform(self.map_frame, msg.header.frame_id, msg.header.stamp)
                return transform_from_ros_transform(transform.transform)
            except (LookupException, ConnectivityException, ExtrapolationException) as exc:
                self.get_logger().warning(
                    f"TF unavailable {msg.header.frame_id}->{self.map_frame}: {exc}",
                    throttle_duration_sec=5.0,
                )
                return None

        def callback(self, msg):
            try:
                decoded = decode_pointcloud2(msg, remove_invalid=True)
                if not len(decoded.points):
                    self.get_logger().warning("Received an empty PointCloud2", throttle_duration_sec=5.0)
                    return
                if len(decoded.points) > self.max_points:
                    idx = np.linspace(0, len(decoded.points) - 1, self.max_points, dtype=int)
                else:
                    idx = slice(None)
                points = decoded.points[idx]
                intensity = None if decoded.intensity is None else decoded.intensity[idx]
                matrix = self._lookup(msg)
                if matrix is None:
                    return
                points = transform_points(points, matrix)
                result = self.pipeline.process_points(points, intensity=intensity, frame=self.frame)
                self._publish(result, msg.header.stamp)
                self._log_rerun(result)
                self.frame += 1
                if self.frame == 1 or self.frame % 25 == 0:
                    stats = result["stats"]
                    labels = np.argmax(result["semantic_probs"], axis=1)
                    counts = np.bincount(labels, minlength=self.pipeline.config.num_classes)
                    self.get_logger().info(
                        "frame=%d points=%d active_cells=%d moving=%d tracks=%d total_ms=%.2f classes=%s"
                        % (self.frame, len(points), int(stats["active_cells"]), int(stats["moving_points"]),
                           int(stats.get("tracks", 0)), float(stats["total_ms"]), counts.tolist())
                    )
            except Exception as exc:
                self.get_logger().error(f"FFEM callback failed: {type(exc).__name__}: {exc}")

        @staticmethod
        def _semantic_rgb(labels):
            labels = np.clip(np.asarray(labels, dtype=np.int32), 0, len(SEMANTIC_PALETTE) - 1)
            return SEMANTIC_PALETTE[labels]

        def _publish_tracks(self, tracks):
            array = MarkerArray()
            for idx, track in enumerate(tracks):
                marker = Marker()
                marker.header.frame_id = self.map_frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "ffem_tracks"
                marker.id = int(track.track_id)
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = float(track.center[0])
                marker.pose.position.y = float(track.center[1])
                marker.pose.position.z = float(track.center[2])
                marker.scale.x = max(0.3, float(track.size[0]))
                marker.scale.y = max(0.3, float(track.size[1]))
                marker.scale.z = max(0.3, float(track.size[2]))
                marker.color.r = 1.0
                marker.color.g = 0.12
                marker.color.b = 0.12
                marker.color.a = 0.85
                array.markers.append(marker)

                text = Marker()
                text.header = marker.header
                text.ns = "ffem_track_labels"
                text.id = 10000 + int(track.track_id)
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose = marker.pose
                text.pose.position.z += marker.scale.z * 0.8
                text.scale.z = 0.45
                text.color.r = 1.0
                text.color.g = 1.0
                text.color.b = 1.0
                text.color.a = 1.0
                text.text = f"T{track.track_id}"
                array.markers.append(text)
            self.tracks_pub.publish(array)

        def _publish(self, result, stamp):
            map_points, map_colors, levels = self.pipeline.mapping.arrays()
            cell_points, traversability, uncertainty, attention, cell_levels = self.pipeline.mapping.diagnostics()

            self.map_pub.publish(
                encode_pointcloud2(
                    map_points, frame_id=self.map_frame, stamp=stamp,
                    rgb=None,
                    intensity=map_points[:, 2] if len(map_points) else None,
                )
            )

            labels = np.argmax(result["semantic_probs"], axis=1).astype(np.int32)
            semantic_rgb = self._semantic_rgb(labels)
            self.semantic_pub.publish(
                encode_pointcloud2(result["points"], frame_id=self.map_frame, stamp=stamp, rgb=semantic_rgb)
            )

            moving = result["points"][result["moving"]]
            moving_rgb = np.tile(np.array([[255, 45, 45]], dtype=np.uint8), (len(moving), 1))
            self.moving_pub.publish(encode_pointcloud2(moving, frame_id=self.map_frame, stamp=stamp, rgb=moving_rgb))

            self.adaptive_pub.publish(
                encode_pointcloud2(cell_points, frame_id=self.map_frame, stamp=stamp,
                                   intensity=cell_levels.astype(np.float32))
            )
            self.traversability_pub.publish(
                encode_pointcloud2(cell_points, frame_id=self.map_frame, stamp=stamp, intensity=traversability)
            )
            self.uncertainty_pub.publish(
                encode_pointcloud2(cell_points, frame_id=self.map_frame, stamp=stamp, intensity=uncertainty)
            )
            self.attention_pub.publish(
                encode_pointcloud2(cell_points, frame_id=self.map_frame, stamp=stamp, intensity=attention)
            )
            self._publish_tracks(result.get("tracks", []))

            stats = result["stats"]
            msg = Float32MultiArray()
            msg.data = [
                float(stats["total_ms"]), float(stats["map_ms"]), float(stats["active_cells"]),
                float(stats["topology_changes"]), float(stats["points"]), float(stats["moving_points"]),
                float(stats.get("tracks", 0)),
            ]
            self.metrics_pub.publish(msg)

            planning = Float32MultiArray()
            planning.data = [
                float(np.mean(1.0 - traversability)) if len(traversability) else 0.0,
                float(np.max(uncertainty)) if len(uncertainty) else 0.0,
                float(np.max(attention)) if len(attention) else 0.0,
                float(np.mean(attention)) if len(attention) else 0.0,
                float(stats["moving_points"]),
                float(stats.get("tracks", 0)),
            ]
            self.planning_risk_pub.publish(planning)

            if self.pipeline.mapping.events:
                event = String()
                event.data = json.dumps(self.pipeline.mapping.events[-1])
                self.events_pub.publish(event)

        def _log_rerun(self, result):
            if not self.rerun_enabled:
                return
            stats = result["stats"]
            map_points, map_colors, levels = self.pipeline.mapping.arrays()
            labels = np.argmax(result["semantic_probs"], axis=1)
            colors = SEMANTIC_PALETTE[np.clip(labels, 0, len(SEMANTIC_PALETTE) - 1)]
            rr.set_time("frame", sequence=self.frame)
            rr.log("world/lidar/raw", rr.Points3D(result["points"]))
            rr.log("world/lidar/semantic", rr.Points3D(result["points"], colors=colors))
            rr.log("world/dynamics/moving_points", rr.Points3D(result["points"][result["moving"]]))
            rr.log("world/dynamics/tracks", rr.Points3D(np.array([t.center for t in result.get("tracks", [])], dtype=np.float32)) if result.get("tracks") else rr.Points3D(np.empty((0, 3), dtype=np.float32)))
            if len(map_points):
                rr.log("world/map/elevation", rr.Points3D(map_points, colors=map_colors))
                rr.log("world/map/adaptive_cells", rr.Points3D(map_points, radii=0.04 + 0.03 * levels, colors=map_colors))
            if self.pipeline.mapping.events:
                events = self.pipeline.mapping.events[-20:]
                event_pts = np.array([[e["cell"][1], e["cell"][2], 0.05] for e in events], dtype=np.float32)
                rr.log("world/adaptation/refinement_events", rr.Points3D(event_pts, radii=0.08))
            rr.log("metrics/latency/total_ms", rr.Scalars([stats["total_ms"]]))
            rr.log("metrics/latency/map_ms", rr.Scalars([stats["map_ms"]]))
            rr.log("metrics/memory/active_cells", rr.Scalars([stats["active_cells"]]))
            rr.log("metrics/topology_changes", rr.Scalars([stats["topology_changes"]]))
            rr.log("metrics/points", rr.Scalars([stats["points"]]))
            rr.log("metrics/moving_points", rr.Scalars([stats["moving_points"]]))

    def main(args=None):
        rclpy.init(args=args)
        node = FFEMNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            if node.rerun_enabled:
                recording = Path(str(node.get_parameter("recording").value))
                recording.parent.mkdir(parents=True, exist_ok=True)
                rr.save(str(recording))
                node.get_logger().info(f"Saved Rerun recording to {recording}")
            node.destroy_node()
            rclpy.shutdown()

else:
    class FFEMNode:
        def __init__(self):
            raise RuntimeError("ROS 2 is not installed.")
    def main(args=None):
        raise RuntimeError("ROS 2 is not installed.")

if __name__ == "__main__":
    main()

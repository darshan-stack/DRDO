#!/usr/bin/env python3
"""Publish the CARLA hero LiDAR pose as a ROS 2 TF transform.

The CARLA world is treated as the FFEM map frame. The broadcaster tracks the
actual ray-cast LiDAR actor attached to the hero vehicle, so ego-motion
compensation uses the same sensor pose as the incoming PointCloud2 messages.
"""
from __future__ import annotations

import math
import time

import carla
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def quaternion_from_euler(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class CarlaLidarTF(Node):
    def __init__(self, host: str, port: int, map_frame: str, sensor_frame: str):
        super().__init__("carla_lidar_tf_broadcaster")
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.map_frame = map_frame
        self.sensor_frame = sensor_frame
        self.sensor = None
        self.hero = None
        self.br = TransformBroadcaster(self)
        self.last_log = 0.0
        self.timer = self.create_timer(0.02, self.tick)
        self.get_logger().info(
            f"CARLA TF broadcaster ready | map={map_frame} sensor={sensor_frame}"
        )

    def _find_actors(self):
        vehicles = list(self.world.get_actors().filter("vehicle.*"))
        self.hero = next(
            (v for v in vehicles if v.attributes.get("role_name") == "hero"),
            vehicles[0] if vehicles else None,
        )
        if self.hero is None:
            self.sensor = None
            return
        sensors = list(self.world.get_actors().filter("sensor.lidar.ray_cast"))
        self.sensor = next(
            (s for s in sensors if s.parent is not None and s.parent.id == self.hero.id),
            None,
        )

    def tick(self):
        try:
            if self.hero is None or not self.hero.is_alive:
                self._find_actors()
            if self.sensor is None or not self.sensor.is_alive:
                self._find_actors()
            if self.sensor is None:
                now = time.monotonic()
                if now - self.last_log > 2.0:
                    self.get_logger().warning(
                        "Waiting for a ray-cast LiDAR sensor attached to CARLA hero"
                    )
                    self.last_log = now
                return

            tf = self.sensor.get_transform()
            msg = TransformStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.map_frame
            msg.child_frame_id = self.sensor_frame
            msg.transform.translation.x = float(tf.location.x)
            msg.transform.translation.y = float(tf.location.y)
            msg.transform.translation.z = float(tf.location.z)

            roll = math.radians(tf.rotation.roll)
            pitch = math.radians(tf.rotation.pitch)
            yaw = math.radians(tf.rotation.yaw)
            qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
            msg.transform.rotation.x = qx
            msg.transform.rotation.y = qy
            msg.transform.rotation.z = qz
            msg.transform.rotation.w = qw
            self.br.sendTransform(msg)
        except Exception as exc:
            self.get_logger().error(
                f"TF broadcast failed: {type(exc).__name__}: {exc}"
            )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--sensor-frame", default="lidar")
    args = parser.parse_args()

    rclpy.init()
    node = CarlaLidarTF(
        args.host,
        args.port,
        args.map_frame,
        args.sensor_frame,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Follow the FFEM local path with a lightweight CARLA controller.

The FFEM planner publishes a short path in the configured map frame. This node
converts map/world coordinates into the hero vehicle frame before applying
CARLA control. It is a demonstration controller, not a certified safety system.
"""
from __future__ import annotations

import math
import time

import numpy as np

import carla
import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node


class FFEMPathFollower(Node):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        target_speed: float = 5.0,
        stale_timeout: float = 0.5,
    ):
        super().__init__("ffem_path_follower")
        self.target_speed = float(target_speed)
        self.stale_timeout = float(stale_timeout)
        self.path = None
        self.path_received_at = 0.0

        self.client = carla.Client(host, int(port))
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        vehicles = list(self.world.get_actors().filter("vehicle.*"))
        self.vehicle = next(
            (v for v in vehicles if v.attributes.get("role_name") == "hero"),
            None,
        )
        if self.vehicle is None:
            raise RuntimeError("No CARLA hero vehicle found")

        self.sub = self.create_subscription(
            Path,
            "/ffem_mapper/planning/path",
            self.path_callback,
            10,
        )
        self.timer = self.create_timer(0.05, self.control_loop)
        self.last_log = 0.0

        self.get_logger().info(
            f"FFEM path follower ready | vehicle={self.vehicle.id} "
            f"| target_speed={self.target_speed:.1f} m/s"
        )

    def path_callback(self, msg: Path):
        self.path = msg
        self.path_received_at = time.monotonic()

    def _speed(self):
        velocity = self.vehicle.get_velocity()
        return float(
            math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        )

    def _world_to_vehicle(self, x: float, y: float):
        transform = self.vehicle.get_transform()
        dx = x - float(transform.location.x)
        dy = y - float(transform.location.y)
        yaw = math.radians(float(transform.rotation.yaw))
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return local_x, local_y

    def _target_xy(self, pose):
        frame = (self.path.header.frame_id or "").lstrip("/")
        x = float(pose.position.x)
        y = float(pose.position.y)
        if frame in ("map", "world", "odom"):
            return self._world_to_vehicle(x, y)
        # FFEM can also operate with map_frame=lidar, where path coordinates are
        # already robot-centric.
        return x, y

    def _control_from_path(self):
        control = carla.VehicleControl()
        control.hand_brake = False
        control.reverse = False

        stale = (
            self.path is None
            or (time.monotonic() - self.path_received_at) > self.stale_timeout
        )
        if stale or len(self.path.poses) < 2:
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 0.8
            return control, None

        idx = min(10, len(self.path.poses) - 1)
        x, y = self._target_xy(self.path.poses[idx].pose)
        distance = max(math.hypot(x, y), 0.1)
        heading_error = math.atan2(y, max(x, 0.1))

        steer = -1.20 * heading_error
        steer = float(np.clip(steer, -1.0, 1.0))

        speed = self._speed()
        speed_error = self.target_speed - speed
        throttle = float(np.clip(0.25 * speed_error, 0.0, 0.45))
        brake = float(np.clip(-0.40 * speed_error, 0.0, 0.55))

        if abs(heading_error) > 0.35:
            throttle = min(throttle, 0.20)
        if distance < 2.0:
            throttle = min(throttle, 0.10)

        control.throttle = throttle
        control.brake = brake
        control.steer = steer
        return control, (speed, heading_error, x, y)

    def control_loop(self):
        try:
            if self.vehicle is None or not self.vehicle.is_alive:
                return
            control, info = self._control_from_path()
            self.vehicle.apply_control(control)
            now = time.monotonic()
            if info is not None and now - self.last_log > 1.0:
                speed, heading_error, tx, ty = info
                self.get_logger().info(
                    f"path-follow | speed={speed:.2f} m/s "
                    f"target=({tx:.2f},{ty:.2f}) "
                    f"heading_error={heading_error:.2f} "
                    f"steer={control.steer:.2f} "
                    f"throttle={control.throttle:.2f} "
                    f"brake={control.brake:.2f}"
                )
                self.last_log = now
        except Exception as exc:
            self.get_logger().error(
                f"controller error: {type(exc).__name__}: {exc}"
            )

    def stop(self):
        try:
            self.vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    steer=0.0,
                    hand_brake=True,
                )
            )
        except Exception:
            pass


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--target-speed", type=float, default=5.0)
    parser.add_argument("--stale-timeout", type=float, default=0.5)
    args = parser.parse_args()

    rclpy.init()
    node = FFEMPathFollower(
        host=args.host,
        port=args.port,
        target_speed=args.target_speed,
        stale_timeout=args.stale_timeout,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

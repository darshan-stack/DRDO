#!/usr/bin/env python3
"""Follow the FFEM local path with a lightweight CARLA pure-pursuit controller.

The FFEM planner publishes a robot-centric nav_msgs/Path. This node converts
that short local path into steering/throttle/brake commands for the CARLA hero
vehicle. It is intentionally a demonstration controller, not a safety-rated
vehicle controller.
"""
from __future__ import annotations

import math
import time

import numpy as np

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise SystemExit("CARLA Python API is not available; use ~/CARLA/carla_env/bin/python3") from exc

import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node


class FFEMPathFollower(Node):
    def __init__(self, host="127.0.0.1", port=2000, target_speed=7.0):
        super().__init__("ffem_path_follower")
        self.target_speed = float(target_speed)
        self.path = None
        self.client = carla.Client(host, int(port))
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        vehicles = list(self.world.get_actors().filter("vehicle.*"))
        self.vehicle = next((v for v in vehicles if v.attributes.get("role_name") == "hero"), None)
        if self.vehicle is None:
            raise RuntimeError("No CARLA hero vehicle found")
        self.sub = self.create_subscription(Path, "/ffem_mapper/planning/path", self.path_callback, 10)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.last_log = 0.0
        self.get_logger().info(
            f"FFEM path follower ready | vehicle={self.vehicle.id} | target_speed={self.target_speed:.1f} m/s"
        )

    def path_callback(self, msg: Path):
        self.path = msg

    def _speed(self):
        velocity = self.vehicle.get_velocity()
        return float(math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))

    def _control_from_path(self):
        control = carla.VehicleControl()
        control.hand_brake = False
        control.reverse = False
        if self.path is None or len(self.path.poses) < 2:
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 0.8
            return control, None

        # The path is in the vehicle/lidar frame: x forward, y lateral.
        idx = min(10, len(self.path.poses) - 1)
        target = self.path.poses[idx].pose.position
        x = float(target.x)
        y = float(target.y)
        distance = max(math.hypot(x, y), 0.1)
        heading_error = math.atan2(y, max(x, 0.1))

        # CARLA's steering sign is opposite the +y lateral convention used by
        # the local FFEM path, hence the negative sign here.
        steer = -1.20 * heading_error
        steer = float(np.clip(steer, -1.0, 1.0))

        speed = self._speed()
        speed_error = self.target_speed - speed
        throttle = float(np.clip(0.25 * speed_error, 0.0, 0.55))
        brake = float(np.clip(-0.40 * speed_error, 0.0, 0.45))

        if abs(heading_error) > 0.35:
            throttle = min(throttle, 0.25)
        if distance < 2.0:
            throttle = min(throttle, 0.15)

        control.throttle = throttle
        control.brake = brake
        control.steer = steer
        return control, (speed, heading_error, target.x, target.y)

    def control_loop(self):
        try:
            if not self.world or not self.vehicle.is_alive:
                return
            control, info = self._control_from_path()
            self.vehicle.apply_control(control)
            now = time.monotonic()
            if info is not None and now - self.last_log > 1.0:
                speed, heading_error, tx, ty = info
                self.get_logger().info(
                    f"path-follow | speed={speed:.2f} m/s target=({tx:.2f},{ty:.2f}) "
                    f"heading_error={heading_error:.2f} steer={control.steer:.2f} "
                    f"throttle={control.throttle:.2f} brake={control.brake:.2f}"
                )
                self.last_log = now
        except Exception as exc:
            self.get_logger().error(f"controller error: {type(exc).__name__}: {exc}")


def main():
    rclpy.init()
    node = FFEMPathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Safer autonomous CARLA demo driver for the native ROS2 hero vehicle.

Uses CARLA's client-side BehaviorAgent when available, with a cautious profile
and an additional obstacle sensor/emergency-brake guard. The native ROS2 stack
still owns the hero vehicle and LiDAR; this helper only controls the vehicle.

This is a research/demo autonomy controller, not a certified autonomous-driving
or safety system.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import carla


# CARLA's example agents live in PythonAPI/carla/agents and are not always on
# PYTHONPATH when the packaged client wheel is used.
CARLA_ROOT = Path.home() / "CARLA"
AGENTS_ROOT = CARLA_ROOT / "PythonAPI"
if str(AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTS_ROOT))

try:
    from agents.navigation.behavior_agent import BehaviorAgent
    AGENT_AVAILABLE = True
except ImportError:
    BehaviorAgent = None
    AGENT_AVAILABLE = False


def update_spectator(
    world: carla.World,
    vehicle: carla.Vehicle,
    distance: float,
    height: float,
) -> None:
    transform = vehicle.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    cam_x = transform.location.x - distance * math.cos(yaw)
    cam_y = transform.location.y - distance * math.sin(yaw)
    cam_z = transform.location.z + height

    dx = transform.location.x - cam_x
    dy = transform.location.y - cam_y
    dz = transform.location.z + 1.2 - cam_z
    horizontal = max(math.hypot(dx, dy), 1e-6)
    pitch = math.degrees(math.atan2(dz, horizontal))

    world.get_spectator().set_transform(
        carla.Transform(
            carla.Location(x=cam_x, y=cam_y, z=cam_z),
            carla.Rotation(pitch=pitch, yaw=transform.rotation.yaw, roll=0.0),
        )
    )


def find_hero(world: carla.World, timeout_s: float) -> carla.Vehicle | None:
    """Wait briefly for the native ROS2 stack to finish spawning the hero."""
    deadline = time.monotonic() + max(timeout_s, 0.0)
    reported = False
    while True:
        vehicles = list(world.get_actors().filter("vehicle.*"))
        hero = next(
            (v for v in vehicles if v.attributes.get("role_name") == "hero"),
            vehicles[0] if vehicles else None,
        )
        if hero is not None:
            return hero
        if time.monotonic() >= deadline:
            return None
        if not reported:
            print("Waiting for CARLA hero vehicle from ros2_native.py ...")
            reported = True
        time.sleep(0.5)


def choose_destination(world: carla.World, hero: carla.Vehicle) -> carla.Location:
    """Choose a forward-ish spawn point as a stable, long demo destination."""
    current = hero.get_location()
    points = [sp.location for sp in world.get_map().get_spawn_points()]
    if not points:
        return current

    transform = hero.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)

    def score(p: carla.Location) -> float:
        dx = p.x - current.x
        dy = p.y - current.y
        distance = math.hypot(dx, dy)
        if distance < 60.0:
            return -1e9
        forward = (dx * fx + dy * fy) / max(distance, 1e-6)
        # Prefer destinations generally ahead while still accepting route options.
        return 180.0 * forward + min(distance, 300.0)

    return max(points, key=score)


class ObstacleGuard:
    """Tracks the nearest forward obstacle reported by CARLA's obstacle sensor."""

    def __init__(self) -> None:
        self.distance = float("inf")
        self.other_actor = None
        self.last_event_time = 0.0

    def callback(self, event: carla.ObstacleDetectionEvent) -> None:
        self.distance = float(event.distance)
        self.other_actor = event.other_actor
        self.last_event_time = time.monotonic()

    def active_distance(self, timeout: float = 0.25) -> float:
        if time.monotonic() - self.last_event_time > timeout:
            return float("inf")
        return self.distance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--camera-distance", type=float, default=16.0)
    ap.add_argument("--camera-height", type=float, default=8.0)
    ap.add_argument("--wait-for-hero", type=float, default=30.0)
    ap.add_argument("--target-speed", type=float, default=22.0, help="Target speed in km/h for cautious demo driving")
    ap.add_argument("--obstacle-distance", type=float, default=12.0, help="Obstacle sensor look-ahead distance in m")
    ap.add_argument("--hard-brake-distance", type=float, default=5.5, help="Emergency brake threshold in m")
    ap.add_argument("--profile", choices=("cautious", "normal"), default="cautious")
    args = ap.parse_args()

    if not AGENT_AVAILABLE:
        raise SystemExit(
            "CARLA BehaviorAgent is unavailable. Make sure ~/CARLA/PythonAPI/carla "
            "contains the CARLA agents package."
        )

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    hero = find_hero(world, args.wait_for_hero)
    if hero is None:
        raise SystemExit(
            f"No CARLA hero vehicle found after waiting {args.wait_for_hero:.1f}s. "
            "Start ros2_native.py -f stack.json first."
        )

    # Client-side route planning/control: lane following, traffic-light response,
    # vehicle/pedestrian avoidance and emergency stopping.
    agent = BehaviorAgent(hero, behavior=args.profile)
    agent.set_target_speed(float(args.target_speed))
    agent.ignore_traffic_lights(active=False)
    agent.ignore_stop_signs(active=False)
    agent.ignore_vehicles(active=False)

    destination = choose_destination(world, hero)
    agent.set_destination(destination)

    # Add an independent obstacle sensor as a final braking guard. CARLA's
    # obstacle detector checks a capsule ahead of the vehicle and can detect
    # world geometry, not only dynamic actors.
    guard = ObstacleGuard()
    obstacle_bp = world.get_blueprint_library().find("sensor.other.obstacle")
    obstacle_bp.set_attribute("distance", str(args.obstacle_distance))
    obstacle_bp.set_attribute("hit_radius", "0.8")
    obstacle_bp.set_attribute("only_dynamics", "false")
    obstacle_bp.set_attribute("sensor_tick", "0.0")
    obstacle = world.spawn_actor(
        obstacle_bp,
        carla.Transform(carla.Location(x=1.8, z=1.2)),
        attach_to=hero,
    )
    obstacle.listen(guard.callback)

    print(f"Autonomous hero id={hero.id}, type={hero.type_id}")
    print(f"BehaviorAgent profile={args.profile}, target_speed={args.target_speed:.1f} km/h")
    print(f"Destination=({destination.x:.1f}, {destination.y:.1f}, {destination.z:.1f})")
    print(
        "Route-aware CARLA autonomy active: lane following, traffic rules, "
        "vehicle/pedestrian avoidance + obstacle emergency-brake guard."
    )
    print("ROS2 LiDAR/FFEM remains live and independent.")
    print("Spectator camera is following the hero vehicle.")

    start = time.monotonic()
    last_print = 0.0
    try:
        while args.duration <= 0 or time.monotonic() - start < args.duration:
            control = agent.run_step()
            obstacle_distance = guard.active_distance()

            # Hard safety override: stop before close obstacle contact.
            if math.isfinite(obstacle_distance) and obstacle_distance <= args.hard_brake_distance:
                control.throttle = 0.0
                control.brake = 1.0
                control.hand_brake = False

            hero.apply_control(control)
            update_spectator(world, hero, args.camera_distance, args.camera_height)

            now = time.monotonic()
            if now - last_print >= 1.0:
                loc = hero.get_location()
                vel = hero.get_velocity()
                speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
                print(
                    f"position=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}) "
                    f"speed={speed:.1f} m/s throttle={control.throttle:.2f} "
                    f"steer={control.steer:.2f} brake={control.brake:.2f} "
                    f"obstacle={obstacle_distance:.1f} m"
                )
                last_print = now

            if agent.done():
                print("Destination reached; selecting a new route.")
                destination = choose_destination(world, hero)
                agent.set_destination(destination)
                print(f"New destination=({destination.x:.1f}, {destination.y:.1f}, {destination.z:.1f})")

            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            obstacle.stop()
            obstacle.destroy()
        except RuntimeError:
            pass
        try:
            hero.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()

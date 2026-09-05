#!/usr/bin/env python3
"""Run a safer autonomous CARLA demo driver for the existing native ROS2 hero.

The native ROS2 stack already owns the vehicle and LiDAR. This helper does not
spawn a second vehicle or replace the LiDAR callback. It configures CARLA
Traffic Manager for conservative collision avoidance / lane changes and keeps
the spectator camera behind the hero vehicle so the autonomous behavior is
visible during the demo.

This is a demo safety controller, not a certified autonomous-driving stack.
The LiDAR/FFEM perception pipeline remains independent and continues receiving
its real ROS2 PointCloud2 stream.
"""
from __future__ import annotations

import argparse
import math
import time

import carla


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
            carla.Rotation(
                pitch=pitch,
                yaw=transform.rotation.yaw,
                roll=0.0,
            ),
        )
    )


def configure_traffic_manager(tm, hero: carla.Vehicle, speed_difference: float) -> None:
    """Configure a conservative Traffic Manager profile for a demo."""
    tm.set_synchronous_mode(False)
    tm.set_global_distance_to_leading_vehicle(12.0)

    # Keep the hero cautious around other actors and allow it to change lanes
    # rather than forcing a collision with a blocked lane.
    tm.distance_to_leading_vehicle(hero, 12.0)
    tm.vehicle_percentage_speed_difference(hero, speed_difference)
    tm.auto_lane_change(hero, True)
    tm.ignore_vehicles_percentage(hero, 0.0)
    tm.ignore_walkers_percentage(hero, 0.0)
    tm.ignore_lights_percentage(hero, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument(
        '--speed-difference',
        type=float,
        default=25.0,
        help='Traffic Manager speed difference in percent; positive values slow the hero.',
    )
    ap.add_argument(
        '--duration',
        type=float,
        default=0.0,
        help='Seconds to run; 0 means until Ctrl-C.',
    )
    ap.add_argument('--camera-distance', type=float, default=14.0)
    ap.add_argument('--camera-height', type=float, default=7.0)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    vehicles = list(world.get_actors().filter('vehicle.*'))
    if not vehicles:
        raise SystemExit('No vehicle found. Start ros2_native.py -f stack.json first.')

    hero = next(
        (v for v in vehicles if v.attributes.get('role_name') == 'hero'),
        vehicles[0],
    )

    tm = client.get_trafficmanager()
    configure_traffic_manager(tm, hero, args.speed_difference)
    hero.set_autopilot(True, tm.get_port())

    print(f'Autonomous hero id={hero.id}, type={hero.type_id}')
    print('Traffic Manager: conservative gap=12 m, auto lane change=ON')
    print('Obstacle-aware CARLA autopilot is active; ROS2 LiDAR/FFEM stays live.')
    print('Spectator camera is following the hero vehicle.')

    last_print = 0.0
    start = time.monotonic()
    try:
        while args.duration <= 0 or time.monotonic() - start < args.duration:
            update_spectator(world, hero, args.camera_distance, args.camera_height)
            now = time.monotonic()
            if now - last_print >= 1.0:
                loc = hero.get_location()
                vel = hero.get_velocity()
                speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
                ctrl = hero.get_control()
                print(
                    f'position=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}) '
                    f'speed={speed:.1f} m/s throttle={ctrl.throttle:.2f} '
                    f'steer={ctrl.steer:.2f} brake={ctrl.brake:.2f}'
                )
                last_print = now
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            hero.set_autopilot(False, tm.get_port())
            hero.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        except RuntimeError:
            pass


if __name__ == '__main__':
    main()

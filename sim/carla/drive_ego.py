#!/usr/bin/env python3
"""Drive the vehicle already spawned by CARLA ROS2 native.

The native ROS2 example creates the vehicle and sensors from stack.json but
leaves the vehicle stationary. This helper only enables CARLA Traffic Manager
autopilot for that existing vehicle, so LiDAR frames continuously change.
During the demo it also keeps the CARLA spectator camera behind/above the hero
vehicle so the moving vehicle remains visible on screen.
"""
from __future__ import annotations
import argparse
import math
import time
import carla


def update_spectator(world: carla.World, vehicle: carla.Vehicle, distance: float, height: float) -> None:
    """Follow the vehicle with a smooth-ish third-person spectator camera."""
    transform = vehicle.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    # Camera sits behind the vehicle along its reverse heading.
    cam_x = transform.location.x - distance * math.cos(yaw)
    cam_y = transform.location.y - distance * math.sin(yaw)
    cam_z = transform.location.z + height

    # Aim toward the vehicle from the camera position.
    dx = transform.location.x - cam_x
    dy = transform.location.y - cam_y
    dz = transform.location.z + 1.2 - cam_z
    horizontal = max(math.hypot(dx, dy), 1e-6)
    pitch = math.degrees(math.atan2(dz, horizontal))
    spectator = world.get_spectator()
    spectator.set_transform(
        carla.Transform(
            carla.Location(x=cam_x, y=cam_y, z=cam_z),
            carla.Rotation(pitch=pitch, yaw=transform.rotation.yaw, roll=0.0),
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--speed-difference', type=float, default=0.0,
                    help='Traffic Manager percentage speed difference; 0 keeps the speed limit.')
    ap.add_argument('--duration', type=float, default=0.0,
                    help='Seconds to keep the script alive; 0 means until Ctrl-C.')
    ap.add_argument('--camera-distance', type=float, default=12.0,
                    help='Third-person spectator distance behind the vehicle (m).')
    ap.add_argument('--camera-height', type=float, default=7.0,
                    help='Third-person spectator height above the vehicle (m).')
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    vehicles = list(world.get_actors().filter('vehicle.*'))
    if not vehicles:
        raise SystemExit('No vehicle found. Start ros2_native.py -f stack.json first.')

    # Prefer the native demo hero; otherwise use the first available vehicle.
    hero = next((v for v in vehicles if v.attributes.get('role_name') == 'hero'), vehicles[0])
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(False)
    hero.set_autopilot(True, tm.get_port())
    tm.vehicle_percentage_speed_difference(hero, args.speed_difference)

    print(f'Driving vehicle id={hero.id}, type={hero.type_id}')
    print('CARLA -> ROS2 LiDAR should now publish continuously.')
    print('Spectator camera is following the hero vehicle.')
    try:
        start = time.monotonic()
        while args.duration <= 0 or time.monotonic() - start < args.duration:
            update_spectator(world, hero, args.camera_distance, args.camera_height)
            time.sleep(0.1)
            if int((time.monotonic() - start) * 10) % 10 == 0:
                loc = hero.get_location()
                vel = hero.get_velocity()
                speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
                print(f'position=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}) speed={speed:.1f} m/s')
    except KeyboardInterrupt:
        pass
    finally:
        hero.set_autopilot(False, tm.get_port())


if __name__ == '__main__':
    main()

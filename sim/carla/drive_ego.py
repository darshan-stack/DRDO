#!/usr/bin/env python3
"""Drive the vehicle already spawned by CARLA ROS2 native.

The native ROS2 example creates the vehicle and sensors from stack.json but
leaves the vehicle stationary. This helper only enables CARLA Traffic Manager
autopilot for that existing vehicle, so LiDAR frames continuously change.
"""
from __future__ import annotations
import argparse
import time
import carla


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--speed-difference', type=float, default=0.0,
                    help='Traffic Manager percentage speed difference; 0 keeps the speed limit.')
    ap.add_argument('--duration', type=float, default=0.0,
                    help='Seconds to keep the script alive; 0 means until Ctrl-C.')
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    vehicles = list(world.get_actors().filter('vehicle.*'))
    if not vehicles:
        raise SystemExit('No vehicle found. Start ros2_native.py -f stack.json first.')

    # Prefer the vehicle carrying the native demo role/name; otherwise use the
    # first vehicle. Do not spawn a second vehicle or second LiDAR here.
    hero = next((v for v in vehicles if v.attributes.get('role_name') == 'hero'), vehicles[0])
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(False)
    hero.set_autopilot(True, tm.get_port())
    tm.vehicle_percentage_speed_difference(hero, args.speed_difference)

    print(f'Driving vehicle id={hero.id}, type={hero.type_id}')
    print('CARLA -> ROS2 LiDAR should now publish continuously.')
    try:
        start = time.monotonic()
        while args.duration <= 0 or time.monotonic() - start < args.duration:
            time.sleep(1.0)
            loc = hero.get_location()
            print(f'position=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})')
    except KeyboardInterrupt:
        pass
    finally:
        hero.set_autopilot(False, tm.get_port())


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""CARLA scenario runner.

Requires the CARLA Python API. It is intentionally independent from ROS; the
CARLA ROS bridge consumes the spawned actors and publishes their sensor data.
"""
from __future__ import annotations
import argparse, random, time
try:
    import carla
except ImportError as exc:
    raise SystemExit('Install the CARLA Python API before running this script.') from exc

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', type=int, default=2000); ap.add_argument('--map', default='Town03'); ap.add_argument('--vehicles', type=int, default=20); ap.add_argument('--seed', type=int, default=7); ap.add_argument('--ticks', type=int, default=0); args=ap.parse_args()
    client=carla.Client(args.host,args.port); client.set_timeout(10.0); world=client.load_world(args.map); original=world.get_settings(); settings=world.get_settings(); settings.synchronous_mode=True; settings.fixed_delta_seconds=0.05; world.apply_settings(settings)
    traffic=client.get_trafficmanager(); traffic.set_synchronous_mode(True); traffic.set_random_device_seed(args.seed); random.seed(args.seed)
    actors=[]
    try:
        bp=world.get_blueprint_library().filter('vehicle.tesla.model3')[0]; spawn=world.get_map().get_spawn_points()[0]; ego=world.try_spawn_actor(bp,spawn); actors.append(ego)
        if ego is None: raise RuntimeError('Could not spawn ego vehicle')
        ego.set_autopilot(True, traffic.get_port())
        lidar_bp=world.get_blueprint_library().find('sensor.lidar.ray_cast'); lidar_bp.set_attribute('channels','32'); lidar_bp.set_attribute('range','80'); lidar_bp.set_attribute('points_per_second','280000'); lidar_bp.set_attribute('rotation_frequency','10'); lidar_bp.set_attribute('upper_fov','10'); lidar_bp.set_attribute('lower_fov','-30'); lidar=world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=2.2)), attach_to=ego); actors.append(lidar)
        lidar.listen(lambda _data: None)
        print('CARLA scenario ready: ego_vehicle, LiDAR, synchronous mode, fixed_delta=0.05s')
        i=0
        while args.ticks <= 0 or i < args.ticks: world.tick(); i+=1
    finally:
        for actor in reversed(actors):
            try: actor.destroy()
            except Exception: pass
        world.apply_settings(original); traffic.set_synchronous_mode(False)
if __name__=='__main__': main()

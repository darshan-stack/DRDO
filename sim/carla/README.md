# CARLA integration

The intended integrated path is:

```text
CARLA -> carla_ros_bridge -> /carla/ego_vehicle/lidar -> ffem_node -> FFEM topics -> RViz2 and ffem_rerun_logger -> Rerun
```

Start the CARLA server first, then install and source the official CARLA ROS 2 bridge. Use the bridge's synchronous mode and a fixed simulation step. The supplied `config.yaml` records the experiment settings and `run_scenario.py` can spawn a deterministic ego vehicle and LiDAR when the CARLA Python API is available.

Typical run sequence:

```bash
# terminal 1: CARLA server
./CarlaUE4.sh -quality-level=Epic

# terminal 2: ROS 2 and CARLA bridge
source /opt/ros/<ros_distro>/setup.bash
ros2 launch carla_ros_bridge carla_ros_bridge.launch.py synchronous_mode:=True fixed_delta_seconds:=0.05

# terminal 3: spawn scenario if using the Python API
python3 sim/carla/run_scenario.py --map Town03 --vehicles 20 --ticks 1000

# terminal 4: FFEM mapper and Rerun logger
source /opt/ros/<ros_distro>/setup.bash
ros2 launch ffem_lidar_mapping ffem_integrated.launch.py input_topic:=/carla/ego_vehicle/lidar map_frame:=map

# terminal 5: RViz2
rviz2
```

The current repository does not install CARLA or the external bridge. Those are environment-level dependencies. The FFEM side is designed to consume the standard ROS `sensor_msgs/PointCloud2` topic produced by the bridge.

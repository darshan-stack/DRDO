# FFEM Final Stack

## End-to-end architecture

```text
CARLA 0.9.16 native ROS2
        |
        +--> /carla/hero/lidar/point_cloud
        +--> /carla/hero/rgb/image
        +--> /carla/hero/imu
        +--> /carla/hero/gnss
        |
        v
ROS 2 PointCloud2
        |
        v
FFEM ROS 2 node
        |
        +--> point validation / point-budget control
        +--> TF transform (optional)
        +--> spherical range projection
        +--> PyTorch LiDAR semantic inference
        +--> scan-to-scan motion residual
        +--> centroid tracking
        +--> elevation mean / variance fusion
        +--> traversability / semantic uncertainty
        +--> attention score
        +--> adaptive radial resolution
        +--> refinement / hysteresis events
        |
        +--> semantic PointCloud2
        +--> 2.5D elevation PointCloud2
        +--> adaptive-cell level PointCloud2
        +--> traversability PointCloud2
        +--> uncertainty PointCloud2
        +--> attention PointCloud2
        +--> moving-point PointCloud2
        +--> dynamic-track MarkerArray
        +--> refinement MarkerArray
        +--> metrics and planning-risk summaries
        |
        +--> Rerun
        +--> RViz2
        +--> browser dashboard
```

## Live ROS 2 topics

- `/carla/hero/lidar/point_cloud`
- `/ffem_mapper/map/semantic`
- `/ffem_mapper/map/elevation`
- `/ffem_mapper/map/adaptive_cells`
- `/ffem_mapper/map/traversability`
- `/ffem_mapper/map/uncertainty`
- `/ffem_mapper/map/attention`
- `/ffem_mapper/map/moving_points`
- `/ffem_mapper/tracks`
- `/ffem_mapper/refinement_markers`
- `/ffem_mapper/metrics`
- `/ffem_mapper/planning/risk`
- `/ffem_mapper/refinement_events`

## RViz2

The package installs `rviz/ffem_2p5d.rviz`. It visualizes the raw LiDAR, semantic RGB point cloud, live elevation map colored by Z, adaptive resolution levels, attention, optional traversability and uncertainty layers, moving points, dynamic tracks, and refinement events.

The integrated launch file defaults to opening RViz2. Disable it with `enable_rviz:=false` when benchmarking or running headless.

## Full-stack command

```bash
cd ~/DRDO
source /opt/ros/humble/setup.bash
source install/setup.bash
source ~/DRDO/sim/carla/native_env.sh

CUDA_VISIBLE_DEVICES="" \
FFEM_TORCH_THREADS=4 \
ros2 launch ffem_lidar_mapping ffem_integrated.launch.py \
  input_topic:=/carla/hero/lidar/point_cloud \
  map_frame:=lidar \
  model_backend:=torch_range \
  enable_rerun:=true \
  enable_rviz:=true \
  range_height:=16 \
  range_width:=512 \
  max_points_per_frame:=6000 \
  max_active_cells:=12000
```

Use the 16x512/6000-point profile for the current workstation's memory-safe demonstration. Use the higher-resolution defaults only when sufficient GPU memory is available.

## Scientific interpretation

The current prototype is adaptive at the cell-level using resolution bands plus semantic uncertainty, motion, traversability, geometry and range. The current cell representation records elevation, variance, semantic probabilities, motion probability, traversability, attention, and resolution level. A full production implementation can later replace the compact range-image backend and extend the hierarchy into a strict persistent parent-child quadtree without changing the ROS topic contract.

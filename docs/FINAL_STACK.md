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
        +--> optional TF transform
        +--> spherical range projection
        +--> PyTorch LiDAR semantic inference
        +--> scan-to-scan motion residual
        +--> centroid tracking
        +--> hierarchical 2.5D elevation fusion
        |       +--> explicit parent -> 4 children
        |       +--> refinement / hysteresis merge
        +--> semantic uncertainty / traversability / geometry
        +--> multi-signal attention
        +--> distance-based adaptive resolution
        |
        +--> preliminary risk-aware local planning
        |       +--> path risk profile
        |       +--> planning-critical regions
        |               |
        |               +--> feedback into map attention/refinement
        |
        +--> final risk-aware local path
        |       |
        |       +--> /ffem_mapper/planning/path
        |       +--> /ffem_mapper/planning/risk
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
        |
        +--> Rerun
        +--> RViz2
        +--> browser dashboard
        |
        +--> CARLA path follower (demonstration controller)
                |
                +--> steering / throttle / brake
                +--> ego motion
                +--> new LiDAR observations
                +--> closed perception -> map -> plan -> control loop
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
- `/ffem_mapper/planning/path`
- `/ffem_mapper/refinement_events`

## Hierarchical adaptive map

The current mapper maintains explicit persistent spatial nodes with a `parent`
field and four child nodes. Refinement replaces an active parent leaf with four
children; hysteresis can merge four quiet siblings back into their parent.
Each leaf stores elevation, variance, semantic probabilities, motion,
traversability, attention and planning criticality.

## Closed-loop planning feedback

The planner evaluates multiple short lateral trajectories using traversability,
motion, semantic uncertainty, attention and smoothness. A normalized risk profile
along the selected path is fed back into map-cell planning criticality. High
criticality can trigger additional refinement before the final path is published.
This closes the research feedback loop rather than treating planning as a
standalone downstream visualization.

## RViz2

The package installs `rviz/ffem_2p5d.rviz`. It visualizes raw LiDAR, semantic RGB
points, live Z-colored elevation, adaptive resolution levels, traversability,
uncertainty, attention, moving points, dynamic tracks, refinement events and the
risk-aware planned path.

The integrated launch file defaults to opening RViz2. Disable it with
`enable_rviz:=false` for headless benchmarking.

## CARLA closed-loop demonstration

`s​im/carla/follow_ffem_path.py` subscribes to `/ffem_mapper/planning/path`,
reads the CARLA hero vehicle, and applies a lightweight pure-pursuit-style
steering/throttle/brake command. It is a demonstration controller, not a
safety-rated automotive controller.

Run it with the CARLA Python environment while the FFEM mapper is publishing a
path:

```bash
cd ~/DRDO
source /opt/ros/humble/setup.bash
source install/setup.bash
source ~/DRDO/sim/carla/native_env.sh
~/CARLA/carla_env/bin/python3 sim/carla/follow_ffem_path.py
```

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

## Final validation

```bash
cd ~/DRDO
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 scripts/final_validation.py
```

Run the existing formal evaluations when the datasets/ground-truth sources are
available. The repository contains a SemanticPOSS held-out evaluator and a
CARLA normal-LiDAR vs semantic-LiDAR generalization evaluator; the validation
report deliberately leaves these as `not_run` until real experiments produce
metrics.

## Scientific interpretation

The final prototype now implements the core PS loop: semantic and dynamic
perception drive adaptive spatial representation; the map preserves elevation;
the planner produces a risk-aware local path; path risk feeds back into map
criticality/refinement; and a CARLA controller can close the demonstration loop.
The semantic backend remains a compact range-image model, while the mapping
representation is explicitly hierarchical and four-way refinable.

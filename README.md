# FFEM LiDAR Mapping

**Feedback-Foveated Elevation Mapping (FFEM)** is a research prototype scaffold for DRDO PS-26053: adaptive variable-resolution 2.5D LiDAR mapping for dynamic-environment perception.

The repository is intentionally organized as a staged robotics research codebase. The first milestone is a reproducible replay pipeline with a fixed-resolution 2.5D baseline and Rerun visualization. Adaptive horizontal cells, semantic uncertainty, dynamic-object likelihood, vertical slices, predictive dilation, and ROS 2 integration are added incrementally.

## Installation without a virtual environment

The project supports Python 3.10 and newer. For a direct system installation, use the same interpreter for both pip and the demo:

```bash
python3 --version
python3 -m pip install -e .
```

If Debian or Ubuntu reports an `externally-managed-environment` error, use the distribution override explicitly:

```bash
python3 -m pip install --break-system-packages -e .
```

On systems where `python3` is Python 3.10, this resolves the package requirement. If the `rerun-sdk` wheel is unavailable for the platform, install the demo dependencies separately and run the smoke test without Rerun:

```bash
python3 -m pip install --break-system-packages numpy pyyaml
python3 scripts/run_replay.py --frames 10 --no-rerun
```

## Architecture

```text
LiDAR / rosbag replay
        -> synchronization and pose transform
        -> deskewing, filtering, ground separation
        -> semantic segmentation + moving-object segmentation
        -> object tracking and terrain-feature extraction
        -> coarse 2.5D elevation map
        -> attention score and budget controller
        -> horizontal split/merge + vertical slice create/merge
        -> traversability map, Rerun logging, metrics
```

## Repository map

- `src/ffem/io`: LiDAR, pose, rosbag, and synchronization interfaces.
- `src/ffem/preprocessing`: deskewing, transforms, ground filtering, and outlier rejection.
- `src/ffem/perception`: semantic, motion, tracking, velocity, and uncertainty backends.
- `src/ffem/mapping`: fixed and adaptive 2.5D maps, hashing, fusion, continuity, and persistence.
- `src/ffem/traversability`: terrain descriptors and cost-map computation.
- `src/ffem/adaptation`: attention score, predictive dilation, hysteresis, and budget policy.
- `src/ffem/visualization`: Rerun entities and optional visualization sinks.
- `src/ffem/evaluation`: accuracy, calibration, dynamic, memory, and latency metrics.
- `configs`: reproducible sensor, model, and experiment configurations.
- `scripts`: replay, benchmarking, recording generation, and figure export entry points.
- `tests`: unit, integration, and regression tests.

## Development stages

1. Build a synthetic or user-supplied LiDAR replay and log raw points, trajectory, and timing to Rerun.
2. Implement the fixed-resolution elevation and traversability baseline.
3. Add semantic probabilities and moving-object probability as map channels.
4. Add budgeted adaptive horizontal cells with parent-child fusion and hysteresis.
5. Add multimodal-height vertical slices and continuity-preserving split/merge.
6. Add velocity-aware predictive dilation and compare against reactive refinement.
7. Add real model backends and ROS 2/rosbag input.

Do not claim research novelty from the scaffold alone. The experimental code must compare fixed-grid, geometry-adaptive, semantic-adaptive, and FFEM variants under identical replay and hardware conditions.

## Rerun visualization entities

The visualization layer should use stable paths including `world/lidar/raw`, `world/lidar/semantic`, `world/dynamics/moving_points`, `world/dynamics/tracks`, `world/map/elevation`, `world/map/uncertainty`, `world/map/traversability`, `world/map/adaptive_cells`, `world/map/vertical_slices`, `world/adaptation/refinement_events`, `world/robot/trajectory`, `metrics/latency`, and `metrics/memory`.

## Data policy

Large datasets and model checkpoints are excluded from Git. Place them under `data/raw/` and `models/checkpoints/` locally, and document download and licensing instructions in `docs/datasets.md`.

## Git workflow

Create a feature branch for each experiment or subsystem. Every commit should include tests or a reproducible example. Keep generated `.rrd` recordings and figures under `outputs/` locally unless a small artifact is intentionally selected for version control.

## Production release status

The production candidate contains a real ROS 2 PointCloud2 path, a seven-class range-image semantic backend, scan-to-scan motion residuals, centroid tracking, hierarchical four-way adaptive 2.5D mapping, uncertainty/traversability/attention channels, closed-loop planning feedback, RViz2/Rerun outputs, a browser dashboard, and a CARLA demonstration controller.

A trained semantic checkpoint is an external artifact and is intentionally excluded from Git. Production operation requires an explicit checkpoint path:

~~~bash
export FFEM_CHECKPOINT=/absolute/path/semanticposs_range_model.pt
~~~

The fallback backend is available only for software smoke tests. The auto backend fails closed when no checkpoint is found.

## Architecture

~~~text
CARLA 0.9.16
    -> native CARLA ROS2 LiDAR
    -> /carla/hero/lidar/point_cloud
    -> FFEM ROS2 node
       -> range-image semantic inference
       -> ego-motion compensated motion residuals
       -> lightweight dynamic tracking
       -> hierarchical 2.5D elevation fusion
       -> uncertainty / traversability / attention
       -> adaptive four-child refinement and hysteresis merge
       -> risk-aware local planning
       -> planning feedback into map criticality
       -> final local path
    -> RViz2 / Rerun / dashboard
    -> optional CARLA path follower
~~~

Each active map leaf stores elevation mean/variance, semantic probabilities, motion probability, traversability cost, attention, and planning criticality. Event history, active-cell capacity, and refinement growth are bounded.

The configured radial resolution bands are:

~~~text
0-8 m      : 0.05 m
8-18 m     : 0.10 m
18-40 m    : 0.25 m
40-100 m   : 0.50 m
~~~

## Installation

~~~bash
cd ~/DRDO
python3 -m pip install --break-system-packages -e ".[dev,ml]"

source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select ffem_lidar_mapping
source install/setup.bash
~~~

The ROS 2 ament package installs the core ffem Python packages as well as the ROS wrapper, so the live mapper does not depend on an accidental source-tree PYTHONPATH.

## Model verification

~~~bash
python3 scripts/inspect_checkpoint.py "$FFEM_CHECKPOINT"
python3 scripts/test_checkpoint_synthetic.py
~~~

Synthetic checkpoint loading is an integration check, not a held-out accuracy result.

## CARLA production demo

Start CARLA 0.9.16, then:

~~~bash
cd ~/DRDO
source /opt/ros/humble/setup.bash
source install/setup.bash
source sim/carla/native_env.sh
export FFEM_CHECKPOINT=/absolute/path/semanticposs_range_model.pt
./sim/carla/run_native_demo.sh
~~~

The native demo verifies the CARLA server, reuses or starts exactly one native ROS2 bridge, verifies the LiDAR publisher, can publish the CARLA LiDAR pose through TF, starts the selected controller, launches the real semantic backend, and exposes the FFEM topics.

Controller modes:

~~~bash
FFEM_CONTROLLER=ffem       # FFEM planning -> CARLA control
FFEM_CONTROLLER=behavior   # CARLA BehaviorAgent demo
FFEM_CONTROLLER=none       # perception/mapping/planning only
~~~

The default controller is ffem. The Open3D viewer is optional and disabled by default so it is not on the critical path.

Lower-cost live settings can be selected explicitly:

~~~bash
FFEM_RANGE_HEIGHT=16
FFEM_RANGE_WIDTH=512
FFEM_MAX_POINTS=12000
FFEM_MAX_ACTIVE_CELLS=12000
FFEM_MAX_TOPOLOGY_CHANGES=24
FFEM_QUEUE_DEPTH=2
~~~

## Direct ROS 2 launch

~~~bash
ros2 launch ffem_lidar_mapping ffem_integrated.launch.py \
  input_topic:=/carla/hero/lidar/point_cloud \
  map_frame:=map \
  use_tf:=true \
  model_backend:=torch_range \
  checkpoint:="$FFEM_CHECKPOINT" \
  range_height:=16 \
  range_width:=512 \
  max_points_per_frame:=12000 \
  max_active_cells:=12000
~~~

Use map_frame=lidar and use_tf=false for a strictly sensor-local run.

## ROS 2 outputs

~~~text
/ffem_mapper/map/elevation
/ffem_mapper/map/semantic
/ffem_mapper/map/adaptive_cells
/ffem_mapper/map/traversability
/ffem_mapper/map/uncertainty
/ffem_mapper/map/attention
/ffem_mapper/map/moving_points
/ffem_mapper/tracks
/ffem_mapper/refinement_markers
/ffem_mapper/metrics
/ffem_mapper/planning/risk
/ffem_mapper/planning/path
/ffem_mapper/refinement_events
~~~

The semantic PointCloud2 carries both RGB visualization data and compact class IDs in intensity, allowing the dashboard to recover labels without a custom ROS message.

## Preflight and health

~~~bash
python3 scripts/preflight.py \
  --checkpoint "$FFEM_CHECKPOINT" \
  --require-checkpoint \
  --check-carla \
  --check-ffem

ros2 topic info -v /carla/hero/lidar/point_cloud
ros2 topic hz /carla/hero/lidar/point_cloud
ros2 topic hz /ffem_mapper/metrics
ros2 topic echo /ffem_mapper/planning/risk --once
ros2 topic echo /ffem_mapper/planning/path --once
~~~

## Validation

~~~bash
python3 -m pytest -q
python3 -m compileall -q src scripts tests
ruff check src scripts tests
python3 scripts/evaluate_segmentation.py ...
python3 scripts/evaluate_carla_generalization.py ...
python3 scripts/benchmark_replay.py ...
python3 scripts/benchmark_performance.py ...
python3 scripts/evaluate_memory_savings.py ...
python3 scripts/final_validation.py
~~~

Do not report semantic accuracy, CARLA generalization, FPS, or memory reduction until the corresponding real experiment has produced the JSON evidence. The memory experiment is a map-storage proxy, not whole-process RSS. The CARLA controller is a research demonstration controller, not a safety-certified system.

## Scientific and operational safeguards

- Mock/fallback perception is for software integration tests only.
- CARLA semantic LiDAR is evaluation ground truth only; ordinary LiDAR remains the FFEM input.
- FPS reports must state hardware, input rate, point budget, model resolution, device/precision, and visualization settings.
- Long-duration validation should monitor dropped frames, P50/P95 latency, active cells, hierarchy node count, topology changes, and process memory.
- Record the exact checkpoint path and checksum with final experiment outputs.

## Documentation

- docs/PRODUCTION_RUNBOOK.md contains the installation, launch, health-check, and troubleshooting procedure.
- docs/FINAL_STACK.md contains the end-to-end architecture and topic contract.
- docs/experiment_protocol.md defines the required baselines and quantitative metrics.

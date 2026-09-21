# FFEM LiDAR Mapping

Feedback-Foveated Elevation Mapping (FFEM) for adaptive variable-resolution 2.5D LiDAR mapping in dynamic environments.

## Production candidate

The `production-ready-ffem` branch is a hardened release candidate for the ROS 2 + native CARLA workflow. It includes a real PointCloud2 ingestion path, seven-class range-image semantic inference, scan-to-scan motion residuals, lightweight tracking, hierarchical four-way adaptive mapping, uncertainty/traversability/attention channels, closed-loop planning feedback, RViz2/Rerun/dashboard outputs, preflight checks, CI configuration, and an optional CARLA controller.

This is not a safety-certified autonomous-driving system. The semantic checkpoint, held-out evaluation, long-duration performance measurements, and the CARLA TF convention still require acceptance testing on the target machine.

## Design

```text
CARLA 0.9.16
    -> native CARLA ROS2 LiDAR
    -> /carla/hero/lidar/point_cloud
    -> FFEM ROS2 node
       -> range-image semantic inference
       -> ego-motion compensated motion residuals
       -> lightweight dynamic tracking
       -> hierarchical 2.5D elevation fusion
       -> uncertainty / traversability / attention
       -> adaptive four-child refinement + hysteresis merge
       -> risk-aware local planning
       -> planning feedback into map criticality
       -> final local path
    -> RViz2 / Rerun / dashboard
    -> optional CARLA controller
```

Each active map leaf stores elevation mean/variance, semantic probabilities, motion probability, traversability cost, attention, and planning criticality. Refinement is explicit 4-ary topology. Event history and active-cell capacity are bounded.

## Resolution policy

```text
0-8 m      : 0.05 m base resolution
8-18 m     : 0.10 m base resolution
18-40 m    : 0.25 m base resolution
40-100 m   : 0.50 m base resolution
```

Attention and planning feedback can refine mapped cells below their radial base rate, subject to the global 0.05 m finest-cell floor and hierarchy depth.

## Repository layout

```text
src/ffem/                 ROS-independent FFEM core
src/ffem/perception/      range projection + semantic backend
src/ffem/mapping/         adaptive map components
src/ffem/planning/        risk-aware local planner
src/ffem/ros2/            ROS 2 adapter
ros2/                     ament_python package + launch/RViz assets
sim/carla/                native CARLA integration + controller
scripts/                  replay, evaluation, benchmark, preflight
tests/                    regression and architecture tests
docs/                     production runbook and experiment protocol
dashboard/                browser dashboard
```

## Installation
## Docker runtime

A headless production runtime is provided in docker/Dockerfile and docker-compose.yml. CARLA remains an external runtime dependency; the FFEM container uses host networking so it can consume the native CARLA ROS2 LiDAR topic.

```bash
docker compose build
docker compose up
```

The container expects models/checkpoints/semanticposs_range_model.pt and fails closed if it is missing. The default container profile is non-actuating, sensor-local, headless, and suitable for validating the ROS2 mapping stack before enabling TF or a controller.

See docker/README.md for GPU pass-through, preflight, and health checks.


Development/replay environment:

```bash
cd ~/DRDO
python3 -m pip install --break-system-packages -e ".[dev,ml]"
```

ROS 2 runtime environment:

```bash
cd ~/DRDO
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select ffem_lidar_mapping
source install/setup.bash
```

The ROS 2 package installs the core `ffem` Python modules with the ament package, so the live node does not depend on an accidental source-tree `PYTHONPATH`.

## Semantic checkpoint

Check the model before live use:

```bash
export FFEM_CHECKPOINT=/absolute/path/semanticposs_range_model.pt
python3 scripts/inspect_checkpoint.py "$FFEM_CHECKPOINT"
python3 scripts/test_checkpoint_synthetic.py
```

`model_backend:=torch_range` requires a compatible checkpoint. `model_backend:=fallback` is explicitly for smoke tests only. The repository does not claim quantitative accuracy from fallback inference or synthetic checkpoint loading.

The current model is a compact range-image network with depth + intensity input and seven FFEM output classes: unknown, ground, vegetation, structure, vehicle, person, obstacle.

## Native CARLA demo

Start CARLA 0.9.16 first, then:

```bash
cd ~/DRDO
source /opt/ros/humble/setup.bash
source install/setup.bash
source sim/carla/native_env.sh
export FFEM_CHECKPOINT=/absolute/path/semanticposs_range_model.pt
./sim/carla/run_native_demo.sh
```

Safe defaults are non-actuating and sensor-local:

```text
FFEM_USE_CARLA_TF=0
FFEM_CONTROLLER=none
```

The script verifies the CARLA server, reuses or starts one native ROS 2 bridge, waits for the LiDAR publisher, launches the real semantic backend, and starts the visualization stack.

### Closed-loop CARLA demonstration

Only after validating the CARLA-to-native-ROS coordinate convention for your exact native LiDAR configuration:

```bash
FFEM_USE_CARLA_TF=1 FFEM_CONTROLLER=ffem ./sim/carla/run_native_demo.sh
```

The FFEM controller brakes on a stale/missing path and is a research demonstration controller, not a safety-rated vehicle controller.

### Lower-cost live profile

```bash
FFEM_RANGE_HEIGHT=16
FFEM_RANGE_WIDTH=512
FFEM_MAX_POINTS=12000
FFEM_MAX_ACTIVE_CELLS=12000
FFEM_MAX_TOPOLOGY_CHANGES=24
FFEM_QUEUE_DEPTH=2
```

## Direct ROS 2 launch

```bash
ros2 launch ffem_lidar_mapping ffem_integrated.launch.py \
  input_topic:=/carla/hero/lidar/point_cloud \
  map_frame:=lidar \
  use_tf:=false \
  model_backend:=torch_range \
  checkpoint:="$FFEM_CHECKPOINT" \
  range_height:=16 \
  range_width:=512 \
  max_points_per_frame:=12000 \
  max_active_cells:=12000
```

For TF-aware dynamic compensation, use `map_frame:=map use_tf:=true` only after the TF convention has been validated.

## ROS 2 topic contract

```text
/carla/hero/lidar/point_cloud
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
```

The native CARLA LiDAR publisher and FFEM subscriber both use BEST_EFFORT QoS. The semantic PointCloud2 carries RGB visualization plus the compact class ID in `intensity` for downstream tools.

## Preflight

```bash
python3 scripts/preflight.py \
  --checkpoint "$FFEM_CHECKPOINT" \
  --require-checkpoint \
  --check-carla \
  --check-ffem
```

Useful live checks:

```bash
ros2 topic info -v /carla/hero/lidar/point_cloud
ros2 topic hz /carla/hero/lidar/point_cloud
ros2 node info /ffem_mapper
ros2 topic hz /ffem_mapper/metrics
ros2 topic echo /ffem_mapper/planning/risk --once
ros2 topic echo /ffem_mapper/planning/path --once
```

Expected `/ffem_mapper` subscriptions include LiDAR and optional `/tf`/`/tf_static`. Expected publishers include all mapping, diagnostics, tracks, refinement, metrics, risk, and path topics above.

## Automated validation

```bash
python3 -m pytest -q
python3 -m compileall -q src scripts tests
ruff check src scripts tests
```

GitHub Actions runs the Python compile, Ruff, unit tests, and ROS 2 package metadata check. A green local run or GitHub run should be treated as a software-quality gate, not as field-performance evidence.

Formal experiment entry points:

```bash
python3 scripts/evaluate_segmentation.py ...
python3 scripts/evaluate_carla_generalization.py ...
python3 scripts/benchmark_replay.py ...
python3 scripts/benchmark_performance.py ...
python3 scripts/evaluate_memory_savings.py ...
python3 scripts/final_validation.py
```

Do not report semantic accuracy, CARLA generalization, FPS, or memory savings until the corresponding real experiment has produced JSON evidence. The memory experiment is a map-storage proxy, not whole-process RSS.

## Motion and TF acceptance

`VoxelMotionDetector` performs voxelized scan-to-scan residual matching and can consume a relative ego-motion transform. It is intentionally lightweight and should be treated as a research detector rather than a fully robust production multi-object motion system.

The CARLA TF broadcaster publishes the actual LiDAR actor pose. Before enabling it for moving-vehicle compensation, validate translation direction and yaw sign against the native ROS 2 PointCloud2 frame using a controlled vehicle translation/rotation test. CARLA and ROS integration paths can use different coordinate conventions, so this acceptance test belongs in the deployment process.

## Documentation

- `docs/PRODUCTION_RUNBOOK.md` — installation, launch, health checks, failure diagnosis, and release gates.
- `docs/FINAL_STACK.md` — end-to-end architecture and topic contract.
- `docs/experiment_protocol.md` — required baselines, metrics, and scientific safeguards.

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

## Current implementation status

The repository now includes a runnable 70%-ready prototype core in `src/ffem/pipeline.py`. It provides deterministic synthetic LiDAR replay, semantic probability channels, moving-point probability, elevation mean/variance fusion, traversability cost, attention scoring, adaptive cell levels, temporal hysteresis, vertical slices for multimodal heights, topology-change limits, and metrics. The perception outputs are deliberately mock backends and must later be replaced with trained LiDAR models.

Run the real-time Rerun demo directly on the system Python:

```bash
python3 -m pip install --break-system-packages -e .
python3 scripts/run_replay.py --frames 300 --recording outputs/ffem_demo.rrd
```

Run a headless smoke test:

```bash
python3 scripts/run_replay.py --frames 100 --no-rerun
python3 -m pytest tests/unit -q
```

The ROS 2 adapter is under `ros2/`. On a sourced ROS 2 installation, build it with:

```bash
cd ros2
python3 -m pip install --break-system-packages -e .
# or use colcon from the parent workspace:
# colcon build --packages-select ffem_lidar_mapping
# source install/setup.bash
# ros2 launch ffem_lidar_mapping ffem.launch.py
```

The current ROS 2 callback is an integration adapter and uses the core pipeline; PointCloud2 decoding and publication should be completed in the next milestone. This keeps the research core testable on machines without ROS 2.

## Integrated CARLA–ROS 2–FFEM–Rerun workflow

The main integration path is now represented in the repository:

```text
CARLA -> CARLA ROS bridge -> PointCloud2 -> FFEM ROS 2 node -> TF transform -> FFEM map
                                                   |-> RViz2 topics
                                                   |-> optional Rerun logger -> .rrd
```

The FFEM node now attempts a timestamped TF lookup from the LiDAR message frame into `map`, processes the actual decoded points, publishes elevation and moving-point clouds, publishes metrics and refinement events, and can optionally log the same data to Rerun. The standalone logger is useful when the mapper should remain independent of visualization.

The CARLA integration assets are under `sim/carla/`. They include `config.yaml`, a synchronous scenario runner, and a complete run guide. The external CARLA server, CARLA Python API, official CARLA ROS bridge, ROS 2 distribution, and RViz2 remain environment dependencies and are not bundled in this repository.

```bash
# after CARLA and the official bridge are installed and sourced
ros2 launch carla_ros_bridge carla_ros_bridge.launch.py synchronous_mode:=True fixed_delta_seconds:=0.05
python3 sim/carla/run_scenario.py --map Town03 --vehicles 20 --ticks 1000
ros2 launch ffem_lidar_mapping ffem_integrated.launch.py input_topic:=/carla/ego_vehicle/lidar map_frame:=map
rviz2
```

For a real CARLA experiment, ordinary LiDAR must be used as the FFEM input. CARLA semantic LiDAR or actor state should be connected only to a separate evaluation node as ground truth; it must not replace the deployed perception input.

## Integration limitations

The integrated path is now connected at the transport and processing level, but model perception remains deterministic mock logic, the CARLA bridge is an external dependency, and the FFEM map output is currently encoded as XYZ point clouds rather than a custom semantic/elevation message. The next production milestone is real TF validation against the target LiDAR driver, a trained semantic/MOS adapter, object tracking, and ground-truth evaluation on CARLA scenarios.

## Real semantic perception backend

The repository now includes a sensor-aware range-image segmentation interface in `src/ffem/perception/segmentation.py`. It provides LiDAR spherical projection, nearest-return z-buffering, a seven-class label contract, a PyTorch range-image model adapter, and an explicit fallback backend for CI only. `src/ffem/io/semantic_kitti.py` loads SemanticKITTI `.bin` and `.label` files and remaps labels into the FFEM class set. `scripts/train_segmentation.py` trains and saves a checkpoint when PyTorch is available.

Install the optional ML stack with:

```bash
python3 -m pip install --break-system-packages -e ".[ml,dev]"
```

Train on labeled scans:

```bash
python3 scripts/train_segmentation.py \
  --data-root /absolute/path/to/SemanticKITTI \
  --sequences 00 01 02 \
  --epochs 20 \
  --checkpoint models/checkpoints/range_segmentation.pt
```

Run a checkpoint in replay:

```bash
python3 scripts/run_replay.py \
  --backend torch_range \
  --checkpoint models/checkpoints/range_segmentation.pt \
  --frames 300 \
  --recording outputs/semantic_demo.rrd
```

Run the same backend in ROS 2:

```bash
ros2 run ffem_lidar_mapping ffem_node --ros-args \
  -p model_backend:=torch_range \
  -p checkpoint:=/absolute/path/range_segmentation.pt \
  -p input_topic:=/carla/ego_vehicle/lidar \
  -p map_frame:=map
```

The fallback backend remains available only so the repository can be tested without a dataset, GPU, or checkpoint. It must not be used for research accuracy claims.

## Drop-in checkpoint workflow

Place the trained checkpoint at:

```text
models/checkpoints/semantic_model.pt
```

With the default `auto` backend, both replay and ROS 2 automatically discover the first `.pt` file in that directory. You can override discovery with `FFEM_CHECKPOINT=/absolute/path/model.pt` or an explicit `--checkpoint`/`checkpoint:=...` argument. If no checkpoint exists, the system prints that it is using the fallback backend; this mode is for smoke tests only.

Install the optional visualization stack:

```bash
python3 -m pip install --break-system-packages -e ".[visual]"
```

Run Open3D and Rerun together:

```bash
python3 scripts/run_replay.py --frames 300 --open3d --recording outputs/ffem_demo.rrd
```

Run with a discovered checkpoint:

```bash
python3 scripts/run_replay.py --frames 300 --open3d --recording outputs/semantic_demo.rrd
```

Run CARLA/ROS 2 with automatic discovery:

```bash
ros2 launch ffem_lidar_mapping ffem_integrated.launch.py \
  input_topic:=/carla/ego_vehicle/lidar \
  map_frame:=map \
  model_backend:=auto
```

The model checkpoint must match the compact range-image architecture and seven-class output contract currently defined in `src/ffem/perception/segmentation.py`. A checkpoint from another architecture cannot be loaded automatically unless a corresponding adapter is added.

## Supplied SemanticPOSS checkpoint

The repository includes `models/checkpoints/semanticposs_first_model.pt`, which was supplied by the project owner and verified against the current seven-class compact range-image architecture. Automatic checkpoint discovery selects it when the `auto` backend is used. The checkpoint metadata reports dataset `semanticposs`, seven classes, and the expected class names: unknown, ground, vegetation, structure, vehicle, person, and obstacle.

Validate the included checkpoint without a dataset scan:

```bash
python3 scripts/inspect_checkpoint.py models/checkpoints/semanticposs_first_model.pt
python3 scripts/test_checkpoint_synthetic.py
```

Run a checkpoint-backed FFEM smoke test:

```bash
python3 scripts/run_replay.py --backend auto --frames 100 --no-rerun
```

This verifies model loading and inference integration. It is not a SemanticPOSS test-set accuracy result; those metrics require the original dataset and held-out labels.

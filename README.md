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

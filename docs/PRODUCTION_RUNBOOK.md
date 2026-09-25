# FFEM Production Runbook

This is the operational path for the ROS 2 + native CARLA demo. The trained semantic checkpoint is an external artifact and is not committed to Git.

## Required environment

The primary submission/demo path is the native Ubuntu + ROS 2 + CARLA stack. Docker is optional for reproducibility and is not required for the final demonstration.

## Docker deployment

The repository now includes a headless FFEM container. Build it from the repository root:

```bash
docker compose build
```

Start CARLA and its native ROS2 LiDAR publisher on the host, then run:

```bash
docker compose up
```

The container uses host networking and the same ROS2/Fast DDS environment as the native deployment. It does not launch RViz or a vehicle controller. This separation keeps container deployment focused on perception, mapping, planning, and observability.

Run the container preflight before a live session:

```bash
docker compose run --rm ffem preflight \\
  --checkpoint /models/checkpoints/semanticposs_range_model.pt \\
  --require-checkpoint \\
  --check-carla \\
  --check-ffem
```

For NVIDIA GPU use, configure NVIDIA Container Toolkit on the host and verify GPU visibility before enabling GPU access for the FFEM service. The application selects CUDA automatically when PyTorch exposes a CUDA device.


- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10+
- CARLA 0.9.16
- A seven-class FFEM range-image checkpoint compatible with the model in `src/ffem/perception/segmentation.py`

Install the repository in editable mode and install development/ML dependencies:

```bash
cd ~/DRDO
python3 -m pip install --break-system-packages -e ".[dev,ml]"
```

For ROS 2, build and source the package from the workspace root:

```bash
cd ~/DRDO
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select ffem_lidar_mapping
source install/setup.bash
```

The ROS2 package now installs the core `ffem` Python packages as part of the ament package, so the live node does not depend on an accidental `PYTHONPATH`.

## Check the checkpoint

```bash
cd ~/DRDO
python3 scripts/inspect_checkpoint.py /absolute/path/semanticposs_range_model.pt
python3 scripts/test_checkpoint_synthetic.py
```

Run with an explicit path in production:

```bash
export FFEM_CHECKPOINT=/absolute/path/semanticposs_range_model.pt
```

The `auto` backend is intentionally fail-closed when no checkpoint exists. Use `model_backend:=fallback` only for software smoke tests.

## Native CARLA environment

All CARLA ROS 2 terminals must share the same environment:

```bash
source ~/DRDO/sim/carla/native_env.sh
source /opt/ros/humble/setup.bash
source ~/DRDO/install/setup.bash
```

Confirm:

```bash
echo "$ROS_DOMAIN_ID"
echo "$RMW_IMPLEMENTATION"
echo "$FASTDDS_BUILTIN_TRANSPORTS"
```

Expected values are `0`, `rmw_fastrtps_cpp`, and `UDPv4`.

## One-command demo

Start CARLA 0.9.16 with native ROS 2 enabled:

```bash
cd ~/CARLA
source /opt/ros/humble/setup.bash
source ~/DRDO/sim/carla/native_env.sh
./CarlaUE4.sh --ros2
```

Then start the native ROS 2 sensor bridge in another terminal:

```bash
cd ~/CARLA/PythonAPI/examples/ros2
source /opt/ros/humble/setup.bash
source ~/DRDO/sim/carla/native_env.sh
~/CARLA/carla_env/bin/python3 ros2_native.py --host 127.0.0.1 --port 2000 --file stack.json --verbose
```

With those terminals running, launch the FFEM demo:

```bash
cd ~/DRDO
export FFEM_CHECKPOINT=/absolute/path/semanticposs_range_model.pt
./sim/carla/run_native_demo.sh
```

The production demo defaults to a non-actuating sensor-local run:

- native CARLA ROS 2 LiDAR
- trained semantic backend
- bounded point and active-cell budgets
- Rerun enabled
- RViz2 enabled by default (`FFEM_RVIZ=0` disables it)
- controller disabled
- automatic CPU fallback when free VRAM is below the configured threshold
- browser dashboard when port 8765 is available

Enable TF and the FFEM controller only after validating the CARLA-to-native-ROS coordinate convention in RViz:

The optional Open3D viewer is disabled by default because it is not required for the core stack.

Controller choices:

```bash
FFEM_CONTROLLER=ffem      # FFEM planning -> CARLA control
FFEM_CONTROLLER=behavior  # independent CARLA BehaviorAgent demo
FFEM_CONTROLLER=none      # perception/mapping/planning only (default)
```

For a closed-loop CARLA demonstration:

```bash
FFEM_USE_CARLA_TF=1 FFEM_CONTROLLER=ffem ./sim/carla/run_native_demo.sh
```

Runtime controls:

```bash
FFEM_DEVICE=auto              # auto/cpu/cuda
FFEM_MIN_FREE_VRAM_MB=1024    # auto mode falls back to CPU below this free VRAM
FFEM_RVIZ=1                   # start RViz2 (default)
FFEM_OPEN3D=1                 # start bounded/throttled Open3D viewer
```

Useful limits:

```bash
FFEM_RANGE_HEIGHT=16
FFEM_RANGE_WIDTH=512
FFEM_MAX_POINTS=12000
FFEM_MAX_ACTIVE_CELLS=12000
FFEM_MAX_TOPOLOGY_CHANGES=24
FFEM_QUEUE_DEPTH=2
```

Set `FFEM_USE_CARLA_TF=0` for the default sensor-local mode. Before enabling TF for dynamic motion compensation, validate that the broadcaster's CARLA pose convention matches the native ROS2 LiDAR point convention in RViz and with a controlled translation/rotation test. This validation is required because CARLA and ROS coordinate conventions differ across integration paths.

## Preflight

Before a live demo:

```bash
cd ~/DRDO
python3 scripts/preflight.py   --checkpoint "$FFEM_CHECKPOINT"   --require-checkpoint   --check-carla   --check-ffem
```

When running only the mapper without CARLA, omit `--check-carla`.

## Live health checks

LiDAR transport:

```bash
ros2 topic info -v /carla/hero/lidar/point_cloud
ros2 topic hz /carla/hero/lidar/point_cloud
```

The native CARLA publisher should be BEST_EFFORT. The FFEM subscriber is also BEST_EFFORT.

FFEM processing:

```bash
ros2 node info /ffem_mapper
ros2 topic hz /ffem_mapper/metrics
ros2 topic echo /ffem_mapper/metrics --once
ros2 topic echo /ffem_mapper/planning/risk --once
ros2 topic echo /ffem_mapper/planning/path --once
```

Expected node subscriptions include the LiDAR topic and optional /tf and /tf_static. Expected publications include elevation, semantic, adaptive cells, traversability, uncertainty, attention, moving points, tracks, refinement markers, metrics, planning risk, and planning path.

## Failure diagnosis

### Publisher count is zero

The native `ros2_native.py` bridge is not currently publishing the LiDAR topic. Keep FFEM running, restart only the native bridge, and confirm:

```bash
ros2 topic info -v /carla/hero/lidar/point_cloud
```

### Publisher exists but FFEM is absent

Source the same ROS 2 and CARLA environment in the FFEM terminal and check:

```bash
ros2 node list | grep ffem
ros2 node info /ffem_mapper
```

### FFEM exists but no frame logs

Check topic QoS and traffic first:

```bash
ros2 topic info -v /carla/hero/lidar/point_cloud
ros2 topic hz /carla/hero/lidar/point_cloud
```

Then inspect the FFEM terminal for `FFEM callback failed`.

### Checkpoint error

The production backend does not silently replace a missing model with mock inference. Set an explicit valid checkpoint:

```bash
export FFEM_CHECKPOINT=/absolute/path/model.pt
```

### Path follower does not move

A stale or missing path causes the controller to brake. Check:

```bash
ros2 topic echo /ffem_mapper/planning/path --once
```

The path follower accepts both robot-local paths and map/world-frame paths. In map mode it transforms the target into the current CARLA vehicle frame before calculating steering.



### PS-26053 extended validation
After generating a fresh `outputs/memory_experiment_frames.npz`, run:

```bash
python3 scripts/benchmark_ps26053.py --frames-file outputs/memory_experiment_frames.npz --checkpoint models/checkpoints/semanticposs_range_model.pt --frames 100 --max-points 6000 --range-height 16 --range-width 512 --device cpu
python3 scripts/ablation_matrix.py --frames-file outputs/memory_experiment_frames.npz --checkpoint models/checkpoints/semanticposs_range_model.pt --frames 100 --max-points 6000 --range-height 16 --range-width 512 --device cpu
python3 scripts/evaluate_carla_dynamic.py --checkpoint models/checkpoints/semanticposs_range_model.pt --frames 100 --device cpu
python3 scripts/evaluate_carla_terrain.py --checkpoint models/checkpoints/semanticposs_range_model.pt --frames 100 --device cpu
```
The dynamic and terrain evaluators use synchronized CARLA semantic LiDAR ground truth. They must be run with a live CARLA server.
## Quantitative validation

Never report unmeasured accuracy or memory numbers. The repository provides:

```bash
python3 scripts/evaluate_segmentation.py ...
python3 scripts/evaluate_carla_generalization.py ...
python3 scripts/benchmark_replay.py ...
python3 scripts/benchmark_ps26053.py --frames-file outputs/memory_experiment_frames.npz ...
python3 scripts/benchmark_performance.py ...
python3 scripts/evaluate_memory_savings.py ...
python3 scripts/final_validation.py ...
```

The original memory experiment reports map-storage reduction only; it is not a whole-process RSS measurement. For actual process memory, use `scripts/benchmark_ps26053.py`, which runs each method in a fresh child process and reports peak RSS. The CARLA controller is a demonstration controller, not a safety-certified controller.

## Release checklist

Run the Python/ROS2 regression suite, build the ament package, run preflight, run the live CARLA demo, capture the performance benchmark on the target machine, and archive the exact checkpoint path/hash and generated evaluation JSON files.

A release is not complete when only the ROS graph is connected. The trained checkpoint, held-out semantic evaluation, CARLA generalization test, performance measurement, and long-duration stability run must all be available as evidence.

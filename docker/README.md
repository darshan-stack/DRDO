# FFEM Docker Runtime

This container packages the FFEM ROS 2 runtime. CARLA itself remains an external runtime dependency and should continue running with its native CARLA 0.9.16 installation.

## Build

From the repository root:

```bash
docker compose build
```

The image expects a trained checkpoint to be mounted from `models/checkpoints`.

## Run

Start CARLA and its native ROS 2 LiDAR publisher first, then:

```bash
docker compose up
```

The container uses host networking and the same Fast DDS environment as the native CARLA integration:

- `ROS_DOMAIN_ID=0`
- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
- `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`

The default container profile is headless and non-actuating:

- ROS 2 FFEM mapper: enabled
- RViz2: disabled
- CARLA controller: not launched
- TF: disabled
- Rerun logging: enabled

This is deliberate: the container should first validate LiDAR transport and mapping independently of vehicle actuation.

## Preflight inside the container

```bash
docker compose run --rm ffem preflight \
  --checkpoint /models/checkpoints/semanticposs_range_model.pt \
  --require-checkpoint \
  --check-carla \
  --check-ffem
```

Because the compose service uses host networking, the CARLA server at `127.0.0.1:2000` is reachable from the container.

## GPU

The Dockerfile installs PyTorch, but GPU pass-through is intentionally not hard-coded into the compose file because the exact NVIDIA Container Toolkit configuration belongs to the host.

For GPU deployment, verify Docker can see the host GPU first:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Then add your host's supported GPU reservation/runtime settings to the compose service. The FFEM application itself selects CUDA automatically when PyTorch reports a CUDA device.

## Health checks

```bash
docker compose ps
docker logs -f ffem_mapper
ros2 topic info -v /carla/hero/lidar/point_cloud
ros2 topic hz /ffem_mapper/metrics
```

A missing checkpoint is a hard startup error. The container never silently switches to mock perception.

## Outputs

Generated Rerun records and runtime JSON/log files are written through the mounted `outputs` directory.

The Docker runtime is intended for repeatable software deployment and headless ROS 2 processing. RViz2 and CARLA GUI remain host-side visualization components.

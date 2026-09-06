#!/usr/bin/env bash
set -euo pipefail

# PS 26053 performance profile.
# Keeps the same trained model and adaptive 2.5D mapper, while reducing
# range-image and point-processing cost for an explicit latency/FPS demo.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source /opt/ros/humble/setup.bash
source "$ROOT/sim/carla/native_env.sh"
source "$ROOT/install/setup.bash"

TOPIC="/carla/hero/lidar/point_cloud"
CHECKPOINT="$ROOT/models/checkpoints/semanticposs_range_model.pt"

if ! ros2 topic list | grep -qx "$TOPIC"; then
  echo "ERROR: $TOPIC is not available. Start CARLA + ros2_native.py first."
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 1
fi

echo "=== FFEM PS 26053 PERFORMANCE PROFILE ==="
echo "range image : 16 x 512"
echo "max points  : 6000 / frame"
echo "active cells: 12000"
echo "queue depth : 2"
echo "Rerun       : enabled"
echo

auto_cuda=""
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  auto_cuda="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

eval "$auto_cuda ros2 launch ffem_lidar_mapping ffem_integrated.launch.py \
  input_topic:=$TOPIC \
  map_frame:=lidar \
  use_tf:=false \
  model_backend:=torch_range \
  checkpoint:=$CHECKPOINT \
  range_height:=16 \
  range_width:=512 \
  max_range:=80.0 \
  max_points_per_frame:=6000 \
  max_active_cells:=12000 \
  max_topology_changes:=24 \
  queue_depth:=2 \
  enable_rerun:=true"

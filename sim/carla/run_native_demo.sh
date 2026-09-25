#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CARLA_ROOT="${CARLA_ROOT:-$HOME/CARLA}"
CARLA_PY="${CARLA_PY:-$CARLA_ROOT/carla_env/bin/python3}"
STACK_DIR="$CARLA_ROOT/PythonAPI/examples/ros2"
STACK_FILE="${STACK_FILE:-$STACK_DIR/stack.json}"
TOPIC="${CARLA_LIDAR_TOPIC:-/carla/hero/lidar/point_cloud}"
CHECKPOINT="${FFEM_CHECKPOINT:-$ROOT/models/checkpoints/semanticposs_range_model.pt}"

source "$ROOT/sim/carla/native_env.sh"
source /opt/ros/humble/setup.bash
if [[ -f "$ROOT/install/setup.bash" ]]; then source "$ROOT/install/setup.bash"; fi
set -u
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

if ! command -v colcon >/dev/null 2>&1; then echo "ERROR: colcon is not installed/available."; exit 1; fi
if [[ ! -x "$CARLA_PY" ]]; then echo "ERROR: CARLA Python environment not found: $CARLA_PY"; exit 1; fi
if [[ ! -f "$STACK_FILE" ]]; then echo "ERROR: Native CARLA stack file not found: $STACK_FILE"; exit 1; fi
if [[ ! -f "$CHECKPOINT" ]]; then echo "ERROR: Checkpoint not found: $CHECKPOINT"; exit 1; fi

cd "$ROOT"; mkdir -p outputs

echo "=== FFEM Native CARLA Demo ==="
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "FASTDDS_BUILTIN_TRANSPORTS=$FASTDDS_BUILTIN_TRANSPORTS"
echo "LiDAR topic=$TOPIC"
echo "Checkpoint=$CHECKPOINT"

"$CARLA_PY" - <<'PY'
import carla
client=carla.Client('127.0.0.1',2000); client.set_timeout(5.0)
print('CARLA server:',client.get_server_version()); print('CARLA map:',client.get_world().get_map().name)
PY

NATIVE_PID=""; CONTROLLER_PID=""; TF_PID=""; OPEN3D_PID=""; DASHBOARD_PID=""; OWN_NATIVE=0
cleanup(){ set +e; [[ -n "${DASHBOARD_PID:-}" ]] && kill "$DASHBOARD_PID" 2>/dev/null || true; [[ -n "${OPEN3D_PID:-}" ]] && kill "$OPEN3D_PID" 2>/dev/null || true; [[ -n "${CONTROLLER_PID:-}" ]] && kill "$CONTROLLER_PID" 2>/dev/null || true; [[ -n "${TF_PID:-}" ]] && kill "$TF_PID" 2>/dev/null || true; if [[ "${OWN_NATIVE:-0}" == "1" && -n "${NATIVE_PID:-}" ]]; then kill "$NATIVE_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT INT TERM

# Reuse a manually started native stack when the topic already exists. Otherwise
# start exactly one stack and wait for the publisher without the ROS2 daemon.
ros2 topic info --no-daemon "$TOPIC" >/tmp/ffem_lidar_topic_check.$$ 2>&1 || true
if grep -q "Publisher count: 1" /tmp/ffem_lidar_topic_check.$$; then
  echo "[PASS] Existing native CARLA LiDAR publisher discovered"
else
  rm -f /tmp/ffem_lidar_topic_check.$$ || true
  echo "Starting native CARLA ROS2 stack..."
  cd "$STACK_DIR"
  "$CARLA_PY" ros2_native.py --host 127.0.0.1 --port 2000 --file "$STACK_FILE" --verbose > "$ROOT/outputs/carla_native_ros2.log" 2>&1 &
  NATIVE_PID=$!; OWN_NATIVE=1; cd "$ROOT"
  echo "Waiting for $TOPIC ..."
  found=0
  for _ in $(seq 1 45); do
    if ros2 topic info --no-daemon "$TOPIC" 2>/dev/null | grep -q "Publisher count: 1"; then found=1; echo "[PASS] Native CARLA LiDAR publisher discovered"; break; fi
    if ! kill -0 "$NATIVE_PID" 2>/dev/null; then echo "ERROR: ros2_native.py exited"; tail -80 outputs/carla_native_ros2.log || true; exit 1; fi
    sleep 1
  done
  if [[ "$found" != "1" ]]; then echo "ERROR: LiDAR publisher was not discovered within 45 seconds."; tail -80 outputs/carla_native_ros2.log || true; exit 1; fi
fi
rm -f /tmp/ffem_lidar_topic_check.$$ || true
ros2 topic info --no-daemon "$TOPIC"

CONTROLLER="${FFEM_CONTROLLER:-none}"
USE_CARLA_TF="${FFEM_USE_CARLA_TF:-0}"
ENABLE_OPEN3D="${FFEM_OPEN3D:-1}"
ENABLE_DASHBOARD="${FFEM_DASHBOARD:-1}"
DEVICE="${FFEM_DEVICE:-auto}"
MIN_FREE_VRAM_MB="${FFEM_MIN_FREE_VRAM_MB:-1024}"
ENABLE_RVIZ="${FFEM_RVIZ:-1}"

MAP_FRAME="lidar"
USE_TF="false"
if [[ "$USE_CARLA_TF" == "1" ]]; then
  MAP_FRAME="map"
  USE_TF="true"
  echo "Starting CARLA LiDAR TF broadcaster..."
  "$CARLA_PY" "$ROOT/sim/carla/carla_tf_broadcaster.py" \
    --host 127.0.0.1 --port 2000 --map-frame map --sensor-frame lidar \
    > "$ROOT/outputs/carla_tf.log" 2>&1 &
  TF_PID=$!
  sleep 1
  if ! kill -0 "$TF_PID" 2>/dev/null; then
    echo "ERROR: CARLA TF broadcaster failed:"
    cat "$ROOT/outputs/carla_tf.log" || true
    exit 1
  fi
  echo "[PASS] CARLA LiDAR TF broadcaster is running"
fi

case "$CONTROLLER" in
  ffem)
    "$CARLA_PY" "$ROOT/sim/carla/follow_ffem_path.py" \
      --host 127.0.0.1 --port 2000 \
      --target-speed "${FFEM_TARGET_SPEED:-5.0}" \
      > "$ROOT/outputs/ffem_controller.log" 2>&1 &
    CONTROLLER_PID=$!
    ;;
  behavior)
    "$CARLA_PY" "$ROOT/sim/carla/drive_ego.py" \
      --host 127.0.0.1 --port 2000 \
      --target-speed "${FFEM_BEHAVIOR_SPEED:-18.0}" \
      > "$ROOT/outputs/carla_drive.log" 2>&1 &
    CONTROLLER_PID=$!
    ;;
  none)
    echo "Controller disabled."
    ;;
  *)
    echo "ERROR: unknown FFEM_CONTROLLER=$CONTROLLER"
    echo "Expected: ffem, behavior, none"
    exit 1
    ;;
esac

if [[ -n "$CONTROLLER_PID" ]]; then
  sleep 2
  if ! kill -0 "$CONTROLLER_PID" 2>/dev/null; then
    echo "ERROR: controller failed to start:"
    cat "$ROOT/outputs/ffem_controller.log" 2>/dev/null || true
    cat "$ROOT/outputs/carla_drive.log" 2>/dev/null || true
    exit 1
  fi
  echo "[PASS] Controller process is running: $CONTROLLER"
fi

if [[ "$ENABLE_OPEN3D" == "1" ]]; then
  echo "Starting optional Open3D viewer..."
  python3 "$ROOT/scripts/live_open3d_ros2.py" --topic "$TOPIC" \
    > "$ROOT/outputs/open3d_live.log" 2>&1 &
  OPEN3D_PID=$!
fi

if [[ "$ENABLE_DASHBOARD" == "1" ]]; then
  if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 1 \
      http://127.0.0.1:8765/ >/dev/null 2>&1; then
    echo "Dashboard already running: http://127.0.0.1:8765/"
  else
    python3 "$ROOT/scripts/live_dashboard.py" \
      > "$ROOT/outputs/dashboard.log" 2>&1 &
    DASHBOARD_PID=$!
    sleep 2
    if kill -0 "$DASHBOARD_PID" 2>/dev/null; then
      echo "Dashboard: http://127.0.0.1:8765/"
    else
      echo "WARNING: dashboard failed to start; continuing with core stack."
    fi
  fi
fi

ros2 launch ffem_lidar_mapping ffem_integrated.launch.py \
  input_topic:="$TOPIC" \
  map_frame:="$MAP_FRAME" \
  use_tf:="$USE_TF" \
  model_backend:=torch_range \
  checkpoint:="$CHECKPOINT" \
  range_height:="${FFEM_RANGE_HEIGHT:-16}" \
  range_width:="${FFEM_RANGE_WIDTH:-512}" \
  max_points_per_frame:="${FFEM_MAX_POINTS:-12000}" \
  max_active_cells:="${FFEM_MAX_ACTIVE_CELLS:-12000}" \
  max_topology_changes:="${FFEM_MAX_TOPOLOGY_CHANGES:-24}" \
  queue_depth:="${FFEM_QUEUE_DEPTH:-2}" \
  enable_rerun:="${FFEM_RERUN:-true}" \
  recording:="$ROOT/outputs/carla_ffem.rrd"

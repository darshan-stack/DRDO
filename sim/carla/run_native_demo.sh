#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CARLA_ROOT="${CARLA_ROOT:-$HOME/CARLA}"
CARLA_PY="${CARLA_PY:-$CARLA_ROOT/carla_env/bin/python3}"
STACK_DIR="$CARLA_ROOT/PythonAPI/examples/ros2"
STACK_FILE="${STACK_FILE:-$STACK_DIR/stack.json}"
TOPIC="${CARLA_LIDAR_TOPIC:-/carla/hero/lidar/point_cloud}"
CHECKPOINT="${FFEM_CHECKPOINT:-$ROOT/models/checkpoints/semanticposs_range_model.pt}"

# Source environment files before enabling nounset: ROS 2 Humble setup may
# reference optional variables that are intentionally unset.
source "$ROOT/sim/carla/native_env.sh"
source /opt/ros/humble/setup.bash
if [[ -f "$ROOT/install/setup.bash" ]]; then
  source "$ROOT/install/setup.bash"
fi
set -u
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

if ! command -v colcon >/dev/null 2>&1; then
  echo "ERROR: colcon is not installed/available. Source ROS 2 Humble first."
  exit 1
fi
if [[ ! -x "$CARLA_PY" ]]; then
  echo "ERROR: CARLA Python environment not found: $CARLA_PY"
  exit 1
fi
if [[ ! -f "$STACK_FILE" ]]; then
  echo "ERROR: Native CARLA stack file not found: $STACK_FILE"
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: Checkpoint not found: $CHECKPOINT"
  exit 1
fi

cd "$ROOT"
mkdir -p outputs

echo "=== FFEM Native CARLA Demo ==="
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "FASTDDS_BUILTIN_TRANSPORTS=$FASTDDS_BUILTIN_TRANSPORTS"
echo "LiDAR topic=$TOPIC"
echo "Checkpoint=$CHECKPOINT"

"$CARLA_PY" - <<'PY'
import carla
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(5.0)
print("CARLA server:", client.get_server_version())
print("CARLA map:", client.get_world().get_map().name)
PY

NATIVE_PID=""
DRIVE_PID=""
OPEN3D_PID=""
OWN_NATIVE=0

cleanup() {
  set +e
  [[ -n "${OPEN3D_PID:-}" ]] && kill "$OPEN3D_PID" 2>/dev/null || true
  [[ -n "${DRIVE_PID:-}" ]] && kill "$DRIVE_PID" 2>/dev/null || true
  if [[ "${OWN_NATIVE:-0}" == "1" && -n "${NATIVE_PID:-}" ]]; then
    kill "$NATIVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# A native stack may already be running from the manual bring-up test. Reuse
# it when the publisher is already visible; otherwise start exactly one stack.
ros2 topic info --no-daemon "$TOPIC" >/tmp/ffem_lidar_topic_check.$$ 2>&1 || true
if grep -q "Publisher count: 1" /tmp/ffem_lidar_topic_check.$$; then
  echo "[PASS] Existing native CARLA LiDAR publisher discovered"
else
  rm -f /tmp/ffem_lidar_topic_check.$$ || true
  echo "Starting native CARLA ROS2 stack..."
  cd "$STACK_DIR"
  "$CARLA_PY" ros2_native.py --host 127.0.0.1 --port 2000 --file "$STACK_FILE" --verbose > "$ROOT/outputs/carla_native_ros2.log" 2>&1 &
  NATIVE_PID=$!
  OWN_NATIVE=1
  cd "$ROOT"

  echo "Waiting for $TOPIC ..."
  found=0
  for _ in $(seq 1 45); do
    if ros2 topic info --no-daemon "$TOPIC" 2>/dev/null | grep -q "Publisher count: 1"; then
      found=1
      echo "[PASS] Native CARLA LiDAR publisher discovered"
      break
    fi
    if ! kill -0 "$NATIVE_PID" 2>/dev/null; then
      echo "ERROR: ros2_native.py exited. See outputs/carla_native_ros2.log"
      tail -80 outputs/carla_native_ros2.log || true
      exit 1
    fi
    sleep 1
  done

  if [[ "$found" != "1" ]]; then
    echo "ERROR: LiDAR publisher was not discovered within 45 seconds."
    echo "--- ROS graph check ---"
    ros2 topic info --no-daemon "$TOPIC" 2>&1 || true
    echo "--- ros2_native.py log ---"
    tail -80 outputs/carla_native_ros2.log || true
    exit 1
  fi
fi
rm -f /tmp/ffem_lidar_topic_check.$$ || true

ros2 topic info --no-daemon "$TOPIC"

"$CARLA_PY" "$ROOT/sim/carla/drive_ego.py" --speed-difference=0 > "$ROOT/outputs/carla_drive.log" 2>&1 &
DRIVE_PID=$!

python3 "$ROOT/scripts/live_open3d_ros2.py" --topic "$TOPIC" > "$ROOT/outputs/open3d_live.log" 2>&1 &
OPEN3D_PID=$!

ros2 launch ffem_lidar_mapping ffem_integrated.launch.py \
  input_topic:="$TOPIC" \
  map_frame:=lidar \
  use_tf:=false \
  model_backend:=torch_range \
  checkpoint:="$CHECKPOINT" \
  enable_rerun:=true \
  recording:="$ROOT/outputs/carla_ffem.rrd"

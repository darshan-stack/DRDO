#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/humble/setup.bash
source /opt/ffem/install/setup.bash

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"

if [[ "${1:-}" == "preflight" ]]; then
  shift
  exec python3 /opt/ffem/scripts/preflight.py "$@"
fi

if [[ -n "${FFEM_CHECKPOINT:-}" && ! -f "${FFEM_CHECKPOINT}" ]]; then
  echo "ERROR: FFEM checkpoint not found: ${FFEM_CHECKPOINT}" >&2
  echo "Mount the trained checkpoint, for example:" >&2
  echo "  - ./models/checkpoints:/models/checkpoints:ro" >&2
  exit 1
fi

echo "=== FFEM container ==="
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS}"
if [[ -n "${FFEM_CHECKPOINT:-}" ]]; then
  echo "FFEM_CHECKPOINT=${FFEM_CHECKPOINT}"
fi

exec "$@"

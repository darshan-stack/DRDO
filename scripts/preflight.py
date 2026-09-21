#!/usr/bin/env python3
"""FFEM production preflight checks.

Checks the local Python/ROS2 environment and, optionally, a live CARLA native
LiDAR publisher and an active FFEM mapper. This script fails closed when a
required dependency is missing.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED_TOPICS = (
    "/ffem_mapper/map/elevation",
    "/ffem_mapper/map/semantic",
    "/ffem_mapper/map/adaptive_cells",
    "/ffem_mapper/map/traversability",
    "/ffem_mapper/map/uncertainty",
    "/ffem_mapper/map/attention",
    "/ffem_mapper/map/moving_points",
    "/ffem_mapper/metrics",
    "/ffem_mapper/planning/risk",
    "/ffem_mapper/planning/path",
)


def run(command: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def require(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        failures.append(message)


def check_python(failures: list[str]) -> None:
    require(
        sys.version_info >= (3, 10),
        f"Python >= 3.10 ({sys.version.split()[0]})",
        failures,
    )
    for module in ("numpy",):
        try:
            __import__(module)
            require(True, f"Python module available: {module}", failures)
        except Exception as exc:
            require(False, f"Python module available: {module} ({exc})", failures)


def check_ros(failures: list[str]) -> None:
    ros2 = shutil.which("ros2")
    require(ros2 is not None, "ros2 executable is available", failures)
    if ros2 is None:
        return

    rc, output = run(["ros2", "node", "list"])
    require(rc == 0, "ROS2 graph is queryable", failures)
    if rc != 0:
        print(output[-2000:])
        return

    if any(line.strip() == "/ffem_mapper" for line in output.splitlines()):
        print("[PASS] /ffem_mapper is running")
    else:
        print("[INFO] /ffem_mapper is not running")


def check_topic(topic: str, failures: list[str]) -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        return
    rc, output = run(["ros2", "topic", "info", "-v", topic])
    require(
        rc == 0 and "Publisher count:" in output,
        f"ROS2 topic exists: {topic}",
        failures,
    )
    if rc == 0:
        print(output.strip())


def check_checkpoint(path: str | None, required: bool, failures: list[str]) -> None:
    if not path:
        if required:
            require(False, "A real FFEM checkpoint path is configured", failures)
        else:
            print("[INFO] No checkpoint requested")
        return
    checkpoint = Path(path).expanduser()
    require(checkpoint.is_file(), f"Checkpoint exists: {checkpoint}", failures)
    if checkpoint.is_file():
        try:
            import torch
            print(f"[PASS] PyTorch available: {torch.__version__}")
        except Exception as exc:
            require(False, f"PyTorch import succeeds ({exc})", failures)


def check_carla(host: str, port: int, failures: list[str]) -> None:
    try:
        import carla
    except Exception as exc:
        require(False, f"CARLA Python API import succeeds ({exc})", failures)
        return
    try:
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        world = client.get_world()
        version = client.get_server_version()
        print(f"[PASS] CARLA connection: {version} | map={world.get_map().name}")
        hero = next(
            (
                actor
                for actor in world.get_actors().filter("vehicle.*")
                if actor.attributes.get("role_name") == "hero"
            ),
            None,
        )
        require(hero is not None, "CARLA hero vehicle exists", failures)
    except Exception as exc:
        require(False, f"CARLA connection succeeds ({exc})", failures)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to the trained seven-class FFEM checkpoint.",
    )
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--check-carla", action="store_true")
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument(
        "--topic",
        default="/carla/hero/lidar/point_cloud",
    )
    parser.add_argument("--check-ffem", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    print("=== FFEM production preflight ===")
    check_python(failures)
    check_ros(failures)
    check_checkpoint(args.checkpoint, args.require_checkpoint, failures)

    if args.check_carla:
        check_carla(args.carla_host, args.carla_port, failures)
        check_topic(args.topic, failures)
    if args.check_ffem:
        rc, output = run(["ros2", "node", "info", "/ffem_mapper"])
        require(rc == 0, "/ffem_mapper introspection succeeds", failures)
        if rc == 0:
            require(
                args.topic in output,
                f"/ffem_mapper subscribes to {args.topic}",
                failures,
            )
            for topic in EXPECTED_TOPICS:
                require(
                    topic in output,
                    f"FFEM publishes {topic}",
                    failures,
                )

    if failures:
        print(f"\nFAILED checks: {len(failures)}")
        return 1
    print("\nAll requested preflight checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

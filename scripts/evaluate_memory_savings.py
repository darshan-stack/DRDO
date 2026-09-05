#!/usr/bin/env python3
"""Controlled uniform-vs-adaptive map storage experiment.

The experiment captures a fixed set of live ROS 2 LiDAR frames and replays
those exact frames through FFEM. The uniform baseline uses the *global finest
FFEM resolution* (0.05 m by default) everywhere, while FFEM keeps its
variable-resolution radial policy. Both methods are evaluated over the same
observations and the same map record-size assumption.

The memory number is a map-storage proxy, not whole-process RSS.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2

from ffem.ros2.pointcloud2_codec import decode_pointcloud2
from ffem.perception.factory import build_segmenter
from ffem.pipeline import FFEMConfig, FFEMPipeline


class Capture(Node):
    def __init__(self, topic: str, target: int):
        super().__init__("ffem_memory_experiment_capture")
        self.target = target
        self.frames: list[tuple[np.ndarray, np.ndarray]] = []
        qos = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.sub = self.create_subscription(PointCloud2, topic, self.cb, qos)
        self.get_logger().info(f"Capturing {target} frames from {topic}")

    def cb(self, msg: PointCloud2) -> None:
        if len(self.frames) >= self.target:
            return
        decoded = decode_pointcloud2(msg, remove_invalid=True)
        points = np.asarray(decoded.points, dtype=np.float32).reshape(-1, 3)
        intensity = (
            np.asarray(decoded.intensity, dtype=np.float32).reshape(-1)
            if decoded.intensity is not None
            else np.zeros(len(points), dtype=np.float32)
        )
        if len(points) == 0:
            return
        self.frames.append((points.copy(), intensity.copy()))
        if len(self.frames) == 1 or len(self.frames) % 10 == 0 or len(self.frames) == self.target:
            self.get_logger().info(
                f"captured {len(self.frames)}/{self.target} points={len(points)}"
            )


def capture_frames(topic: str, count: int, timeout: float) -> list[tuple[np.ndarray, np.ndarray]]:
    rclpy.init()
    node = Capture(topic, count)
    deadline = time.monotonic() + timeout
    try:
        while rclpy.ok() and len(node.frames) < count and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        frames = list(node.frames)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if len(frames) < count:
        raise RuntimeError(
            f"Captured only {len(frames)}/{count} frames within {timeout:.1f}s"
        )
    return frames


def occupied_cells(frames: list[tuple[np.ndarray, np.ndarray]], cell_size: float) -> int:
    keys: set[tuple[int, int]] = set()
    for points, _ in frames:
        q = np.floor(points[:, :2] / cell_size).astype(np.int64)
        keys.update(map(tuple, q.tolist()))
    return len(keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/carla/hero/lidar/point_cloud")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--capture-timeout", type=float, default=60.0)
    ap.add_argument("--checkpoint", default="models/checkpoints/semanticposs_range_model.pt")
    ap.add_argument(
        "--uniform-cell",
        type=float,
        default=0.05,
        help="Global finest FFEM cell size used everywhere by the uniform baseline.",
    )
    ap.add_argument("--cell-bytes", type=int, default=64)
    ap.add_argument("--output", default="outputs/memory_savings.json")
    ap.add_argument("--save-frames", default="outputs/memory_experiment_frames.npz")
    args = ap.parse_args()

    if args.frames < 2:
        raise SystemExit("--frames must be at least 2")
    if args.uniform_cell <= 0 or args.cell_bytes <= 0:
        raise SystemExit("--uniform-cell and --cell-bytes must be positive")

    print("=== FFEM Controlled Memory Experiment ===")
    print(f"topic={args.topic}")
    print(f"frames={args.frames}")
    print(f"uniform_global_finest_cell={args.uniform_cell:.3f} m")
    print(f"cell_record_size={args.cell_bytes} bytes")
    print("baseline=uniform global finest resolution vs FFEM variable radial resolution")

    frames = capture_frames(args.topic, args.frames, args.capture_timeout)

    save_path = Path(args.save_frames)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        points=np.array([p for p, _ in frames], dtype=object),
        intensity=np.array([i for _, i in frames], dtype=object),
    )

    print("Replaying the exact captured frames through FFEM...")
    segmenter, selected = build_segmenter("torch_range", args.checkpoint, 7)
    cfg = FFEMConfig(
        finest_cell_size=args.uniform_cell,
        max_active_cells=200_000,
    )
    pipeline = FFEMPipeline(config=cfg, segmenter=segmenter)

    uniform_keys: set[tuple[int, int]] = set()
    rows: list[dict[str, float | int]] = []
    for frame_id, (points, intensity) in enumerate(frames, start=1):
        q = np.floor(points[:, :2] / args.uniform_cell).astype(np.int64)
        uniform_keys.update(map(tuple, q.tolist()))

        out = pipeline.process_points(points, intensity, frame=frame_id)
        adaptive_cells = int(out["stats"]["active_cells"])
        uniform_cells = int(len(uniform_keys))
        reduction = 100.0 * (1.0 - adaptive_cells / uniform_cells) if uniform_cells else 0.0
        rows.append(
            {
                "frame": frame_id,
                "points": int(len(points)),
                "uniform_cells": uniform_cells,
                "adaptive_cells": adaptive_cells,
                "uniform_bytes": uniform_cells * args.cell_bytes,
                "adaptive_bytes": adaptive_cells * args.cell_bytes,
                "cell_reduction_pct": reduction,
                "memory_reduction_pct": reduction,
                "map_ms": float(out["stats"]["map_ms"]),
                "total_ms": float(out["stats"]["total_ms"]),
            }
        )
        if frame_id == 1 or frame_id % 10 == 0 or frame_id == len(frames):
            print(
                f"frame={frame_id:03d} uniform={uniform_cells} adaptive={adaptive_cells} "
                f"reduction={reduction:.2f}%"
            )

    uniform_final = int(rows[-1]["uniform_cells"])
    adaptive_final = int(rows[-1]["adaptive_cells"])
    final_reduction = float(rows[-1]["memory_reduction_pct"])
    adaptive_counts = np.array([r["adaptive_cells"] for r in rows], dtype=np.float64)
    uniform_counts = np.array([r["uniform_cells"] for r in rows], dtype=np.float64)

    report = {
        "experiment": "uniform_vs_ffem_adaptive",
        "topic": args.topic,
        "frames": len(frames),
        "checkpoint": selected,
        "uniform_global_finest_cell_size_m": args.uniform_cell,
        "cell_record_bytes": args.cell_bytes,
        "baseline": "global finest 0.05 m uniform XY cells over the exact same LiDAR frames",
        "ffem_resolution_policy": "0.05 m / 0.10 m / 0.25 m / 0.50 m radial bands with adaptive refinement",
        "memory_metric": "map-storage proxy; fixed bytes per stored cell; not whole-process RSS",
        "final": {
            "uniform_cells": uniform_final,
            "adaptive_cells": adaptive_final,
            "uniform_bytes": uniform_final * args.cell_bytes,
            "adaptive_bytes": adaptive_final * args.cell_bytes,
            "cell_reduction_pct": final_reduction,
            "memory_reduction_pct": final_reduction,
            "adaptive_vs_uniform_ratio": adaptive_final / max(uniform_final, 1),
        },
        "summary": {
            "mean_uniform_cells": float(uniform_counts.mean()),
            "mean_adaptive_cells": float(adaptive_counts.mean()),
            "peak_uniform_cells": int(uniform_counts.max()),
            "peak_adaptive_cells": int(adaptive_counts.max()),
            "mean_map_ms": float(np.mean([r["map_ms"] for r in rows])),
            "mean_total_ms": float(np.mean([r["total_ms"] for r in rows])),
        },
        "per_frame": rows,
        "captured_frames_file": str(save_path),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== RESULT ===")
    print(f"Uniform cells : {uniform_final}")
    print(f"FFEM cells    : {adaptive_final}")
    print(f"Cell reduction: {final_reduction:.2f}%")
    print(f"Memory proxy  : {final_reduction:.2f}%")
    print(f"Saved         : {out}")


if __name__ == "__main__":
    main()

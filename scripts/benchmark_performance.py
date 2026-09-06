#!/usr/bin/env python3
"""Benchmark FFEM processing latency on captured LiDAR frames.

Uses exact frames captured by the controlled memory experiment. Supports an
explicit fast profile so PS-26053 latency/FPS claims are backed by a repeatable
measurement using the same trained model and the same captured observations.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from ffem.perception.factory import build_segmenter
from ffem.pipeline import FFEMConfig, FFEMPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-file", default="outputs/memory_experiment_frames.npz")
    ap.add_argument("--checkpoint", default="models/checkpoints/semanticposs_range_model.pt")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--max-points", type=int, default=6000)
    ap.add_argument("--range-height", type=int, default=16)
    ap.add_argument("--range-width", type=int, default=512)
    ap.add_argument("--max-range", type=float, default=80.0)
    ap.add_argument("--max-active-cells", type=int, default=12000)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--output", default="outputs/performance_benchmark.json")
    args = ap.parse_args()

    data = np.load(args.frames_file, allow_pickle=True)
    points = list(data["points"])
    intensities = list(data["intensity"])
    count = min(args.frames, len(points))
    if count < 2:
        raise SystemExit("Need at least 2 captured frames for a meaningful benchmark.")

    segmenter, selected = build_segmenter(
        "torch_range",
        args.checkpoint,
        7,
        args.range_height,
        args.range_width,
        args.max_range,
    )
    pipeline = FFEMPipeline(
        config=FFEMConfig(max_active_cells=args.max_active_cells),
        segmenter=segmenter,
    )

    def prepare(i: int):
        p = np.asarray(points[i], dtype=np.float32)
        it = np.asarray(intensities[i], dtype=np.float32)
        if len(p) > args.max_points:
            idx = np.linspace(0, len(p) - 1, args.max_points, dtype=int)
            p = p[idx]
            it = it[idx]
        return p, it

    warm = min(args.warmup, count)
    for i in range(warm):
        p, it = prepare(i)
        pipeline.process_points(p, it, frame=i)

    # Reset the map so warmup does not inflate the measured steady-state map.
    pipeline.mapping.cells.clear()
    pipeline.mapping.events.clear()
    pipeline.history.clear()

    timings: list[float] = []
    map_timings: list[float] = []
    point_counts: list[int] = []
    active_counts: list[int] = []

    for i in range(count):
        p, it = prepare(i)
        t = time.perf_counter()
        out = pipeline.process_points(p, it, frame=i + 1)
        elapsed = (time.perf_counter() - t) * 1000.0
        timings.append(elapsed)
        map_timings.append(float(out["stats"]["map_ms"]))
        point_counts.append(len(p))
        active_counts.append(int(out["stats"]["active_cells"]))

    mean_ms = statistics.fmean(timings)
    median_ms = statistics.median(timings)
    p95_ms = float(np.percentile(timings, 95))
    fps = 1000.0 / max(mean_ms, 1e-9)

    report = {
        "profile": "ps26053_fast",
        "frames": count,
        "warmup_frames": warm,
        "checkpoint": selected,
        "range_image": {"height": args.range_height, "width": args.range_width, "max_range_m": args.max_range},
        "max_points_per_frame": args.max_points,
        "max_active_cells": args.max_active_cells,
        "mean_points": statistics.fmean(point_counts),
        "mean_active_cells": statistics.fmean(active_counts),
        "peak_active_cells": max(active_counts),
        "mean_total_ms": mean_ms,
        "median_total_ms": median_ms,
        "p95_total_ms": p95_ms,
        "mean_map_ms": statistics.fmean(map_timings),
        "p95_map_ms": float(np.percentile(map_timings, 95)),
        "estimated_fps": fps,
        "fps_definition": "1000 / mean end-to-end FFEM processing milliseconds",
        "note": "Measured on the current workstation using the exact captured LiDAR frames. FPS is a measured software benchmark, not a certified real-time guarantee.",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("=== FFEM PS 26053 PERFORMANCE BENCHMARK ===")
    print(f"profile         : {report['profile']}")
    print(f"frames          : {count}")
    print(f"mean points     : {report['mean_points']:.0f}")
    print(f"range image     : {args.range_height} x {args.range_width}")
    print(f"mean total      : {mean_ms:.2f} ms")
    print(f"median total    : {median_ms:.2f} ms")
    print(f"p95 total       : {p95_ms:.2f} ms")
    print(f"mean map        : {report['mean_map_ms']:.2f} ms")
    print(f"estimated FPS   : {fps:.2f}")
    print(f"saved           : {out}")


if __name__ == "__main__":
    main()

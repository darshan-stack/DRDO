#!/usr/bin/env python3
"""PS-26053 three-way latency/RAM benchmark on identical LiDAR frames.

Modes:
  uniform_fine   global 5 cm map
  uniform_coarse global 50 cm map
  ffem           production adaptive 5/10/25/50 cm policy

Each mode runs in a fresh child process so peak RSS is isolated. The same
captured frames and the same semantic model are used for every mode.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def make_pipeline(mode: str, checkpoint: str, device: str, height: int, width: int, max_range: float):
    from ffem.perception.factory import build_segmenter
    from ffem.pipeline import FFEMConfig, FFEMPipeline

    if mode == "uniform_fine":
        cfg = FFEMConfig(
            base_cell_size=0.05,
            finest_cell_size=0.05,
            max_level=0,
            max_active_cells=1_000_000,
        )
    elif mode == "uniform_coarse":
        cfg = FFEMConfig(
            base_cell_size=0.50,
            finest_cell_size=0.50,
            max_level=0,
            max_active_cells=1_000_000,
        )
    else:
        cfg = FFEMConfig(
            base_cell_size=1.0,
            finest_cell_size=0.05,
            max_level=2,
            max_active_cells=20_000,
        )

    seg, selected = build_segmenter(
        "torch_range",
        checkpoint,
        7,
        height,
        width,
        max_range,
        device=device,
        min_free_vram_mb=512,
    )
    pipe = FFEMPipeline(cfg, segmenter=seg)

    if mode != "ffem":
        from ffem.mapping.radial_resolution import RadialResolutionPolicy, ResolutionBand

        size = 0.05 if mode == "uniform_fine" else 0.50
        pipe.mapping.radial = RadialResolutionPolicy(
            (ResolutionBand(100.0, size, "uniform"),)
        )
        pipe.mapping._limits = np.asarray([100.0], dtype=np.float32)
        pipe.mapping._sizes = np.asarray([size], dtype=np.float32)

    return pipe, selected


def load_frames(path: str, count: int, max_points: int):
    data = np.load(path, allow_pickle=True)
    points = list(data["points"])[:count]
    intensities = list(data["intensity"])[:count]
    if len(points) < 2:
        raise RuntimeError("Need at least two captured frames.")
    prepared = []
    for p, it in zip(points, intensities):
        p = np.asarray(p, dtype=np.float32).reshape(-1, 3)
        it = np.asarray(it, dtype=np.float32).reshape(-1)
        if len(p) != len(it):
            raise RuntimeError("Point/intensity lengths do not match.")
        if len(p) > max_points:
            idx = np.linspace(0, len(p) - 1, max_points, dtype=np.int64)
            p, it = p[idx], it[idx]
        prepared.append((p, it))
    return prepared


def worker(args) -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")

    from ffem.planning.local_planner import LocalRiskPlanner

    frames = load_frames(args.frames_file, args.frames, args.max_points)
    pipe, selected = make_pipeline(
        args.mode,
        args.checkpoint,
        args.device,
        args.range_height,
        args.range_width,
        args.max_range,
    )
    planner = LocalRiskPlanner()

    warmup = min(args.warmup, len(frames))
    for i in range(warmup):
        pipe.process_points(frames[i][0], frames[i][1], frame=i)

    pipe.reset()

    segmenter = getattr(pipe, "perception", None)
    if hasattr(segmenter, "torch") and getattr(segmenter, "device", "") == "cuda":
        segmenter.torch.cuda.reset_peak_memory_stats()

    latencies = []
    map_latencies = []
    active_cells = []
    point_counts = []

    for i, (points, intensity) in enumerate(frames, start=1):
        start = time.perf_counter()
        result = pipe.process_points(points, intensity, frame=i)
        preliminary = planner.plan(pipe.mapping)
        pipe.mapping.apply_planning_feedback(
            preliminary["points"], preliminary["risk_profile"], i
        )
        planner.plan(pipe.mapping)
        latencies.append((time.perf_counter() - start) * 1000.0)
        map_latencies.append(float(result["stats"]["map_ms"]))
        active_cells.append(int(result["stats"]["active_cells"]))
        point_counts.append(len(points))

    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gpu_mib = None
    if hasattr(segmenter, "torch") and getattr(segmenter, "device", "") == "cuda":
        peak_gpu_mib = float(
            segmenter.torch.cuda.max_memory_allocated() / (1024 ** 2)
        )

    result = {
        "mode": args.mode,
        "frames": len(frames),
        "checkpoint": selected,
        "device": getattr(segmenter, "device", args.device),
        "mean_points": statistics.fmean(point_counts),
        "mean_latency_ms": statistics.fmean(latencies),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "max_latency_ms": max(latencies),
        "estimated_fps": 1000.0 / max(statistics.fmean(latencies), 1e-9),
        "mean_map_ms": statistics.fmean(map_latencies),
        "p95_map_ms": float(np.percentile(map_latencies, 95)),
        "mean_active_cells": statistics.fmean(active_cells),
        "final_active_cells": active_cells[-1],
        "peak_active_cells": max(active_cells),
        "peak_rss_mib": rss_kib / 1024.0,
        "peak_gpu_allocated_mib": peak_gpu_mib,
    }
    print("RESULT_JSON:" + json.dumps(result))


def run_worker(args, mode: str) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--mode",
        mode,
        "--frames-file",
        args.frames_file,
        "--checkpoint",
        args.checkpoint,
        "--frames",
        str(args.frames),
        "--max-points",
        str(args.max_points),
        "--range-height",
        str(args.range_height),
        "--range-width",
        str(args.range_width),
        "--max-range",
        str(args.max_range),
        "--warmup",
        str(args.warmup),
        "--device",
        args.device,
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"{mode} benchmark failed with exit code {proc.returncode}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line.split(":", 1)[1])
    raise RuntimeError(f"No RESULT_JSON found for {mode}. Output was:\n{proc.stdout}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-file", default="outputs/memory_experiment_frames.npz")
    ap.add_argument("--checkpoint", default="models/checkpoints/semanticposs_range_model.pt")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--max-points", type=int, default=6000)
    ap.add_argument("--range-height", type=int, default=16)
    ap.add_argument("--range-width", type=int, default=512)
    ap.add_argument("--max-range", type=float, default=80.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--output", default="outputs/ps26053_comparison.json")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mode", choices=["uniform_fine", "uniform_coarse", "ffem"])
    args = ap.parse_args()

    if args.worker:
        if not args.mode:
            raise SystemExit("--mode is required in --worker mode")
        worker(args)
        return

    modes = ["uniform_fine", "uniform_coarse", "ffem"]
    results = [run_worker(args, mode) for mode in modes]
    by_mode = {r["mode"]: r for r in results}

    baseline = by_mode["uniform_fine"]
    out_ffem = by_mode["ffem"]
    latency_reduction = 100.0 * (
        1.0 - out_ffem["mean_latency_ms"] / baseline["mean_latency_ms"]
    )
    p95_reduction = 100.0 * (
        1.0 - out_ffem["p95_latency_ms"] / baseline["p95_latency_ms"]
    )
    rss_reduction = 100.0 * (
        1.0 - out_ffem["peak_rss_mib"] / baseline["peak_rss_mib"]
    )
    cell_reduction = 100.0 * (
        1.0 - out_ffem["final_active_cells"] / baseline["final_active_cells"]
    )

    report = {
        "experiment": "PS-26053 uniform-5cm vs uniform-50cm vs FFEM",
        "same_frames": args.frames_file,
        "same_checkpoint": args.checkpoint,
        "range_image": [args.range_height, args.range_width],
        "max_points_per_frame": args.max_points,
        "device_request": args.device,
        "latency_definition": (
            "process_points + preliminary local plan + planning-to-mapping feedback "
            "+ final local plan; excludes ROS serialization, visualization, and CARLA rendering"
        ),
        "memory_definition": "child-process peak resident set size (RSS); each mode runs in a fresh process",
        "results": results,
        "ffem_vs_uniform_5cm": {
            "mean_latency_reduction_pct": latency_reduction,
            "p95_latency_reduction_pct": p95_reduction,
            "peak_rss_reduction_pct": rss_reduction,
            "final_active_cell_reduction_pct": cell_reduction,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("=== PS-26053 THREE-WAY COMPARISON ===")
    for r in results:
        print(
            f"{r['mode']:16s} "
            f"mean={r['mean_latency_ms']:.2f} ms "
            f"p50={r['p50_latency_ms']:.2f} ms "
            f"p95={r['p95_latency_ms']:.2f} ms "
            f"FPS={r['estimated_fps']:.2f} "
            f"RSS={r['peak_rss_mib']:.1f} MiB "
            f"cells={r['final_active_cells']}"
        )
    print(f"FFEM mean-latency reduction vs uniform 5cm: {latency_reduction:.2f}%")
    print(f"FFEM P95-latency reduction vs uniform 5cm:  {p95_reduction:.2f}%")
    print(f"FFEM peak-RSS reduction vs uniform 5cm:     {rss_reduction:.2f}%")
    print(f"FFEM active-cell reduction vs uniform 5cm: {cell_reduction:.2f}%")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

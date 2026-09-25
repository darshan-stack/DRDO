#!/usr/bin/env python3
"""Run the PS-26053 baseline/ablation matrix on identical captured LiDAR frames.

The matrix separates structural/performance effects from semantic/elevation
ground truth. Ground-truth metrics are intentionally not fabricated here;
those belong to SemanticPOSS/CARLA evaluators.
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


MODES = (
    "uniform_fine",
    "uniform_coarse",
    "geometry_only",
    "semantic_only",
    "ffem",
    "no_predictive_dilation",
    "no_planner_feedback",
)


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


def make_pipeline(mode, checkpoint, device, height, width, max_range, max_active_cells):
    from ffem.perception.factory import build_segmenter
    from ffem.pipeline import FFEMConfig, FFEMPipeline

    cfg = FFEMConfig(max_active_cells=max_active_cells)

    if mode == "uniform_fine":
        cfg.base_cell_size = 0.05
        cfg.finest_cell_size = 0.05
        cfg.max_level = 0
    elif mode == "uniform_coarse":
        cfg.base_cell_size = 0.50
        cfg.finest_cell_size = 0.50
        cfg.max_level = 0
    elif mode == "geometry_only":
        cfg.semantic_weight = 0.0
        cfg.motion_weight = 0.0
        cfg.traversability_weight = 0.0
        cfg.geometry_weight = 1.0
        cfg.range_weight = 0.0
        cfg.planning_weight = 0.0
    elif mode == "semantic_only":
        cfg.semantic_weight = 1.0
        cfg.motion_weight = 0.0
        cfg.traversability_weight = 0.0
        cfg.geometry_weight = 0.0
        cfg.range_weight = 0.0
        cfg.planning_weight = 0.0
    elif mode == "no_predictive_dilation":
        cfg.predictive_dilation_frames = 0
        cfg.predictive_dilation_radius_m = 0.0

    if mode == "no_planner_feedback":
        cfg.planning_weight = 0.0

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
    pipe = FFEMPipeline(config=cfg, segmenter=seg)

    if mode in {"uniform_fine", "uniform_coarse"}:
        from ffem.mapping.radial_resolution import RadialResolutionPolicy, ResolutionBand

        size = 0.05 if mode == "uniform_fine" else 0.50
        pipe.mapping.radial = RadialResolutionPolicy(
            (ResolutionBand(100.0, size, "uniform"),)
        )
        pipe.mapping._limits = np.asarray([100.0], dtype=np.float32)
        pipe.mapping._sizes = np.asarray([size], dtype=np.float32)

    return pipe, selected


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
        args.max_active_cells,
    )
    planner = LocalRiskPlanner()

    warmup = min(args.warmup, len(frames))
    for i in range(warmup):
        pipe.process_points(frames[i][0], frames[i][1], frame=i)
        if args.mode != "no_planner_feedback":
            preliminary = planner.plan(pipe.mapping)
            pipe.mapping.apply_planning_feedback(
                preliminary["points"], preliminary["risk_profile"], i
            )
    pipe.reset()

    latencies = []
    map_latencies = []
    active_cells = []
    topology_changes = []
    moving_points = []
    plan_risks = []
    feedback_refines = []
    point_counts = []

    for i, (points, intensity) in enumerate(frames, start=1):
        start = time.perf_counter()
        result = pipe.process_points(points, intensity, frame=i)

        feedback_changes = 0
        final_plan = planner.plan(pipe.mapping)
        if args.mode != "no_planner_feedback":
            feedback_changes = pipe.mapping.apply_planning_feedback(
                final_plan["points"], final_plan["risk_profile"], i
            )
            final_plan = planner.plan(pipe.mapping)

        latencies.append((time.perf_counter() - start) * 1000.0)
        map_latencies.append(float(result["stats"]["map_ms"]))
        active_cells.append(int(result["stats"]["active_cells"]))
        topology_changes.append(int(result["stats"]["topology_changes"]))
        moving_points.append(int(result["stats"]["moving_points"]))
        feedback_refines.append(int(feedback_changes))
        plan_risks.append(float(final_plan["mean_risk"]))
        point_counts.append(len(points))

    segmenter = getattr(pipe, "perception", None)
    peak_gpu_mib = None
    if hasattr(segmenter, "torch") and getattr(segmenter, "device", "") == "cuda":
        peak_gpu_mib = float(
            segmenter.torch.cuda.max_memory_allocated() / (1024 ** 2)
        )

    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
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
        "mean_topology_changes": statistics.fmean(topology_changes),
        "mean_moving_points": statistics.fmean(moving_points),
        "mean_feedback_refines": statistics.fmean(feedback_refines),
        "mean_plan_risk": statistics.fmean(plan_risks),
        "peak_rss_mib": rss_mib,
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
        "--max-active-cells",
        str(args.max_active_cells),
        "--warmup",
        str(args.warmup),
        "--device",
        args.device,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, env=os.environ.copy())
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"{mode} ablation failed with exit code {proc.returncode}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line.split(":", 1)[1])
    raise RuntimeError(f"No RESULT_JSON returned for {mode}.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-file", default="outputs/memory_experiment_frames.npz")
    ap.add_argument(
        "--checkpoint",
        default="models/checkpoints/semanticposs_range_model.pt",
    )
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--max-points", type=int, default=6000)
    ap.add_argument("--range-height", type=int, default=16)
    ap.add_argument("--range-width", type=int, default=512)
    ap.add_argument("--max-range", type=float, default=80.0)
    ap.add_argument("--max-active-cells", type=int, default=12000)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument(
        "--output",
        default="outputs/ps26053_ablation_matrix.json",
    )
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mode", choices=MODES)
    args = ap.parse_args()

    if args.worker:
        if args.mode is None:
            raise SystemExit("--mode is required in --worker mode")
        worker(args)
        return

    results = [run_worker(args, mode) for mode in MODES]
    baseline = results[0]
    for item in results:
        item["mean_latency_reduction_vs_uniform_fine_pct"] = 100.0 * (
            1.0 - item["mean_latency_ms"] / baseline["mean_latency_ms"]
        )
        item["p95_latency_reduction_vs_uniform_fine_pct"] = 100.0 * (
            1.0 - item["p95_latency_ms"] / baseline["p95_latency_ms"]
        )
        item["rss_reduction_vs_uniform_fine_pct"] = 100.0 * (
            1.0 - item["peak_rss_mib"] / baseline["peak_rss_mib"]
        )
        item["active_cell_reduction_vs_uniform_fine_pct"] = 100.0 * (
            1.0 - item["final_active_cells"] / baseline["final_active_cells"]
        )

    report = {
        "experiment": "PS-26053 baseline and ablation matrix",
        "same_frames": args.frames_file,
        "same_checkpoint": args.checkpoint,
        "measurement_definition": (
            "Latency is process_points + planning + optional feedback/final plan. "
            "RSS is child-process peak resident memory. "
            "No semantic/elevation accuracy is claimed without external ground truth."
        ),
        "modes": list(MODES),
        "results": results,
        "ground_truth_metrics": {
            "status": "not_available_in_this_replay",
            "required_sources": ["SemanticPOSS labels", "CARLA semantic LiDAR"],
            "metrics": [
                "semantic mIoU",
                "elevation RMSE/MAE",
                "traversability accuracy",
                "false-drivable rate",
                "false-obstacle rate",
            ],
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("=== PS-26053 ABLATION MATRIX ===")
    for r in results:
        print(
            f"{r['mode']:24s} "
            f"mean={r['mean_latency_ms']:.2f} ms "
            f"p95={r['p95_latency_ms']:.2f} ms "
            f"FPS={r['estimated_fps']:.2f} "
            f"RSS={r['peak_rss_mib']:.1f} MiB "
            f"cells={r['final_active_cells']} "
            f"feedback={r['mean_feedback_refines']:.1f}"
        )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

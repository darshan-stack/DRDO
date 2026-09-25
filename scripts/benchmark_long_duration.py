#!/usr/bin/env python3
"""Long-duration FFEM replay stability benchmark on a captured frame set."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from ffem.perception.factory import build_segmenter
from ffem.pipeline import FFEMConfig, FFEMPipeline
from ffem.planning.local_planner import LocalRiskPlanner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-file", default="outputs/memory_experiment_frames.npz")
    ap.add_argument("--checkpoint", default="models/checkpoints/semanticposs_range_model.pt")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--max-points", type=int, default=6000)
    ap.add_argument("--range-height", type=int, default=16)
    ap.add_argument("--range-width", type=int, default=512)
    ap.add_argument("--max-range", type=float, default=80.0)
    ap.add_argument("--max-active-cells", type=int, default=12000)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--output", default="outputs/long_duration_stability.json")
    args = ap.parse_args()

    data = np.load(args.frames_file, allow_pickle=True)
    points = list(data["points"])
    intensity = list(data["intensity"])
    if len(points) < 2:
        raise SystemExit("Need at least two captured frames.")

    segmenter, selected = build_segmenter(
        "torch_range",
        args.checkpoint,
        7,
        args.range_height,
        args.range_width,
        args.max_range,
        device=args.device,
    )
    pipeline = FFEMPipeline(
        FFEMConfig(max_active_cells=args.max_active_cells),
        segmenter=segmenter,
    )
    planner = LocalRiskPlanner()

    errors = []
    latencies = []
    active = []
    events = []
    start = time.perf_counter()

    for i in range(args.iterations):
        idx = i % len(points)
        p = np.asarray(points[idx], dtype=np.float32)
        it = np.asarray(intensity[idx], dtype=np.float32)
        if len(p) > args.max_points:
            sample = np.linspace(
                0, len(p) - 1, args.max_points, dtype=np.int64
            )
            p, it = p[sample], it[sample]
        t0 = time.perf_counter()
        try:
            result = pipeline.process_points(p, it, frame=i + 1)
            plan = planner.plan(pipeline.mapping)
            pipeline.mapping.apply_planning_feedback(
                plan["points"], plan["risk_profile"], i + 1
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            active.append(int(result["stats"]["active_cells"]))
            events.append(len(pipeline.mapping.events))
        except Exception as exc:
            errors.append(
                {
                    "iteration": i + 1,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    elapsed = time.perf_counter() - start
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    report = {
        "frames_file": str(Path(args.frames_file).resolve()),
        "checkpoint": str(Path(selected or args.checkpoint).resolve()),
        "device": getattr(segmenter, "device", args.device),
        "iterations_requested": args.iterations,
        "iterations_completed": len(latencies),
        "error_count": len(errors),
        "elapsed_s": elapsed,
        "throughput_iterations_per_s": len(latencies) / max(elapsed, 1e-9),
        "latency": {
            "mean_ms": float(np.mean(latencies)) if latencies else None,
            "p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
            "p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "max_ms": float(max(latencies)) if latencies else None,
        },
        "active_cells": {
            "mean": float(np.mean(active)) if active else None,
            "max": int(max(active)) if active else None,
            "final": int(active[-1]) if active else None,
        },
        "event_history": {
            "final_size": int(events[-1]) if events else 0,
            "configured_bound": int(pipeline.cfg.max_event_history),
        },
        "peak_rss_mib": rss_mib,
        "stable": len(errors) == 0 and len(latencies) == args.iterations,
        "errors": errors[:20],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["stable"] else 1)


if __name__ == "__main__":
    main()

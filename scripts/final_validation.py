#!/usr/bin/env python3
"""Run final-round FFEM architecture checks and report available evidence.

This script never invents accuracy numbers. It reports hierarchy/planning
checks immediately and reads formal SemanticPOSS/CARLA validation JSON files
only when those experiments have actually been run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ffem.pipeline import AdaptiveElevationMap, FFEMConfig


def architecture_checks() -> dict:
    cfg = FFEMConfig(max_level=2, refine_threshold=0.50, planning_weight=0.20)
    mapping = AdaptiveElevationMap(cfg)
    root_id = mapping._ensure_root(3.0, 1.0)
    root = mapping.nodes[root_id]
    root.attention = 0.8
    split_ok = mapping._split(root)
    parent_child_ok = split_ok and len(root.children) == 4 and all(
        mapping.nodes[cid].parent == root_id for cid in root.children
    )

    feedback_changes = mapping.apply_planning_feedback(
        np.asarray([[3.0, 1.0], [4.0, 1.0]], dtype=np.float32),
        np.asarray([1.0, 0.9], dtype=np.float32),
        frame=1,
    )
    feedback_ok = feedback_changes > 0 or any(
        mapping.nodes[cid].planning_criticality > 0 for cid in root.children if cid in mapping.nodes
    )

    return {
        "explicit_parent_child_4ary": bool(parent_child_ok),
        "planning_feedback_refinement": bool(feedback_ok),
        "hierarchy_nodes_after_check": int(len(mapping.nodes)),
        "active_leaves_after_check": int(len(mapping.leaves)),
    }


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic-eval", default="outputs/semanticposs_eval.json")
    ap.add_argument("--carla-eval", default="outputs/carla_generalization.json")
    ap.add_argument("--performance", default="outputs/performance_benchmark.json")
    ap.add_argument("--memory", default="outputs/memory_savings.json")
    ap.add_argument("--output", default="outputs/final_validation_report.json")
    args = ap.parse_args()

    report = {
        "architecture": architecture_checks(),
        "evidence": {},
        "notes": [
            "Semantic accuracy is reported only from an actual SemanticPOSS evaluation JSON.",
            "CARLA generalization is reported only from the normal-LiDAR-vs-semantic-LiDAR experiment.",
            "Memory is a map-storage proxy, not whole-process RSS.",
            "The CARLA controller is a demonstration path follower, not a safety-rated controller.",
        ],
    }

    for name, path_str in (
        ("semanticposs", args.semantic_eval),
        ("carla_generalization", args.carla_eval),
        ("performance", args.performance),
        ("memory", args.memory),
    ):
        data = load_json(Path(path_str))
        report["evidence"][name] = data if data is not None else {"status": "not_run", "path": path_str}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("=== FFEM FINAL-ROUND VALIDATION ===")
    print("Hierarchy / parent-child:", "PASS" if report["architecture"]["explicit_parent_child_4ary"] else "FAIL")
    print("Planning feedback:", "PASS" if report["architecture"]["planning_feedback_refinement"] else "FAIL")
    for name, data in report["evidence"].items():
        if data.get("status") == "not_run":
            print(f"{name}: NOT RUN -> {data['path']}")
        else:
            print(f"{name}: available")
            for key in ("mean_iou", "overall_accuracy", "mIoU", "accuracy", "estimated_fps", "mean_total_ms", "memory_reduction_pct", "cell_reduction_pct"):
                if key in data:
                    print(f"  {key}: {data[key]}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

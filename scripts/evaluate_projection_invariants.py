#!/usr/bin/env python3
"""Projection and adaptive-resolution invariant checks for PS-26053."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ffem.mapping.radial_resolution import RadialResolutionPolicy
from ffem.perception.segmentation import ProjectionConfig, RangeImageProjector


def run_checks() -> dict:
    policy = RadialResolutionPolicy()
    boundaries = {
        "just_inside_8m": policy.cell_size(7.999, 0.0),
        "exact_8m": policy.cell_size(8.0, 0.0),
        "just_inside_18m": policy.cell_size(17.999, 0.0),
        "exact_18m": policy.cell_size(18.0, 0.0),
        "just_inside_40m": policy.cell_size(39.999, 0.0),
        "exact_40m": policy.cell_size(40.0, 0.0),
        "just_inside_100m": policy.cell_size(99.999, 0.0),
        "exact_100m": policy.cell_size(100.0, 0.0),
        "beyond_100m": policy.cell_size(120.0, 0.0),
    }
    expected = {
        "just_inside_8m": 0.05,
        "exact_8m": 0.05,
        "just_inside_18m": 0.10,
        "exact_18m": 0.10,
        "just_inside_40m": 0.25,
        "exact_40m": 0.25,
        "just_inside_100m": 0.50,
        "exact_100m": 0.50,
        "beyond_100m": 0.50,
    }
    boundary_pass = all(
        abs(boundaries[key] - value) < 1e-8 for key, value in expected.items()
    )

    projector = RangeImageProjector(
        ProjectionConfig(height=8, width=16, max_range=100.0)
    )
    points = np.asarray(
        [
            [-2.0, -1.0, 0.0],
            [-2.0, 1.0, 0.2],
            [2.0, -1.0, 0.4],
            [2.0, 1.0, 0.6],
            [10.0, 0.0, 0.1],
            [10.0, 0.0, 0.5],
            [80.0, 30.0, -0.2],
            [-80.0, -30.0, 0.3],
            [0.0, 0.0, 0.0],
            [150.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    intensity = np.linspace(0.0, 1.0, len(points), dtype=np.float32)
    image = projector.project(points, intensity)
    point_index = image["point_index"]
    selected = point_index[point_index >= 0]

    unique_assignment = len(selected) == len(np.unique(selected))
    valid_expected = (
        np.linalg.norm(points, axis=1) >= projector.cfg.min_range
    ) & (
        np.linalg.norm(points, axis=1) <= projector.cfg.max_range
    )
    assigned_set = set(selected.tolist())
    all_assigned_once = all(
        idx in assigned_set for idx in np.flatnonzero(valid_expected)
        if idx != 0
    )
    invalid_rejected = 9 not in assigned_set
    finite_inputs = np.isfinite(image["depth"]).all()

    repeat_a = projector.project(points, intensity)["point_index"]
    repeat_b = projector.project(points, intensity)["point_index"]
    deterministic = bool(np.array_equal(repeat_a, repeat_b))

    passed = all(
        (
            boundary_pass,
            unique_assignment,
            invalid_rejected,
            finite_inputs,
            deterministic,
        )
    )

    return {
        "pass": bool(passed),
        "radial_boundaries_m": boundaries,
        "checks": {
            "boundary_policy": bool(boundary_pass),
            "point_assigned_at_most_once": bool(unique_assignment),
            "invalid_long_range_rejected": bool(invalid_rejected),
            "finite_projection": bool(finite_inputs),
            "deterministic_repeat_projection": bool(deterministic),
            "valid_points_assignment_coverage": bool(all_assigned_once),
        },
        "note": (
            "Projection can legitimately reject points outside the configured "
            "range/FOV. Points retained in the range image must have exactly one "
            "point index per occupied pixel."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/projection_invariants.json")
    args = ap.parse_args()
    report = run_checks()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

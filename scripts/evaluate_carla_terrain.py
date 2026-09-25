#!/usr/bin/env python3
"""Evaluate FFEM elevation and traversability against CARLA semantic LiDAR.

The evaluator uses normal LiDAR as FFEM input and synchronized semantic LiDAR
as spatial ground truth. Ground/road/sidewalk/terrain tags provide drivable
truth; visible ground points provide per-cell elevation references.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ffem.perception.factory import build_segmenter
from ffem.pipeline import FFEMConfig, FFEMPipeline
from evaluate_carla_generalization import (
    normal_xyz_intensity,
    semantic_xyz_instance_tag,
)


GROUND_TAGS = {6, 7, 13, 18}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="models/checkpoints/semanticposs_range_model.pt",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--voxel", type=float, default=0.20)
    ap.add_argument("--max-points", type=int, default=6000)
    ap.add_argument("--traversability-threshold", type=float, default=0.35)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument(
        "--output",
        default="outputs/elevation_traversability_metrics.json",
    )
    args = ap.parse_args()

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    vehicles = list(world.get_actors().filter("vehicle.*"))
    if not vehicles:
        raise SystemExit("No hero/vehicle found in CARLA.")

    hero = next(
        (actor for actor in vehicles if actor.attributes.get("role_name") == "hero"),
        vehicles[0],
    )
    library = world.get_blueprint_library()
    raw_bp = library.find("sensor.lidar.ray_cast")
    semantic_bp = library.find("sensor.lidar.ray_cast_semantic")
    attrs = {
        "range": "80",
        "channels": "64",
        "points_per_second": "600000",
        "rotation_frequency": "20",
        "upper_fov": "10",
        "lower_fov": "-30",
        "sensor_tick": "0.05",
    }
    for key, value in attrs.items():
        if raw_bp.has_attribute(key):
            raw_bp.set_attribute(key, value)
        if semantic_bp.has_attribute(key):
            semantic_bp.set_attribute(key, value)

    transform = carla.Transform(carla.Location(z=2.4))
    raw_sensor = world.spawn_actor(raw_bp, transform, attach_to=hero)
    semantic_sensor = world.spawn_actor(
        semantic_bp,
        transform,
        attach_to=hero,
    )

    segmenter, selected = build_segmenter(
        "torch_range",
        args.checkpoint,
        7,
        device=args.device,
    )
    pipeline = FFEMPipeline(
        FFEMConfig(max_active_cells=20000, max_topology_changes=32),
        segmenter=segmenter,
    )

    raw_box = {}
    semantic_box = {}
    elevation_errors = []
    elevation_abs_errors = []
    cell_drivable = []
    cell_not_drivable = []
    disambiguation_cases = 0
    frames = 0

    def raw_callback(measurement):
        raw_box[int(measurement.frame)] = measurement

    def semantic_callback(measurement):
        semantic_box[int(measurement.frame)] = measurement

    raw_sensor.listen(raw_callback)
    semantic_sensor.listen(semantic_callback)

    print("Collecting synchronized CARLA LiDAR for terrain/elevation metrics...")
    try:
        deadline = time.time() + max(30.0, args.frames * 0.6 + 15.0)
        while frames < args.frames and time.time() < deadline:
            world.wait_for_tick(1.0)
            common = sorted(set(raw_box).intersection(semantic_box))
            while common and frames < args.frames:
                fid = common.pop(0)
                raw = raw_box.pop(fid)
                semantic = semantic_box.pop(fid)
                points, intensity = normal_xyz_intensity(raw)
                gt_points, _, tags = semantic_xyz_instance_tag(semantic)
                if not len(points) or not len(gt_points):
                    continue

                if len(points) > args.max_points:
                    idx = np.linspace(
                        0,
                        len(points) - 1,
                        args.max_points,
                        dtype=np.int64,
                    )
                    points = points[idx]
                    intensity = intensity[idx]

                result = pipeline.process_points(
                    points,
                    intensity,
                    frame=frames + 1,
                )

                cell_stats = {}
                for point, tag in zip(gt_points, tags):
                    node_id = pipeline.mapping.peek_leaf(
                        float(point[0]),
                        float(point[1]),
                    )
                    if node_id is None:
                        continue
                    stats = cell_stats.setdefault(
                        node_id,
                        {"ground_z": [], "ground": 0, "other": 0},
                    )
                    if int(tag) in GROUND_TAGS:
                        stats["ground_z"].append(float(point[2]))
                        stats["ground"] += 1
                    else:
                        stats["other"] += 1

                for node_id, stats in cell_stats.items():
                    node = pipeline.mapping.nodes.get(node_id)
                    if node is None or not node.active:
                        continue

                    total = stats["ground"] + stats["other"]
                    if total:
                        gt_drivable = stats["ground"] >= stats["other"]
                        predicted_drivable = (
                            node.traversability < args.traversability_threshold
                        )
                        cell_drivable.append(
                            (predicted_drivable, gt_drivable)
                        )

                    if stats["ground_z"]:
                        gt_z = float(np.median(stats["ground_z"]))
                        error = float(node.elevation - gt_z)
                        elevation_errors.append(error)
                        elevation_abs_errors.append(abs(error))

                    if stats["ground"] > 0 and stats["other"] > 0:
                        if node.variance > 0.02:
                            disambiguation_cases += 1

                frames += 1
                print(
                    f"frame={frames}/{args.frames} "
                    f"cells={len(cell_stats)} "
                    f"active={result['stats']['active_cells']}"
                )
    finally:
        raw_sensor.stop()
        semantic_sensor.stop()
        raw_sensor.destroy()
        semantic_sensor.destroy()

    if elevation_errors:
        errors = np.asarray(elevation_errors, dtype=np.float64)
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
    else:
        mae = rmse = 0.0

    if cell_drivable:
        pred = np.asarray([item[0] for item in cell_drivable], dtype=bool)
        truth = np.asarray([item[1] for item in cell_drivable], dtype=bool)
        accuracy = float(np.mean(pred == truth))
        false_drivable = float(np.mean(pred & ~truth))
        false_obstacle = float(np.mean(~pred & truth))
    else:
        accuracy = false_drivable = false_obstacle = 0.0

    report = {
        "dataset": "CARLA semantic LiDAR terrain ground truth",
        "frames": frames,
        "checkpoint": str(Path(selected or args.checkpoint).resolve()),
        "device": getattr(segmenter, "device", args.device),
        "voxel_m": args.voxel,
        "elevation": {
            "rmse_m": rmse,
            "mae_m": mae,
            "reference": "median visible ground/road/sidewalk/terrain z per active map cell",
            "samples": len(elevation_errors),
        },
        "traversability": {
            "accuracy": accuracy,
            "false_drivable_rate": false_drivable,
            "false_obstacle_rate": false_obstacle,
            "threshold": args.traversability_threshold,
            "samples": len(cell_drivable),
            "ground_truth_definition": "majority CARLA semantic LiDAR tag in each active cell",
        },
        "height_information": {
            "mixed_ground_obstacle_cells": disambiguation_cases,
            "note": (
                "These are cells where visible ground and non-ground semantic returns "
                "coexist and the FFEM map retains measurable height variance. A direct "
                "2D-vs-2.5D safety comparison remains a separate experiment."
            ),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["elevation"], indent=2))
    print(json.dumps(report["traversability"], indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

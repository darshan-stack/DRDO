#!/usr/bin/env python3
"""Measure dynamic-object handling against CARLA semantic LiDAR truth.

CARLA semantic LiDAR provides the actor ID and semantic tag for each ray hit.
Actors above the configured ground-truth speed threshold are treated as
moving. Normal LiDAR is fed to FFEM and its scan-to-scan moving mask is scored
against voxel-associated CARLA truth.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ffem.evaluation.metrics import binary_precision_recall
from ffem.perception.factory import build_segmenter
from ffem.pipeline import FFEMConfig, FFEMPipeline


def semantic_records(measurement):
    raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
    if not raw.size:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.uint32),
            np.empty((0,), dtype=np.int32),
        )
    data = raw.reshape(-1, 6)
    points = data[:, :3].copy()
    object_ids = data[:, 4].view(np.uint32)
    tags = data[:, 5].view(np.uint32).astype(np.int32)
    return points, object_ids, tags


def normal_records(measurement):
    raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
    if not raw.size:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    data = raw.reshape(-1, 4)
    return data[:, :3].copy(), data[:, 3].copy()


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
    ap.add_argument("--min-speed", type=float, default=0.5)
    ap.add_argument("--max-points", type=int, default=6000)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--spawn-npc", action="store_true")
    ap.add_argument(
        "--output",
        default="outputs/dynamic_scene_metrics.json",
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

    spawned = []
    if args.spawn_npc:
        blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
        transform = hero.get_transform()
        transform.location.x += 18.0
        npc = world.try_spawn_actor(blueprint, transform)
        if npc is not None:
            npc.set_autopilot(True)
            spawned.append(npc)
            print(f"Spawned dynamic NPC actor={npc.id}")

    blueprint_library = world.get_blueprint_library()
    raw_bp = blueprint_library.find("sensor.lidar.ray_cast")
    semantic_bp = blueprint_library.find("sensor.lidar.ray_cast_semantic")
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

    raw_box = {}
    semantic_box = {}
    pipeline = FFEMPipeline(
        FFEMConfig(
            max_active_cells=20000,
            max_topology_changes=32,
        ),
        segmenter=build_segmenter(
            "torch_range",
            args.checkpoint,
            7,
            device=args.device,
            min_free_vram_mb=512,
        )[0],
    )

    tp = fp = fn = 0.0
    frame_rows = []
    gt_present = []
    joint_hit = []
    first_gt_frame = None
    first_refinement_frame = None
    ghost_points = 0
    predicted_points = 0

    def raw_callback(measurement):
        raw_box[int(measurement.frame)] = measurement

    def semantic_callback(measurement):
        semantic_box[int(measurement.frame)] = measurement

    raw_sensor.listen(raw_callback)
    semantic_sensor.listen(semantic_callback)
    print("Collecting synchronized CARLA raw + semantic LiDAR for dynamic metrics...")

    try:
        deadline = time.time() + max(30.0, args.frames * 0.6 + 15.0)
        processed = 0
        while processed < args.frames and time.time() < deadline:
            world.wait_for_tick(1.0)
            common = sorted(set(raw_box).intersection(semantic_box))
            while common and processed < args.frames:
                fid = common.pop(0)
                raw = raw_box.pop(fid)
                semantic = semantic_box.pop(fid)

                points, intensity = normal_records(raw)
                sp, object_ids, tags = semantic_records(semantic)
                if not len(points) or not len(sp):
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

                actors = {
                    actor.id: float(actor.get_velocity().length())
                    for actor in world.get_actors()
                }
                moving_ids = {
                    int(object_id)
                    for object_id, tag in zip(object_ids, tags)
                    if tag in (4, 9)
                    and actors.get(int(object_id), 0.0) >= args.min_speed
                }
                voxel_keys = np.floor(sp / args.voxel).astype(np.int64)
                voxel_truth = {}
                for key, object_id, tag in zip(voxel_keys, object_ids, tags):
                    voxel_truth.setdefault(tuple(key.tolist()), []).append(
                        (
                            int(object_id),
                            int(tag),
                            int(object_id) in moving_ids,
                        )
                    )

                gt = []
                keep = []
                for point in points:
                    key = tuple(
                        np.floor(point / args.voxel).astype(np.int64).tolist()
                    )
                    values = voxel_truth.get(key)
                    if not values:
                        keep.append(False)
                        gt.append(False)
                        continue
                    keep.append(True)
                    gt.append(any(item[2] for item in values))

                keep = np.asarray(keep, dtype=bool)
                if not np.any(keep):
                    continue

                result = pipeline.process_points(
                    points[keep],
                    intensity[keep],
                    frame=processed + 1,
                )
                predicted = np.asarray(result["moving"], dtype=bool)
                truth = np.asarray(gt, dtype=bool)

                tp += float(np.sum(predicted & truth))
                fp += float(np.sum(predicted & ~truth))
                fn += float(np.sum(~predicted & truth))
                predicted_points += int(predicted.sum())
                ghost_points += int(np.sum(predicted & ~truth))

                truth_any = bool(truth.any())
                hit_any = bool((predicted & truth).any())
                gt_present.append(truth_any)
                joint_hit.append(hit_any)
                if truth_any and first_gt_frame is None:
                    first_gt_frame = processed + 1
                if (
                    truth_any
                    and first_gt_frame is not None
                    and first_refinement_frame is None
                    and (
                        int(result["stats"]["topology_changes"]) > 0
                        or len(pipeline.mapping.events) > 0
                    )
                ):
                    first_refinement_frame = processed + 1

                frame_rows.append(
                    {
                        "frame": processed + 1,
                        "ground_truth_moving_points": int(truth.sum()),
                        "predicted_moving_points": int(predicted.sum()),
                        "true_positive": int(np.sum(predicted & truth)),
                        "false_positive": int(np.sum(predicted & ~truth)),
                        "false_negative": int(np.sum(~predicted & truth)),
                        "active_cells": int(result["stats"]["active_cells"]),
                        "topology_changes": int(result["stats"]["topology_changes"]),
                        "track_count": len(result.get("tracks", [])),
                    }
                )
                processed += 1
                print(
                    f"frame={processed}/{args.frames} "
                    f"gt_moving={int(truth.sum())} "
                    f"pred_moving={int(predicted.sum())}"
                )
    finally:
        raw_sensor.stop()
        semantic_sensor.stop()
        raw_sensor.destroy()
        semantic_sensor.destroy()
        for actor in spawned:
            if actor.is_alive:
                actor.destroy()

    continuity = float(
        np.mean(np.asarray(joint_hit, dtype=np.float32))
    ) if joint_hit else 0.0

    fragment_count = 0
    for prev, current in zip(joint_hit, joint_hit[1:]):
        if prev and not current and any(gt_present):
            fragment_count += 1

    report = {
        "dataset": "CARLA semantic LiDAR dynamic ground truth",
        "frames": len(frame_rows),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": getattr(pipeline.perception, "device", args.device),
        "voxel_m": args.voxel,
        "min_speed_mps": args.min_speed,
        "point_level": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(tp / max(tp + fp, 1.0)),
            "recall": float(tp / max(tp + fn, 1.0)),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1.0)),
        },
        "frame_level": {
            "joint_detection_rate": continuity,
            "fragmentation_transitions": fragment_count,
            "point_level_from_helper": binary_precision_recall(
                np.asarray(
                    [row["predicted_moving_points"] > 0 for row in frame_rows],
                    dtype=bool,
                ),
                np.asarray(
                    [row["ground_truth_moving_points"] > 0 for row in frame_rows],
                    dtype=bool,
                ),
            ),
            "frames_with_gt_motion": int(sum(gt_present)),
            "frames_with_predicted_motion": int(
                sum(row["predicted_moving_points"] > 0 for row in frame_rows)
            ),
        },
        "ghost_trail": {
            "predicted_moving_points": predicted_points,
            "ghost_points": ghost_points,
            "ghost_rate": float(
                ghost_points / max(predicted_points, 1)
            ),
        },
        "refinement_latency_frames": (
            int(first_refinement_frame - first_gt_frame)
            if first_gt_frame is not None and first_refinement_frame is not None
            else None
        ),
        "tracks": {
            "mean_count": float(
                np.mean([row["track_count"] for row in frame_rows])
            ) if frame_rows else 0.0,
            "max_count": int(
                max((row["track_count"] for row in frame_rows), default=0)
            ),
        },
        "carla_dynamic_actor_condition": (
            "Evaluated using actor velocity >= min_speed and semantic tags 4/9."
        ),
        "note": (
            "Ground truth is derived from CARLA semantic LiDAR object IDs and "
            "actor velocities. Refinement latency is a system-level first-refinement "
            "measurement and is not a causality proof by itself."
        ),
        "per_frame": frame_rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["point_level"], indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

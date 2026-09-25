#!/usr/bin/env python3
"""Evaluate the SemanticPOSS-trained model on an unseen CARLA domain.

Normal CARLA LiDAR is the model input. Synchronized CARLA semantic LiDAR is
used as ground truth through voxel association. CARLA semantic LiDAR supplies
instance and semantic information for each ray hit.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ffem.io.semantic_poss import CLASS_NAMES
from ffem.perception.factory import build_segmenter


def carla_to_ffem(tag: int) -> int:
    if tag in (6, 7, 13, 18):
        return 1
    if tag == 8:
        return 2
    if tag == 9:
        return 4
    if tag == 4:
        return 5
    if tag in (1, 2, 5, 10, 11, 14, 15):
        return 3
    return 6


def normal_xyz_intensity(measurement):
    raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
    if not raw.size:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    data = raw.reshape(-1, 4)
    return data[:, :3].copy(), data[:, 3].copy()


def semantic_xyz_instance_tag(measurement):
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


def confusion(pred, truth, classes=7):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (truth, pred), 1)
    return matrix


def metrics(matrix):
    tp = np.diag(matrix).astype(float)
    fp = matrix.sum(0) - tp
    fn = matrix.sum(1) - tp
    support = matrix.sum(1)
    iou_den = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn
    iou = np.divide(tp, iou_den, out=np.zeros(len(tp)), where=iou_den > 0)
    precision = np.divide(
        tp,
        precision_den,
        out=np.zeros(len(tp)),
        where=precision_den > 0,
    )
    recall = np.divide(
        tp,
        recall_den,
        out=np.zeros(len(tp)),
        where=recall_den > 0,
    )
    present = support > 0
    return {
        "mIoU": float(iou[present].mean()) if present.any() else 0.0,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "accuracy": float(tp.sum() / max(matrix.sum(), 1)),
        "per_class_iou": dict(zip(CLASS_NAMES, iou.tolist())),
        "per_class_precision": dict(zip(CLASS_NAMES, precision.tolist())),
        "per_class_recall": dict(zip(CLASS_NAMES, recall.tolist())),
        "support": support.astype(int).tolist(),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="models/checkpoints/semanticposs_range_model.pt",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--voxel", type=float, default=0.12)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--output", default="outputs/carla_generalization.json")
    ap.add_argument("--max-points", type=int, default=5000)
    args = ap.parse_args()

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    vehicles = list(world.get_actors().filter("vehicle.*"))
    if not vehicles:
        raise SystemExit("No vehicle found in CARLA. Start the native stack first.")

    hero = next(
        (vehicle for vehicle in vehicles if vehicle.attributes.get("role_name") == "hero"),
        vehicles[0],
    )
    blueprints = world.get_blueprint_library()
    raw_bp = blueprints.find("sensor.lidar.ray_cast")
    semantic_bp = blueprints.find("sensor.lidar.ray_cast_semantic")

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
    matrix = np.zeros((7, 7), dtype=np.int64)
    matched_points = 0
    raw_points_seen = 0
    frames = 0
    segmenter, selected = build_segmenter(
        "torch_range",
        args.checkpoint,
        7,
        device=args.device,
    )

    def raw_callback(measurement):
        raw_box[int(measurement.frame)] = measurement

    def semantic_callback(measurement):
        semantic_box[int(measurement.frame)] = measurement

    raw_sensor.listen(raw_callback)
    semantic_sensor.listen(semantic_callback)
    print("Collecting synchronized CARLA raw + semantic LiDAR...")

    try:
        import time

        deadline = time.time() + max(20.0, args.frames * 0.5 + 10.0)
        while frames < args.frames and time.time() < deadline:
            world.wait_for_tick(1.0)
            common = sorted(set(raw_box).intersection(semantic_box))
            while common and frames < args.frames:
                frame_id = common.pop(0)
                raw = raw_box.pop(frame_id)
                semantic = semantic_box.pop(frame_id)

                raw_points, intensities = normal_xyz_intensity(raw)
                semantic_points, _, semantic_tags = semantic_xyz_instance_tag(semantic)
                raw_points_seen += len(raw_points)
                if not len(raw_points) or not len(semantic_points):
                    continue

                if len(raw_points) > args.max_points:
                    indices = np.linspace(
                        0,
                        len(raw_points) - 1,
                        args.max_points,
                        dtype=np.int64,
                    )
                    raw_points = raw_points[indices]
                    intensities = intensities[indices]

                voxel_keys = np.floor(semantic_points / args.voxel).astype(np.int64)
                voxel_tags = {}
                for key, tag in zip(voxel_keys, semantic_tags):
                    voxel_tags.setdefault(tuple(key.tolist()), []).append(int(tag))

                keep = []
                truth = []
                for point in raw_points:
                    key = tuple(
                        np.floor(point / args.voxel).astype(np.int64).tolist()
                    )
                    values = voxel_tags.get(key)
                    keep.append(bool(values))
                    if values:
                        truth.append(
                            carla_to_ffem(
                                Counter(values).most_common(1)[0][0]
                            )
                        )

                keep = np.asarray(keep, dtype=bool)
                if not np.any(keep):
                    continue

                predicted, _ = segmenter.predict(
                    raw_points[keep],
                    intensities[keep],
                )
                truth_array = np.asarray(truth, dtype=np.int32)
                matrix += confusion(predicted, truth_array)
                matched_points += len(truth_array)
                frames += 1
                print(
                    f"frame={frames}/{args.frames} "
                    f"matched={len(truth_array)}"
                )
    finally:
        raw_sensor.stop()
        semantic_sensor.stop()
        raw_sensor.destroy()
        semantic_sensor.destroy()

    result = metrics(matrix)
    result.update(
        {
            "dataset": "CARLA unseen-domain ground truth",
            "frames": frames,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "device": getattr(segmenter, "device", args.device),
            "matched_points": matched_points,
            "raw_points_seen": raw_points_seen,
            "match_coverage": float(
                matched_points / max(raw_points_seen, 1)
            ),
            "voxel_m": args.voxel,
            "note": (
                "Normal CARLA LiDAR is the model input. CARLA semantic LiDAR "
                "provides instance/semantic ground truth; association is voxel-based."
            ),
        }
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "mIoU",
                    "macro_precision",
                    "macro_recall",
                    "accuracy",
                    "match_coverage",
                )
            },
            indent=2,
        )
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()

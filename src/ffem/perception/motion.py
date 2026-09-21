"""Scan-to-scan motion residuals and lightweight cluster tracking."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    track_id: int
    center: np.ndarray
    velocity: np.ndarray
    size: np.ndarray
    age: int = 1
    missed: int = 0


class VoxelMotionDetector:
    """Fast voxel residual detector with optional ego-motion compensation."""

    def __init__(self, voxel_size: float = 0.35, threshold: float = 0.45):
        self.voxel_size = float(voxel_size)
        self.threshold = float(threshold)
        self.previous_points = np.empty((0, 3), dtype=np.float32)

    def _structured_keys(self, points: np.ndarray):
        q = np.floor(np.asarray(points, dtype=np.float32) / self.voxel_size).astype(np.int64)
        out = np.empty(len(q), dtype=[("x", "i8"), ("y", "i8"), ("z", "i8")])
        out["x"] = q[:, 0]
        out["y"] = q[:, 1]
        out["z"] = q[:, 2]
        return out

    @staticmethod
    def _apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("ego_transform must have shape (4, 4)")
        p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if not len(p):
            return p.copy()
        hom = np.concatenate([p, np.ones((len(p), 1), dtype=np.float32)], axis=1)
        return (hom @ matrix.T)[:, :3].astype(np.float32)

    def detect(
        self,
        points: np.ndarray,
        ego_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return per-point motion residuals after optional platform compensation.

        ego_transform maps points from the previous sensor frame into the current
        sensor frame. Static-scene returns are compared after this transform.
        """
        current = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        out = np.zeros(len(current), dtype=np.float32)

        if len(self.previous_points):
            previous = self.previous_points
            if ego_transform is not None:
                previous = self._apply_transform(previous, ego_transform)

            previous_keys = self._structured_keys(previous)
            current_keys = self._structured_keys(current)
            order = np.argsort(previous_keys, order=("x", "y", "z"))
            sorted_keys = previous_keys[order]
            locations = np.searchsorted(sorted_keys, current_keys)
            valid = locations < len(sorted_keys)
            if valid.any():
                current_idx = np.flatnonzero(valid)
                matched = sorted_keys[locations[current_idx]] == current_keys[current_idx]
                current_idx = current_idx[matched]
                if len(current_idx):
                    previous_points = previous[order[locations[current_idx]]]
                    residual = np.linalg.norm(
                        current[current_idx] - previous_points, axis=1
                    )
                    out[current_idx] = (residual > self.threshold).astype(np.float32)

        self.previous_points = current.copy()
        return out

    def _key(self, x):
        return tuple(np.floor(np.asarray(x) / self.voxel_size).astype(int))

    @staticmethod
    def _transform(previous, points, matrix):
        return VoxelMotionDetector._apply_transform(previous, matrix)


class CentroidTracker:
    """Lightweight tracker used for live diagnostics."""

    def __init__(self, max_distance: float = 3.0, max_missed: int = 5):
        self.max_distance = float(max_distance)
        self.max_missed = int(max_missed)
        self.tracks = {}
        self.next_id = 1

    def update(self, points: np.ndarray, motion: np.ndarray) -> list[Track]:
        p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        active = p[np.asarray(motion).reshape(-1) > 0.5]
        clusters = self._clusters(active)
        used = set()

        for center, size in clusters:
            best = None
            distance = self.max_distance
            for tid, track in self.tracks.items():
                d = float(np.linalg.norm(center - track.center))
                if d < distance and tid not in used:
                    best, distance = tid, d
            if best is None:
                self.tracks[self.next_id] = Track(
                    self.next_id,
                    center,
                    np.zeros(3, dtype=np.float32),
                    size,
                )
                used.add(self.next_id)
                self.next_id += 1
            else:
                track = self.tracks[best]
                track.velocity = center - track.center
                track.center = center
                track.size = size
                track.age += 1
                track.missed = 0
                used.add(best)

        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid].missed += 1
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]
        return list(self.tracks.values())

    @staticmethod
    def _clusters(points):
        if len(points) == 0:
            return []
        bins = {}
        for point in points:
            key = (
                int(np.floor(point[0] / 2.0)),
                int(np.floor(point[1] / 2.0)),
            )
            bins.setdefault(key, []).append(point)
        return [
            (np.mean(values, axis=0), np.ptp(values, axis=0) + 0.1)
            for values in bins.values()
            if len(values) >= 3
        ]

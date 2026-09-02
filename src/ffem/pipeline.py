"""Runnable 70%-ready FFEM research prototype core.

The perception backends are deterministic stand-ins until real model checkpoints
are supplied. The map, adaptation, fusion, metrics, and visualization contracts
are real and testable.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time
import numpy as np
from ffem.perception.motion import VoxelMotionDetector, CentroidTracker
from ffem.mapping.radial_resolution import RadialResolutionPolicy

@dataclass
class FFEMConfig:
    base_cell_size: float = 1.0
    finest_cell_size: float = 0.25
    max_level: int = 2
    refine_threshold: float = 0.60
    merge_threshold: float = 0.25
    dwell_frames: int = 5
    max_active_cells: int = 20000
    max_topology_changes: int = 32
    predictive_dilation_frames: int = 2
    semantic_weight: float = 0.30
    motion_weight: float = 0.30
    traversability_weight: float = 0.20
    geometry_weight: float = 0.15
    range_weight: float = 0.05
    num_classes: int = 7

@dataclass
class Cell:
    key: tuple[int, int, int]
    level: int = 0
    count: int = 0
    elevation: float = 0.0
    variance: float = 0.0
    semantic_probs: np.ndarray = field(default_factory=lambda: np.ones(7) / 7)
    motion_probability: float = 0.0
    traversability: float = 0.0
    attention: float = 0.0
    quiet_frames: int = 0
    size_m: float = 1.0
    slices: list[dict[str, float]] = field(default_factory=list)

    @property
    def size(self) -> float:
        return self.size_m / (2 ** self.level)

class SyntheticLidar:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)

    def frame(self, index: int, n: int = 2200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = self.rng.uniform(-np.pi, np.pi, n)
        radius = self.rng.uniform(2.0, 45.0, n)
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        z = 0.10 * np.sin(x / 4.0) + 0.07 * np.cos(y / 3.0)
        # A moving obstacle crossing the map; this is only a deterministic demo backend.
        cx, cy = 8.0 + 0.18 * index, 2.0 + 0.05 * np.sin(index / 5)
        moving = ((x - cx) ** 2 + (y - cy) ** 2) < 3.5
        z[moving] += 1.0
        intensity = np.clip(0.4 + 0.3 * np.sin(x) + 0.2 * self.rng.normal(size=n), 0, 1)
        return np.column_stack((x, y, z)), intensity, moving

class MockPerception:
    """Deterministic placeholder for a future PTv3/RangeFormer/MOS backend."""
    def __init__(self, num_classes: int = 4): self.num_classes = num_classes

    def infer(self, points: np.ndarray, moving: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        labels = np.zeros(len(points), dtype=np.int64)
        labels[(points[:, 2] > 0.25) & ~moving] = 1
        labels[(intensity > 0.72) & ~moving] = 2
        labels[moving] = 3
        probs = np.full((len(points), self.num_classes), 0.04 / max(1, self.num_classes - 1), dtype=np.float32)
        probs[np.arange(len(points)), labels] = 0.96
        return probs, moving.astype(np.float32)

class AdaptiveElevationMap:
    def __init__(self, config: FFEMConfig):
        self.cfg = config
        self.cells: dict[tuple[int, int, int], Cell] = {}
        self.radial = RadialResolutionPolicy()
        self.events: list[dict[str, Any]] = []

    def _key(self, x: float, y: float, level: int = 0) -> tuple[int, int, int]:
        radius = float(np.hypot(x, y))
        band = next((i for i, b in enumerate(self.radial.bands) if radius <= b.max_radius_m), len(self.radial.bands) - 1)
        size = self.radial.bands[band].cell_size_m / (2 ** level)
        return band, int(np.floor(x / size)), int(np.floor(y / size))
    def _cell(self, key: tuple[int, int, int], level: int = 0) -> Cell:
        if key not in self.cells: self.cells[key] = Cell(key=key, level=level, size_m=self.radial.bands[key[0]].cell_size_m, semantic_probs=np.ones(self.cfg.num_classes) / self.cfg.num_classes)

        return self.cells[key]

    def update(self, points: np.ndarray, semantic_probs: np.ndarray, motion: np.ndarray, frame: int) -> dict[str, float]:
        t0 = time.perf_counter(); observations: dict[tuple[int, int], list[int]] = {}
        for i, (x, y, _) in enumerate(points): observations.setdefault(self._key(x, y), []).append(i)
        for key, inds in observations.items():
            c = self._cell(key); z = points[inds, 2]; old = c.elevation
            c.count += len(inds); c.elevation = float(np.mean(z))
            c.variance = float(np.var(z)) if len(inds) > 1 else c.variance
            c.semantic_probs = 0.85 * c.semantic_probs + 0.15 * np.mean(semantic_probs[inds], axis=0)
            c.motion_probability = float(0.80 * c.motion_probability + 0.20 * np.mean(motion[inds]))
            c.traversability = float(np.clip(3.0 * np.sqrt(c.variance + 1e-6) + abs(c.elevation - old), 0, 1))
            entropy = float(-np.sum(c.semantic_probs * np.log(c.semantic_probs + 1e-8)) / np.log(self.cfg.num_classes))
            c.attention = float(np.clip(self.cfg.semantic_weight * entropy + self.cfg.motion_weight * c.motion_probability + self.cfg.traversability_weight * c.traversability + self.cfg.geometry_weight * min(1, 4*c.variance) + self.cfg.range_weight * min(1, np.hypot(key[1] * c.size, key[2] * c.size) / 50), 0, 1))
            if len(inds) >= 8 and c.variance > 0.04 and not c.slices:
                c.slices = [{'height': float(np.min(z)), 'support': float(np.sum(z < np.mean(z)) / len(z))}, {'height': float(np.max(z)), 'support': float(np.sum(z >= np.mean(z)) / len(z))}]
        changes = 0
        ranked = sorted(self.cells.values(), key=lambda c: c.attention, reverse=True)
        for c in ranked:
            if changes >= self.cfg.max_topology_changes: break
            if c.attention >= self.cfg.refine_threshold and c.level < self.cfg.max_level:
                c.level += 1; c.quiet_frames = 0; changes += 1
                self.events.append({'frame': frame, 'cell': c.key, 'old_level': c.level-1, 'new_level': c.level, 'reason': 'attention', 'score': c.attention})
            elif c.attention < self.cfg.merge_threshold and c.level > 0:
                c.quiet_frames += 1
                if c.quiet_frames >= self.cfg.dwell_frames:
                    c.level -= 1; c.quiet_frames = 0; changes += 1
                    self.events.append({'frame': frame, 'cell': c.key, 'old_level': c.level+1, 'new_level': c.level, 'reason': 'hysteresis', 'score': c.attention})
        if len(self.cells) > self.cfg.max_active_cells:
            for key in list(self.cells)[:len(self.cells)-self.cfg.max_active_cells]: del self.cells[key]
        return {'map_ms': (time.perf_counter()-t0)*1000, 'active_cells': len(self.cells), 'topology_changes': changes}

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.cells: return np.empty((0,3)), np.empty((0,3), dtype=np.uint8), np.empty((0,))
        pts=[]; colors=[]; levels=[]
        palette=np.array([[90,90,90],[70,140,220],[70,210,100],[180,120,60],[230,70,60],[220,80,180],[245,190,40]], dtype=np.uint8)
        for c in self.cells.values():
            size=c.size; x=(c.key[1]+.5)*size; y=(c.key[2]+.5)*size
            pts.append([x,y,c.elevation]); colors.append(palette[int(np.argmax(c.semantic_probs))]); levels.append(c.level)
        return np.asarray(pts), np.asarray(colors), np.asarray(levels)

class FFEMPipeline:
    def __init__(self, config: FFEMConfig | None = None, seed: int = 7, segmenter=None, motion_detector=None, tracker=None):
        self.config = config or FFEMConfig(); self.sensor=SyntheticLidar(seed); self.perception=MockPerception(self.config.num_classes); self.segmenter=segmenter; self.motion_detector=motion_detector or VoxelMotionDetector(); self.tracker=tracker or CentroidTracker(); self.mapping=AdaptiveElevationMap(self.config)
        self.history: list[dict[str, float]]=[]

    def process_points(self, points: np.ndarray, intensity: np.ndarray | None = None, motion: np.ndarray | None = None, frame: int = 0) -> dict[str, Any]:
        t0 = time.perf_counter()
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        intensity = np.zeros(len(points), dtype=np.float32) if intensity is None else np.asarray(intensity, dtype=np.float32)
        motion = self.motion_detector.detect(points) if motion is None else np.asarray(motion, dtype=np.float32)
        if self.segmenter is not None:
            _, probs = self.segmenter.predict(points, intensity)
            inferred_motion = motion
        else:
            probs, inferred_motion = self.perception.infer(points, motion > 0.5, intensity)
        motion = np.maximum(motion, inferred_motion)
        stats = self.mapping.update(points, probs, motion, frame)
        tracks = self.tracker.update(points, motion)
        stats.update({'frame': frame, 'total_ms': (time.perf_counter()-t0)*1000, 'points': len(points), 'moving_points': int((motion > 0.5).sum()), 'tracks': len(tracks)})
        self.history.append(stats)
        return {'points': points, 'intensity': intensity, 'moving': motion > 0.5, 'motion_probability': motion, 'semantic_probs': probs, 'stats': stats}

    def step(self, frame: int) -> dict[str, Any]:
        points, intensity, moving = self.sensor.frame(frame)
        return self.process_points(points, intensity, moving.astype(np.float32), frame)

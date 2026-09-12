"""Adaptive FFEM mapping and perception pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
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
    semantic_weight: float = 0.27
    motion_weight: float = 0.27
    traversability_weight: float = 0.18
    geometry_weight: float = 0.13
    range_weight: float = 0.05
    planning_weight: float = 0.10
    num_classes: int = 7


@dataclass
class HierarchyNode:
    """Explicit parent/4-child node in the adaptive spatial hierarchy."""
    node_id: tuple[int, int, int, int]
    parent: tuple[int, int, int, int] | None
    children: list[tuple[int, int, int, int]] = field(default_factory=list)
    level: int = 0
    count: int = 0
    elevation: float = 0.0
    variance: float = 0.0
    semantic_probs: np.ndarray = field(default_factory=lambda: np.ones(7) / 7)
    motion_probability: float = 0.0
    traversability: float = 0.0
    attention: float = 0.0
    planning_criticality: float = 0.0
    quiet_frames: int = 0
    active: bool = True
    size_m: float = 1.0

    @property
    def size(self) -> float:
        return self.size_m / (2 ** self.level)

    @property
    def center(self) -> tuple[float, float]:
        _, _, ix, iy = self.node_id
        s = self.size
        return (float((ix + 0.5) * s), float((iy + 0.5) * s))


class SyntheticLidar:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)

    def frame(self, index, n=2200):
        theta = self.rng.uniform(-np.pi, np.pi, n)
        radius = self.rng.uniform(2, 45, n)
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        z = 0.10 * np.sin(x / 4) + 0.07 * np.cos(y / 3)
        cx, cy = 8 + 0.18 * index, 2 + 0.05 * np.sin(index / 5)
        moving = ((x - cx) ** 2 + (y - cy) ** 2) < 3.5
        z[moving] += 1
        intensity = np.clip(0.4 + 0.3 * np.sin(x) + 0.2 * self.rng.normal(size=n), 0, 1)
        return np.column_stack((x, y, z)), intensity, moving


class MockPerception:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes

    def infer(self, points, moving, intensity):
        labels = np.zeros(len(points), dtype=np.int64)
        labels[(points[:, 2] > 0.25) & ~moving] = 1
        labels[(intensity > 0.72) & ~moving] = 2
        labels[moving] = 3
        probs = np.full(
            (len(points), self.num_classes),
            0.04 / max(1, self.num_classes - 1),
            dtype=np.float32,
        )
        probs[np.arange(len(points)), labels] = 0.96
        return probs, moving.astype(np.float32)


class AdaptiveElevationMap:
    """Hierarchical adaptive 2.5D map with explicit persistent parent/4-child topology."""

    def __init__(self, config):
        self.cfg = config
        self.nodes: dict[tuple[int, int, int, int], HierarchyNode] = {}
        self.leaves: set[tuple[int, int, int, int]] = set()
        self.cells: dict[tuple[int, int, int, int], HierarchyNode] = {}
        self.radial = RadialResolutionPolicy()
        self.events = []
        self._limits = np.asarray([b.max_radius_m for b in self.radial.bands], dtype=np.float32)
        self._sizes = np.asarray([b.cell_size_m for b in self.radial.bands], dtype=np.float32)

    def _band(self, x, y):
        r = float(np.hypot(x, y))
        return int(np.clip(np.searchsorted(self._limits, r, side="left"), 0, len(self._sizes) - 1))

    def _root_id(self, x, y):
        band = self._band(x, y)
        size = float(self._sizes[band])
        return band, 0, int(np.floor(x / size)), int(np.floor(y / size))

    def _make_node(self, node_id, parent=None, stats=None):
        band, level, _, _ = node_id
        stats = stats or {}
        node = HierarchyNode(
            node_id=node_id,
            parent=parent,
            level=level,
            size_m=float(self._sizes[band]),
            count=int(stats.get("count", 0)),
            elevation=float(stats.get("elevation", 0.0)),
            variance=float(stats.get("variance", 0.0)),
            semantic_probs=np.array(
                stats.get("semantic_probs", np.ones(self.cfg.num_classes) / self.cfg.num_classes),
                dtype=np.float64,
                copy=True,
            ),
            motion_probability=float(stats.get("motion_probability", 0.0)),
            traversability=float(stats.get("traversability", 0.0)),
            attention=float(stats.get("attention", 0.0)),
            planning_criticality=float(stats.get("planning_criticality", 0.0)),
        )
        self.nodes[node_id] = node
        return node

    def _ensure_root(self, x, y):
        root_id = self._root_id(x, y)
        if root_id not in self.nodes:
            self._make_node(root_id)
            self.leaves.add(root_id)
        return root_id

    def locate_leaf(self, x, y):
        """Return the current leaf containing x,y."""
        node_id = self._ensure_root(x, y)
        while True:
            node = self.nodes[node_id]
            if not node.children:
                return node_id
            band, level, ix, iy = node_id
            size = float(self._sizes[band]) / (2 ** level)
            center_x = (ix + 0.5) * size
            center_y = (iy + 0.5) * size
            dx = 1 if x >= center_x else 0
            dy = 1 if y >= center_y else 0
            child_id = (band, level + 1, ix * 2 + dx, iy * 2 + dy)
            if child_id not in self.nodes:
                return node_id
            node_id = child_id

    def _split(self, node):
        if node.level >= self.cfg.max_level or node.children:
            return False
        child_stats = {
            "count": max(0, node.count // 4),
            "elevation": node.elevation,
            "variance": node.variance,
            "semantic_probs": node.semantic_probs,
            "motion_probability": node.motion_probability,
            "traversability": node.traversability,
            "attention": node.attention,
            "planning_criticality": node.planning_criticality,
        }
        band, level, ix, iy = node.node_id
        for dx in (0, 1):
            for dy in (0, 1):
                child_id = (band, level + 1, ix * 2 + dx, iy * 2 + dy)
                self._make_node(child_id, parent=node.node_id, stats=child_stats)
                node.children.append(child_id)
                self.leaves.add(child_id)
        node.active = False
        self.leaves.discard(node.node_id)
        return True

    def _merge(self, parent):
        if parent.level == 0 or len(parent.children) != 4:
            return False
        children = [self.nodes[cid] for cid in parent.children if cid in self.nodes]
        if len(children) != 4 or not all(c.active and not c.children for c in children):
            return False
        weights = np.asarray([max(c.count, 1) for c in children], dtype=np.float64)
        parent.count = sum(c.count for c in children)
        parent.elevation = float(np.average([c.elevation for c in children], weights=weights))
        parent.variance = float(np.average([c.variance for c in children], weights=weights))
        probs = np.average(np.stack([c.semantic_probs for c in children]), axis=0, weights=weights)
        parent.semantic_probs = probs / max(float(probs.sum()), 1e-12)
        parent.motion_probability = float(np.average([c.motion_probability for c in children], weights=weights))
        parent.traversability = float(np.average([c.traversability for c in children], weights=weights))
        parent.attention = float(np.average([c.attention for c in children], weights=weights))
        parent.planning_criticality = float(np.average([c.planning_criticality for c in children], weights=weights))
        for child in children:
            child.active = False
            self.leaves.discard(child.node_id)
        parent.children = []
        parent.active = True
        self.leaves.add(parent.node_id)
        return True

    @staticmethod
    def _aggregate(inv, values, groups):
        return np.bincount(inv, weights=np.asarray(values, dtype=np.float64), minlength=groups)

    def update(self, points, semantic_probs, motion, frame):
        t0 = time.perf_counter()
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        n = len(points)
        if n == 0:
            return {"map_ms": (time.perf_counter() - t0) * 1000, "active_cells": len(self.leaves), "topology_changes": 0}

        leaf_ids = [self.locate_leaf(float(x), float(y)) for x, y in points[:, :2]]
        key_dtype = [("b", "i2"), ("l", "i1"), ("x", "i8"), ("y", "i8")]
        keys = np.empty(n, dtype=key_dtype)
        keys["b"] = [node_id[0] for node_id in leaf_ids]
        keys["l"] = [node_id[1] for node_id in leaf_ids]
        keys["x"] = [node_id[2] for node_id in leaf_ids]
        keys["y"] = [node_id[3] for node_id in leaf_ids]
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        groups = len(unique_keys)
        counts = np.bincount(inverse, minlength=groups).astype(np.float64)
        z = points[:, 2].astype(np.float64)
        sum_z = self._aggregate(inverse, z, groups)
        mean_z = sum_z / np.maximum(counts, 1)
        variance = self._aggregate(inverse, (z - mean_z[inverse]) ** 2, groups) / np.maximum(counts, 1)
        motion_mean = self._aggregate(inverse, np.asarray(motion).reshape(-1), groups) / np.maximum(counts, 1)
        probs = np.asarray(semantic_probs, dtype=np.float32)
        if probs.ndim != 2 or probs.shape[0] != n:
            raise ValueError("semantic_probs must have shape [N, num_classes]")
        class_means = np.empty((groups, self.cfg.num_classes), dtype=np.float64)
        for class_id in range(self.cfg.num_classes):
            class_means[:, class_id] = self._aggregate(inverse, probs[:, class_id], groups) / np.maximum(counts, 1)

        for gi in range(groups):
            node_id = (int(unique_keys["b"][gi]), int(unique_keys["l"][gi]), int(unique_keys["x"][gi]), int(unique_keys["y"][gi]))
            node = self.nodes[node_id]
            old_elevation = node.elevation
            node.count += int(counts[gi])
            node.elevation = float(mean_z[gi])
            node.variance = float(variance[gi])
            node.semantic_probs = 0.85 * node.semantic_probs + 0.15 * class_means[gi]
            node.semantic_probs /= max(float(node.semantic_probs.sum()), 1e-12)
            node.motion_probability = float(0.8 * node.motion_probability + 0.2 * motion_mean[gi])
            node.traversability = float(np.clip(3 * np.sqrt(node.variance + 1e-6) + abs(node.elevation - old_elevation), 0, 1))
            entropy = float(-np.sum(node.semantic_probs * np.log(node.semantic_probs + 1e-8)) / np.log(self.cfg.num_classes))
            x, y = node.center
            range_term = min(1.0, np.hypot(x, y) / 50.0)
            node.planning_criticality *= 0.90
            node.attention = float(np.clip(
                self.cfg.semantic_weight * entropy
                + self.cfg.motion_weight * node.motion_probability
                + self.cfg.traversability_weight * node.traversability
                + self.cfg.geometry_weight * min(1.0, 4 * node.variance)
                + self.cfg.range_weight * range_term
                + self.cfg.planning_weight * node.planning_criticality,
                0,
                1,
            ))

        changes = 0
        refine = [self.nodes[nid] for nid in self.leaves if self.nodes[nid].level < self.cfg.max_level and self.nodes[nid].attention >= self.cfg.refine_threshold]
        refine.sort(key=lambda node: node.attention, reverse=True)
        for node in refine[:self.cfg.max_topology_changes]:
            old_level = node.level
            if self._split(node):
                self.events.append({
                    "frame": frame,
                    "cell": node.node_id,
                    "old_level": old_level,
                    "new_level": old_level + 1,
                    "reason": "attention",
                    "score": node.attention,
                    "hierarchy": "split_4",
                    "parent": node.node_id,
                })
                changes += 1

        if changes < self.cfg.max_topology_changes:
            visited_parents = set()
            for leaf_id in list(self.leaves):
                node = self.nodes[leaf_id]
                if node.parent is None or node.parent in visited_parents:
                    continue
                parent = self.nodes[node.parent]
                siblings = [self.nodes[cid] for cid in parent.children if cid in self.nodes]
                if len(siblings) != 4 or not all(child.active and not child.children for child in siblings):
                    continue
                if any(child.attention >= self.cfg.merge_threshold for child in siblings):
                    parent.quiet_frames = 0
                    continue
                parent.quiet_frames += 1
                visited_parents.add(parent.node_id)
                if parent.quiet_frames >= self.cfg.dwell_frames and changes < self.cfg.max_topology_changes:
                    if self._merge(parent):
                        self.events.append({
                            "frame": frame,
                            "cell": parent.node_id,
                            "old_level": parent.level + 1,
                            "new_level": parent.level,
                            "reason": "hysteresis",
                            "score": parent.attention,
                            "hierarchy": "merge_4",
                            "parent": parent.node_id,
                        })
                        changes += 1

        if len(self.leaves) > self.cfg.max_active_cells:
            excess = len(self.leaves) - self.cfg.max_active_cells
            drop = sorted(self.leaves, key=lambda nid: (self.nodes[nid].attention, self.nodes[nid].level))[:excess]
            for node_id in drop:
                self.leaves.discard(node_id)
                self.nodes[node_id].active = False

        self.cells = {node_id: self.nodes[node_id] for node_id in self.leaves}
        return {
            "map_ms": (time.perf_counter() - t0) * 1000,
            "active_cells": len(self.leaves),
            "topology_changes": changes,
            "hierarchy_nodes": len(self.nodes),
        }

    def apply_planning_feedback(self, plan_points, risk_profile, frame):
        """Feed planner-critical regions back into map attention/refinement."""
        points = np.asarray(plan_points, dtype=np.float32).reshape(-1, 2)
        risks = np.asarray(risk_profile, dtype=np.float32).reshape(-1) if len(points) else np.empty((0,), dtype=np.float32)
        changes = 0
        for (x, y), risk in zip(points, risks):
            node_id = self.locate_leaf(float(x), float(y))
            node = self.nodes[node_id]
            node.planning_criticality = float(np.clip(max(node.planning_criticality, float(risk)), 0, 1))
            node.attention = float(np.clip(node.attention + self.cfg.planning_weight * node.planning_criticality, 0, 1))
            if node.level < self.cfg.max_level and node.attention >= self.cfg.refine_threshold and changes < self.cfg.max_topology_changes:
                old_level = node.level
                if self._split(node):
                    self.events.append({
                        "frame": frame,
                        "cell": node.node_id,
                        "old_level": old_level,
                        "new_level": old_level + 1,
                        "reason": "planning_feedback",
                        "score": node.attention,
                        "hierarchy": "split_4",
                        "parent": node.node_id,
                    })
                    changes += 1
        self.cells = {node_id: self.nodes[node_id] for node_id in self.leaves}
        return changes

    def arrays(self):
        active_nodes = [self.nodes[node_id] for node_id in self.leaves if self.nodes[node_id].active]
        if not active_nodes:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), np.empty((0,))
        points = np.empty((len(active_nodes), 3), dtype=np.float32)
        colors = np.empty((len(active_nodes), 3), dtype=np.uint8)
        levels = np.empty(len(active_nodes), dtype=np.int8)
        palette = np.array([[90, 90, 90], [70, 140, 220], [70, 210, 100], [180, 120, 60], [230, 70, 60], [220, 80, 180], [245, 190, 40]], dtype=np.uint8)
        for i, node in enumerate(active_nodes):
            points[i] = [node.center[0], node.center[1], node.elevation]
            colors[i] = palette[int(np.argmax(node.semantic_probs))]
            levels[i] = node.level
        return points, colors, levels

    def diagnostics(self):
        active_nodes = [self.nodes[node_id] for node_id in self.leaves if self.nodes[node_id].active]
        if not active_nodes:
            empty3 = np.empty((0, 3), dtype=np.float32)
            empty = np.empty((0,), dtype=np.float32)
            empty_i = np.empty((0,), dtype=np.int32)
            return empty3, empty, empty, empty, empty_i
        points = np.empty((len(active_nodes), 3), dtype=np.float32)
        traversability = np.empty(len(active_nodes), dtype=np.float32)
        uncertainty = np.empty(len(active_nodes), dtype=np.float32)
        attention = np.empty(len(active_nodes), dtype=np.float32)
        levels = np.empty(len(active_nodes), dtype=np.int32)
        for i, node in enumerate(active_nodes):
            points[i] = [node.center[0], node.center[1], node.elevation]
            probs = node.semantic_probs / max(float(node.semantic_probs.sum()), 1e-12)
            uncertainty[i] = float(np.clip(-np.sum(probs * np.log(probs + 1e-8)) / np.log(self.cfg.num_classes), 0, 1))
            traversability[i] = node.traversability
            attention[i] = node.attention
            levels[i] = node.level
        return points, traversability, uncertainty, attention, levels

    def hierarchy_arrays(self):
        parent_points = []
        child_segments = []
        for node in self.nodes.values():
            if not node.children:
                continue
            px, py = node.center
            parent_points.append([px, py, node.level])
            for child_id in node.children:
                child = self.nodes[child_id]
                cx, cy = child.center
                child_segments.append([px, py, cx, cy, child.level])
        return np.asarray(parent_points, dtype=np.float32).reshape(-1, 3), np.asarray(child_segments, dtype=np.float32).reshape(-1, 5)


class FFEMPipeline:
    def __init__(self, config=None, seed=7, segmenter=None, motion_detector=None, tracker=None):
        self.config = config or FFEMConfig()
        self.sensor = SyntheticLidar(seed)
        self.perception = MockPerception(self.config.num_classes)
        self.segmenter = segmenter
        self.motion_detector = motion_detector or VoxelMotionDetector()
        self.tracker = tracker or CentroidTracker()
        self.mapping = AdaptiveElevationMap(self.config)
        self.history = []

    def process_points(self, points, intensity=None, motion=None, frame=0):
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
        stats.update({
            "frame": frame,
            "total_ms": (time.perf_counter() - t0) * 1000,
            "points": len(points),
            "moving_points": int((motion > 0.5).sum()),
            "tracks": len(tracks),
        })
        self.history.append(stats)
        return {
            "points": points,
            "intensity": intensity,
            "moving": motion > 0.5,
            "motion_probability": motion,
            "semantic_probs": probs,
            "tracks": tracks,
            "stats": stats,
        }

    def step(self, frame):
        points, intensity, moving = self.sensor.frame(frame)
        return self.process_points(points, intensity, moving.astype(np.float32), frame)

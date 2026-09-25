"""Core FFEM perception, adaptive 2.5D mapping, and feedback pipeline.

The module is intentionally ROS-independent so the same implementation can be
used by replay tests, CARLA, and the ROS 2 adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from ffem.mapping.radial_resolution import RadialResolutionPolicy
from ffem.perception.motion import CentroidTracker, VoxelMotionDetector


@dataclass
class FFEMConfig:
    base_cell_size: float = 1.0
    finest_cell_size: float = 0.05
    max_level: int = 2
    refine_threshold: float = 0.60
    merge_threshold: float = 0.25
    dwell_frames: int = 5
    max_active_cells: int = 20_000
    max_topology_changes: int = 32
    predictive_dilation_frames: int = 2
    predictive_dilation_radius_m: float = 1.0
    semantic_weight: float = 0.27
    motion_weight: float = 0.27
    traversability_weight: float = 0.18
    geometry_weight: float = 0.13
    range_weight: float = 0.05
    planning_weight: float = 0.10
    num_classes: int = 7
    max_event_history: int = 512
    max_history: int = 256


@dataclass
class HierarchyNode:
    """Persistent parent/4-child spatial hierarchy node."""

    node_id: tuple[int, int, int, int]
    parent: tuple[int, int, int, int] | None
    children: list[tuple[int, int, int, int]] = field(default_factory=list)
    level: int = 0
    count: int = 0
    elevation: float = 0.0
    variance: float = 0.0
    semantic_probs: np.ndarray = field(default_factory=lambda: np.ones(7) / 7.0)
    motion_probability: float = 0.0
    traversability: float = 0.0
    attention: float = 0.0
    planning_criticality: float = 0.0
    quiet_frames: int = 0
    active: bool = True
    size_m: float = 1.0

    @property
    def size(self) -> float:
        return self.size_m / (2.0**self.level)

    @property
    def center(self) -> tuple[float, float]:
        _, _, ix, iy = self.node_id
        s = self.size
        return float((ix + 0.5) * s), float((iy + 0.5) * s)


class SyntheticLidar:
    """Deterministic synthetic source for unit tests and smoke tests only."""

    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)

    def frame(self, index: int, n: int = 2200):
        theta = self.rng.uniform(-np.pi, np.pi, n)
        radius = self.rng.uniform(2.0, 45.0, n)
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        z = 0.10 * np.sin(x / 4.0) + 0.07 * np.cos(y / 3.0)
        cx, cy = 8.0 + 0.18 * index, 2.0 + 0.05 * np.sin(index / 5.0)
        moving = ((x - cx) ** 2 + (y - cy) ** 2) < 3.5
        z[moving] += 1.0
        intensity = np.clip(
            0.4 + 0.3 * np.sin(x) + 0.2 * self.rng.normal(size=n),
            0.0,
            1.0,
        )
        return np.column_stack((x, y, z)).astype(np.float32), intensity.astype(
            np.float32
        ), moving.astype(np.float32)


class MockPerception:
    """Deterministic non-neural backend for software integration tests."""

    def __init__(self, num_classes: int = 7) -> None:
        self.num_classes = num_classes

    def predict(self, points: np.ndarray, intensity: np.ndarray | None = None):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        moving = np.zeros(len(points), dtype=np.float32)
        if len(points):
            moving = (
                points[:, 2] > np.percentile(points[:, 2], 92)
            ).astype(np.float32)
        intensity = (
            np.zeros(len(points), dtype=np.float32)
            if intensity is None
            else np.asarray(intensity, dtype=np.float32).reshape(-1)
        )
        labels = np.zeros(len(points), dtype=np.int64)
        labels[(points[:, 2] > 0.25) & (moving < 0.5)] = 1
        labels[(intensity > 0.72) & (moving < 0.5)] = 2
        labels[moving > 0.5] = 5
        probs = np.full(
            (len(points), self.num_classes),
            0.04 / max(self.num_classes - 1, 1),
            dtype=np.float32,
        )
        if len(points):
            probs[np.arange(len(points)), labels] = 0.96
        return labels, probs

    def infer(self, points, moving, intensity):
        _, probs = self.predict(points, intensity)
        return probs, np.asarray(moving, dtype=np.float32)


class AdaptiveElevationMap:
    """Hierarchical adaptive 2.5D map with explicit persistent 4-ary topology."""

    def __init__(self, config: FFEMConfig) -> None:
        self.cfg = config
        self.nodes: dict[tuple[int, int, int, int], HierarchyNode] = {}
        self.leaves: set[tuple[int, int, int, int]] = set()
        self.cells: dict[tuple[int, int, int, int], HierarchyNode] = {}
        self.radial = RadialResolutionPolicy()
        self.events: list[dict[str, Any]] = []
        self._limits = np.asarray(
            [b.max_radius_m for b in self.radial.bands], dtype=np.float32
        )
        self._sizes = np.asarray(
            [b.cell_size_m for b in self.radial.bands], dtype=np.float32
        )
        self._planning_forecast: dict[tuple[int, int, int, int], tuple[float, int]] = {}

    def _record_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        if len(self.events) > self.cfg.max_event_history:
            del self.events[:-self.cfg.max_event_history]

    def _band(self, x: float, y: float) -> int:
        radius = float(np.hypot(x, y))
        return int(
            np.clip(
                np.searchsorted(self._limits, radius, side="left"),
                0,
                len(self._sizes) - 1,
            )
        )

    def _root_id(self, x: float, y: float) -> tuple[int, int, int, int]:
        band = self._band(x, y)
        size = float(self._sizes[band])
        return band, 0, int(np.floor(x / size)), int(np.floor(y / size))

    def _make_node(self, node_id, parent=None, stats=None) -> HierarchyNode:
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
                stats.get(
                    "semantic_probs",
                    np.ones(self.cfg.num_classes, dtype=np.float64)
                    / self.cfg.num_classes,
                ),
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

    def _ensure_root(self, x: float, y: float):
        root_id = self._root_id(x, y)
        if root_id not in self.nodes:
            self._make_node(root_id)
            self.leaves.add(root_id)
        return root_id

    def peek_leaf(self, x: float, y: float):
        """Return an existing leaf without allocating unknown map cells."""
        root_id = self._root_id(x, y)
        if root_id not in self.nodes or root_id not in self.leaves:
            return None
        node_id = root_id
        while True:
            node = self.nodes[node_id]
            if not node.children:
                return node_id if node.active else None
            band, level, ix, iy = node_id
            size = float(self._sizes[band]) / (2.0**level)
            cx = (ix + 0.5) * size
            cy = (iy + 0.5) * size
            dx = 1 if x >= cx else 0
            dy = 1 if y >= cy else 0
            child_id = (band, level + 1, ix * 2 + dx, iy * 2 + dy)
            if child_id not in self.nodes:
                return node_id if node_id in self.leaves else None
            node_id = child_id

    def locate_leaf(self, x: float, y: float):
        """Return or create the current leaf containing x,y."""
        node_id = self._ensure_root(x, y)
        while True:
            node = self.nodes[node_id]
            if not node.children:
                return node_id
            band, level, ix, iy = node_id
            size = float(self._sizes[band]) / (2.0**level)
            cx = (ix + 0.5) * size
            cy = (iy + 0.5) * size
            dx = 1 if x >= cx else 0
            dy = 1 if y >= cy else 0
            child_id = (band, level + 1, ix * 2 + dx, iy * 2 + dy)
            if child_id not in self.nodes:
                return node_id
            node_id = child_id

    def _split(self, node: HierarchyNode) -> bool:
        if node.level >= self.cfg.max_level or node.children:
            return False
        if node.size * 0.5 < self.cfg.finest_cell_size - 1e-12:
            return False
        if len(self.leaves) + 3 > self.cfg.max_active_cells:
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
        node.children.clear()
        for dx in (0, 1):
            for dy in (0, 1):
                child_id = (
                    band,
                    level + 1,
                    ix * 2 + dx,
                    iy * 2 + dy,
                )
                self._make_node(child_id, parent=node.node_id, stats=child_stats)
                node.children.append(child_id)
                self.leaves.add(child_id)
        node.active = False
        self.leaves.discard(node.node_id)
        return True

    def _merge(self, parent: HierarchyNode) -> bool:
        """Merge four active leaf children back into their parent, including roots."""
        if len(parent.children) != 4:
            return False
        children = [self.nodes[cid] for cid in parent.children if cid in self.nodes]
        if len(children) != 4 or not all(
            child.active and not child.children for child in children
        ):
            return False
        weights = np.asarray([max(child.count, 1) for child in children], dtype=np.float64)
        parent.count = int(sum(child.count for child in children))
        parent.elevation = float(
            np.average([child.elevation for child in children], weights=weights)
        )
        parent.variance = float(
            np.average([child.variance for child in children], weights=weights)
        )
        probs = np.average(
            np.stack([child.semantic_probs for child in children]),
            axis=0,
            weights=weights,
        )
        parent.semantic_probs = probs / max(float(probs.sum()), 1e-12)
        parent.motion_probability = float(
            np.average([child.motion_probability for child in children], weights=weights)
        )
        parent.traversability = float(
            np.average([child.traversability for child in children], weights=weights)
        )
        parent.attention = float(
            np.average([child.attention for child in children], weights=weights)
        )
        parent.planning_criticality = float(
            np.average([child.planning_criticality for child in children], weights=weights)
        )
        for child in children:
            child.active = False
            self.leaves.discard(child.node_id)
            self.nodes.pop(child.node_id, None)
        parent.children = []
        parent.active = True
        parent.quiet_frames = 0
        self.leaves.add(parent.node_id)
        return True

    @staticmethod
    def _aggregate(inverse, values, groups):
        return np.bincount(
            inverse, weights=np.asarray(values, dtype=np.float64), minlength=groups
        )

    def _enforce_capacity(self) -> None:
        while len(self.leaves) > self.cfg.max_active_cells:
            parent_candidates = []
            seen = set()
            for leaf_id in self.leaves:
                parent_id = self.nodes[leaf_id].parent
                if parent_id is None or parent_id in seen:
                    continue
                seen.add(parent_id)
                parent = self.nodes.get(parent_id)
                if parent is None:
                    continue
                children = [
                    self.nodes[cid]
                    for cid in parent.children
                    if cid in self.nodes
                ]
                if len(children) == 4 and all(
                    child.active and not child.children for child in children
                ):
                    parent_candidates.append(
                        (float(np.mean([child.attention for child in children])), parent)
                    )
            if parent_candidates:
                _, parent = min(parent_candidates, key=lambda item: item[0])
                if self._merge(parent):
                    continue
            roots = [self.nodes[nid] for nid in self.leaves if self.nodes[nid].level == 0]
            if not roots:
                break
            victim = min(roots, key=lambda node: node.attention)
            self.leaves.discard(victim.node_id)
            self.nodes.pop(victim.node_id, None)

    def update(
        self,
        points,
        semantic_probs,
        motion,
        frame: int,
    ) -> dict[str, float | int]:
        t0 = time.perf_counter()
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        motion = np.asarray(motion, dtype=np.float32).reshape(-1)
        probs = np.asarray(semantic_probs, dtype=np.float32)
        if len(points) != len(motion):
            raise ValueError("motion length must match points")
        if probs.shape != (len(points), self.cfg.num_classes):
            raise ValueError(
                f"semantic_probs must have shape {(len(points), self.cfg.num_classes)}"
            )
        if len(points) == 0:
            return {
                "map_ms": (time.perf_counter() - t0) * 1000.0,
                "active_cells": len(self.leaves),
                "topology_changes": 0,
                "hierarchy_nodes": len(self.nodes),
            }

        if self._planning_forecast:
            next_forecast: dict[tuple[int, int, int, int], tuple[float, int]] = {}
            for node_id, (risk, remaining) in self._planning_forecast.items():
                if remaining <= 0:
                    continue
                node = self.nodes.get(node_id)
                targets = (
                    [node]
                    if node is not None and node.active
                    else [
                        self.nodes[child_id]
                        for child_id in (node.children if node is not None else ())
                        if child_id in self.nodes and self.nodes[child_id].active
                    ]
                )
                for target in targets:
                    target.planning_criticality = float(
                        np.clip(
                            max(target.planning_criticality, risk),
                            0.0,
                            1.0,
                        )
                    )
                if remaining > 1:
                    next_forecast[node_id] = (risk, remaining - 1)
            self._planning_forecast = next_forecast

        leaf_ids = [
            self.locate_leaf(float(x), float(y))
            for x, y in points[:, :2]
        ]
        key_dtype = [("b", "i2"), ("l", "i1"), ("x", "i8"), ("y", "i8")]
        keys = np.empty(len(points), dtype=key_dtype)
        keys["b"] = [node_id[0] for node_id in leaf_ids]
        keys["l"] = [node_id[1] for node_id in leaf_ids]
        keys["x"] = [node_id[2] for node_id in leaf_ids]
        keys["y"] = [node_id[3] for node_id in leaf_ids]
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        groups = len(unique_keys)
        counts = np.bincount(inverse, minlength=groups).astype(np.float64)
        z = points[:, 2].astype(np.float64)
        mean_z = self._aggregate(inverse, z, groups) / np.maximum(counts, 1.0)
        variance = self._aggregate(
            inverse, (z - mean_z[inverse]) ** 2, groups
        ) / np.maximum(counts, 1.0)
        motion_mean = self._aggregate(
            inverse, motion, groups
        ) / np.maximum(counts, 1.0)

        class_means = np.empty((groups, self.cfg.num_classes), dtype=np.float64)
        for class_id in range(self.cfg.num_classes):
            class_means[:, class_id] = self._aggregate(
                inverse, probs[:, class_id], groups
            ) / np.maximum(counts, 1.0)

        for gi in range(groups):
            node_id = (
                int(unique_keys["b"][gi]),
                int(unique_keys["l"][gi]),
                int(unique_keys["x"][gi]),
                int(unique_keys["y"][gi]),
            )
            node = self.nodes[node_id]
            old_elevation = node.elevation
            node.count += int(counts[gi])

            # Stable exponential fusion: current scan adapts quickly but does not
            # erase the previous map state.
            alpha = 1.0 if node.count <= int(counts[gi]) else 0.20
            node.elevation = (
                (1.0 - alpha) * node.elevation + alpha * float(mean_z[gi])
            )
            node.variance = (
                (1.0 - alpha) * node.variance + alpha * float(variance[gi])
            )
            node.semantic_probs = (
                (1.0 - alpha) * node.semantic_probs
                + alpha * class_means[gi]
            )
            node.semantic_probs /= max(float(node.semantic_probs.sum()), 1e-12)
            node.motion_probability = float(
                (1.0 - alpha) * node.motion_probability
                + alpha * float(motion_mean[gi])
            )
            node.traversability = float(
                np.clip(
                    3.0 * np.sqrt(node.variance + 1e-6)
                    + abs(node.elevation - old_elevation),
                    0.0,
                    1.0,
                )
            )
            entropy = float(
                -np.sum(
                    node.semantic_probs
                    * np.log(node.semantic_probs + 1e-8)
                )
                / np.log(self.cfg.num_classes)
            )
            x, y = node.center
            range_term = min(1.0, float(np.hypot(x, y)) / 50.0)
            node.planning_criticality *= 0.90
            node.attention = float(
                np.clip(
                    self.cfg.semantic_weight * entropy
                    + self.cfg.motion_weight * node.motion_probability
                    + self.cfg.traversability_weight * node.traversability
                    + self.cfg.geometry_weight * min(1.0, 4.0 * node.variance)
                    + self.cfg.range_weight * range_term
                    + self.cfg.planning_weight * node.planning_criticality,
                    0.0,
                    1.0,
                )
            )

        changes = 0
        refine = [
            self.nodes[nid]
            for nid in self.leaves
            if self.nodes[nid].level < self.cfg.max_level
            and self.nodes[nid].attention >= self.cfg.refine_threshold
        ]
        refine.sort(key=lambda node: node.attention, reverse=True)
        for node in refine[: self.cfg.max_topology_changes]:
            old_level = node.level
            if self._split(node):
                self._record_event(
                    {
                        "frame": frame,
                        "cell": node.node_id,
                        "old_level": old_level,
                        "new_level": old_level + 1,
                        "reason": "attention",
                        "score": node.attention,
                        "hierarchy": "split_4",
                        "parent": node.node_id,
                    }
                )
                changes += 1

        if changes < self.cfg.max_topology_changes:
            visited_parents = set()
            for leaf_id in list(self.leaves):
                node = self.nodes[leaf_id]
                if node.parent is None or node.parent in visited_parents:
                    continue
                parent = self.nodes.get(node.parent)
                if parent is None:
                    continue
                siblings = [
                    self.nodes[cid]
                    for cid in parent.children
                    if cid in self.nodes
                ]
                if len(siblings) != 4 or not all(
                    child.active and not child.children for child in siblings
                ):
                    continue
                visited_parents.add(parent.node_id)
                if any(
                    child.attention >= self.cfg.merge_threshold for child in siblings
                ):
                    parent.quiet_frames = 0
                    continue
                parent.quiet_frames += 1
                if (
                    parent.quiet_frames >= self.cfg.dwell_frames
                    and changes < self.cfg.max_topology_changes
                    and self._merge(parent)
                ):
                    self._record_event(
                        {
                            "frame": frame,
                            "cell": parent.node_id,
                            "old_level": parent.level + 1,
                            "new_level": parent.level,
                            "reason": "hysteresis",
                            "score": parent.attention,
                            "hierarchy": "merge_4",
                            "parent": parent.node_id,
                        }
                    )
                    changes += 1

        self._enforce_capacity()
        self.cells = {node_id: self.nodes[node_id] for node_id in self.leaves}
        return {
            "map_ms": (time.perf_counter() - t0) * 1000.0,
            "active_cells": len(self.leaves),
            "topology_changes": changes,
            "hierarchy_nodes": len(self.nodes),
        }

    def apply_planning_feedback(self, plan_points, risk_profile, frame: int) -> int:
        points = np.asarray(plan_points, dtype=np.float32).reshape(-1, 2)
        risks = (
            np.asarray(risk_profile, dtype=np.float32).reshape(-1)
            if len(points)
            else np.empty((0,), dtype=np.float32)
        )
        changes = 0
        horizon = max(0, int(self.cfg.predictive_dilation_frames))
        radius = max(0.0, float(self.cfg.predictive_dilation_radius_m))

        for (x, y), risk in zip(points, risks):
            node_id = self.peek_leaf(float(x), float(y))
            if node_id is None:
                continue

            candidate_ids = {node_id}
            if horizon > 0 and radius > 0.0:
                for dx, dy in (
                    (-radius, 0.0),
                    (radius, 0.0),
                    (0.0, -radius),
                    (0.0, radius),
                ):
                    neighbor = self.peek_leaf(float(x + dx), float(y + dy))
                    if neighbor is not None:
                        candidate_ids.add(neighbor)

            for candidate_id in candidate_ids:
                old_risk, old_horizon = self._planning_forecast.get(
                    candidate_id, (0.0, 0)
                )
                if horizon > 0:
                    self._planning_forecast[candidate_id] = (
                        max(float(risk), old_risk),
                        max(horizon, old_horizon),
                    )

                node = self.nodes.get(candidate_id)
                if node is None or not node.active:
                    continue

                node.planning_criticality = float(
                    np.clip(
                        max(node.planning_criticality, float(risk)),
                        0.0,
                        1.0,
                    )
                )
                node.attention = float(
                    np.clip(
                        node.attention
                        + self.cfg.planning_weight * node.planning_criticality,
                        0.0,
                        1.0,
                    )
                )

                if (
                    node.level < self.cfg.max_level
                    and node.attention >= self.cfg.refine_threshold
                    and changes < self.cfg.max_topology_changes
                ):
                    old_level = node.level
                    if self._split(node):
                        self._record_event(
                            {
                                "frame": frame,
                                "cell": node.node_id,
                                "old_level": old_level,
                                "new_level": old_level + 1,
                                "reason": "planning_feedback",
                                "score": node.attention,
                                "hierarchy": "split_4",
                                "parent": node.node_id,
                            }
                        )
                        changes += 1

        self._enforce_capacity()
        self.cells = {node_id: self.nodes[node_id] for node_id in self.leaves}
        return changes

    def arrays(self):
        active = [
            self.nodes[nid] for nid in self.leaves if self.nodes[nid].active
        ]
        if not active:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int8),
            )
        points = np.empty((len(active), 3), dtype=np.float32)
        colors = np.empty((len(active), 3), dtype=np.uint8)
        levels = np.empty(len(active), dtype=np.int8)
        palette = np.array(
            [
                [90, 90, 90],
                [70, 140, 220],
                [70, 210, 100],
                [180, 120, 60],
                [230, 70, 60],
                [220, 80, 180],
                [245, 190, 40],
            ],
            dtype=np.uint8,
        )
        for i, node in enumerate(active):
            points[i] = [node.center[0], node.center[1], node.elevation]
            colors[i] = palette[int(np.argmax(node.semantic_probs))]
            levels[i] = node.level
        return points, colors, levels

    def diagnostics(self):
        active = [
            self.nodes[nid] for nid in self.leaves if self.nodes[nid].active
        ]
        if not active:
            empty3 = np.empty((0, 3), dtype=np.float32)
            empty = np.empty((0,), dtype=np.float32)
            empty_i = np.empty((0,), dtype=np.int32)
            return empty3, empty, empty, empty, empty_i
        points = np.empty((len(active), 3), dtype=np.float32)
        traversability = np.empty(len(active), dtype=np.float32)
        uncertainty = np.empty(len(active), dtype=np.float32)
        attention = np.empty(len(active), dtype=np.float32)
        levels = np.empty(len(active), dtype=np.int32)
        for i, node in enumerate(active):
            points[i] = [node.center[0], node.center[1], node.elevation]
            traversability[i] = node.traversability
            uncertainty[i] = float(
                -np.sum(
                    node.semantic_probs
                    * np.log(node.semantic_probs + 1e-8)
                )
                / np.log(self.cfg.num_classes)
            )
            attention[i] = node.attention
            levels[i] = node.level
        return points, traversability, uncertainty, attention, levels

    def hierarchy_arrays(self):
        """Return active parent centers and levels for visualization."""
        parents = [
            node for node in self.nodes.values() if node.children
        ]
        if not parents:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int8)
        points = np.asarray(
            [[node.center[0], node.center[1], node.elevation] for node in parents],
            dtype=np.float32,
        )
        levels = np.asarray([node.level for node in parents], dtype=np.int8)
        return points, levels


class FFEMPipeline:
    """Production pipeline wrapper shared by replay and ROS 2."""

    def __init__(
        self,
        cfg: FFEMConfig | None = None,
        seed: int = 7,
        segmenter=None,
        config: FFEMConfig | None = None,
    ) -> None:
        self.cfg = config or cfg or FFEMConfig()
        self.config = self.cfg
        self.sensor = SyntheticLidar(seed=seed)
        self.map = AdaptiveElevationMap(self.cfg)
        self.mapping = self.map
        self.motion = VoxelMotionDetector()
        self.tracker = CentroidTracker()
        self.perception = segmenter or MockPerception(self.cfg.num_classes)
        self.history: list[dict[str, float | int]] = []

    def reset(self) -> None:
        """Reset all temporal state while preserving the configured segmenter."""
        segmenter = self.perception
        self.mapping = AdaptiveElevationMap(self.cfg)
        self.map = self.mapping
        self.motion = VoxelMotionDetector()
        self.tracker = CentroidTracker()
        self.perception = segmenter
        self._planning_forecast = {}
        self.history.clear()

    def _record_history(self, stats: dict[str, float | int]) -> None:
        self.history.append(dict(stats))
        if len(self.history) > self.cfg.max_history:
            del self.history[:-self.cfg.max_history]

    @staticmethod
    def _normalize_probs(probs, count: int, classes: int) -> np.ndarray:
        arr = np.asarray(probs, dtype=np.float32)
        if arr.shape != (count, classes):
            raise ValueError(
                f"segmenter returned probability shape {arr.shape}, expected {(count, classes)}"
            )
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        sums = arr.sum(axis=1, keepdims=True)
        zero = sums[:, 0] <= 1e-8
        if np.any(zero):
            arr[zero] = 0.0
            arr[zero, 0] = 1.0
            sums = arr.sum(axis=1, keepdims=True)
        return arr / np.maximum(sums, 1e-8)

    def _segment(self, points, intensity):
        if hasattr(self.perception, "predict"):
            _, probs = self.perception.predict(points, intensity)
        else:
            motion_hint = np.zeros(len(points), dtype=np.float32)
            probs, _ = self.perception.infer(points, motion_hint, intensity)
        return self._normalize_probs(probs, len(points), self.cfg.num_classes)

    def process_points(
        self,
        points,
        intensity=None,
        *,
        frame: int = 0,
        ego_transform: np.ndarray | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        intensity = (
            np.zeros(len(points), dtype=np.float32)
            if intensity is None
            else np.asarray(intensity, dtype=np.float32).reshape(-1)
        )
        if len(intensity) != len(points):
            raise ValueError("intensity length must match points")
        valid = np.isfinite(points).all(axis=1) & np.isfinite(intensity)
        points = points[valid]
        intensity = intensity[valid]
        if len(points) == 0:
            stats = {
                "points": 0,
                "moving_points": 0,
                "tracks": 0,
                "map_ms": 0.0,
                "total_ms": (time.perf_counter() - start) * 1000.0,
                "active_cells": len(self.mapping.leaves),
                "topology_changes": 0,
                "hierarchy_nodes": len(self.mapping.nodes),
            }
            self._record_history(stats)
            return {
                "points": points,
                "intensity": intensity,
                "semantic_probs": np.empty(
                    (0, self.cfg.num_classes), dtype=np.float32
                ),
                "moving": np.empty((0,), dtype=bool),
                "tracks": [],
                "stats": stats,
            }

        semantic_probs = self._segment(points, intensity)
        motion = self.motion.detect(points, ego_transform=ego_transform)
        moving = motion > 0.5
        tracks = self.tracker.update(points, motion)
        map_stats = self.mapping.update(
            points,
            semantic_probs,
            motion,
            frame,
        )
        stats = {
            **map_stats,
            "points": len(points),
            "moving_points": int(np.count_nonzero(moving)),
            "tracks": len(tracks),
            "total_ms": (time.perf_counter() - start) * 1000.0,
        }
        self._record_history(stats)
        return {
            "points": points,
            "intensity": intensity,
            "semantic_probs": semantic_probs,
            "moving": moving,
            "tracks": tracks,
            "stats": stats,
        }

    def step(self, frame: int) -> dict[str, Any]:
        """Deterministic synthetic step preserving the legacy test contract."""
        points, intensity, moving_hint = self.sensor.frame(frame)
        start = time.perf_counter()
        semantic_probs = self._segment(points, intensity)
        tracks = self.tracker.update(points, moving_hint)
        map_stats = self.mapping.update(
            points,
            semantic_probs,
            moving_hint,
            frame,
        )
        stats = {
            **map_stats,
            "points": len(points),
            "moving_points": int(np.count_nonzero(moving_hint)),
            "tracks": len(tracks),
            "total_ms": (time.perf_counter() - start) * 1000.0,
        }
        self._record_history(stats)
        return {
            "frame": frame,
            "points": points,
            "intensity": intensity,
            "semantic_probs": semantic_probs,
            "moving": moving_hint.astype(bool),
            "tracks": tracks,
            "stats": stats,
        }

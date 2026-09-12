"""Lightweight risk-aware local planner for the FFEM 2.5D map."""
from __future__ import annotations

import math
import numpy as np


class LocalRiskPlanner:
    """Choose a short robot-centric path using FFEM cell risk.

    This is a planning/output layer for the mapping stack rather than a vehicle
    controller. It scores candidate lateral offsets using traversability,
    motion, semantic uncertainty, attention, and path smoothness.
    """

    def __init__(self, lookahead_m: float = 18.0, lateral_limit_m: float = 4.0, lateral_step_m: float = 0.5):
        self.lookahead_m = float(lookahead_m)
        self.lateral_limit_m = float(lateral_limit_m)
        self.lateral_step_m = float(lateral_step_m)

    @staticmethod
    def _candidate_offsets(limit: float, step: float) -> np.ndarray:
        return np.arange(-limit, limit + 0.5 * step, step, dtype=np.float32)

    def _cell_at(self, mapping, x: float, y: float):
        r = math.hypot(x, y)
        band = int(np.clip(np.searchsorted(mapping._limits, r, side="left"), 0, len(mapping._sizes) - 1))
        size = float(mapping._sizes[band])
        key = (band, int(math.floor(x / size)), int(math.floor(y / size)))
        cell = mapping.cells.get(key)
        if cell is not None:
            return cell
        # Small neighborhood lookup makes the planner robust to cell-boundary
        # quantization and sparse observations.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = mapping.cells.get((band, key[1] + dx, key[2] + dy))
                if cell is not None:
                    return cell
        return None

    def _risk(self, cell) -> float:
        if cell is None:
            return 0.25  # unknown/free-space prior; remain conservative but usable
        p = np.asarray(cell.semantic_probs, dtype=np.float64)
        p = p / max(float(p.sum()), 1e-12)
        entropy = float(np.clip(-np.sum(p * np.log(p + 1e-8)) / np.log(len(p)), 0.0, 1.0))
        return float(
            1.50 * (1.0 - float(cell.traversability))
            + 1.25 * float(cell.motion_probability)
            + 0.75 * entropy
            + 0.75 * float(cell.attention)
        )

    def plan(self, mapping):
        offsets = self._candidate_offsets(self.lateral_limit_m, self.lateral_step_m)
        xs = np.linspace(1.5, self.lookahead_m, 36, dtype=np.float32)
        best = None
        results = []
        center_bias = 0.05 * np.abs(offsets)
        for target_y, bias in zip(offsets, center_bias):
            total = float(bias)
            points = []
            prev_y = 0.0
            for x in xs:
                y = float(target_y) * float(x / self.lookahead_m)
                cell = self._cell_at(mapping, float(x), y)
                total += self._risk(cell)
                total += 0.08 * abs(y - prev_y)
                points.append((float(x), y))
                prev_y = y
            results.append((total, float(target_y), points))
        results.sort(key=lambda v: v[0])
        best = results[0]
        return {
            "cost": float(best[0]),
            "target_lateral_m": float(best[1]),
            "points": np.asarray(best[2], dtype=np.float32),
            "candidates": [(float(c), float(y)) for c, y, _ in results],
        }

"""Lightweight risk-aware local planner for the FFEM 2.5D map."""
from __future__ import annotations
import math
import numpy as np


class LocalRiskPlanner:
    """Choose a short robot-centric path using FFEM cell risk.

    The planner scores candidate lateral trajectories using traversability,
    motion, semantic uncertainty, attention and smoothness. It also returns a
    normalized per-path risk profile so the mapper can feed planning-critical
    regions back into its adaptive resolution decision.
    """

    def __init__(self, lookahead_m: float = 18.0, lateral_limit_m: float = 4.0, lateral_step_m: float = 0.5):
        self.lookahead_m = float(lookahead_m)
        self.lateral_limit_m = float(lateral_limit_m)
        self.lateral_step_m = float(lateral_step_m)

    @staticmethod
    def _candidate_offsets(limit: float, step: float) -> np.ndarray:
        return np.arange(-limit, limit + 0.5 * step, step, dtype=np.float32)

    def _cell_at(self, mapping, x: float, y: float):
        cell_id = mapping.peek_leaf(float(x), float(y))
        return mapping.nodes.get(cell_id) if cell_id is not None else None

    def _risk(self, cell) -> float:
        if cell is None:
            return 0.25
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
        results = []
        for target_y in offsets:
            total = float(0.05 * abs(target_y))
            points = []
            risks = []
            previous_y = 0.0
            for x in xs:
                y = float(target_y) * float(x / self.lookahead_m)
                cell = self._cell_at(mapping, float(x), y)
                risk = self._risk(cell)
                total += risk + 0.08 * abs(y - previous_y)
                points.append((float(x), y))
                risks.append(risk)
                previous_y = y
            results.append((total, float(target_y), points, risks))
        results.sort(key=lambda item: item[0])
        best = results[0]
        risk_profile = np.asarray(best[3], dtype=np.float32)
        if len(risk_profile):
            lo, hi = float(risk_profile.min()), float(risk_profile.max())
            if hi > lo:
                normalized_risk = (risk_profile - lo) / (hi - lo)
            else:
                normalized_risk = np.clip(risk_profile / max(1.0, hi), 0, 1)
        else:
            normalized_risk = risk_profile
        return {
            "cost": float(best[0]),
            "target_lateral_m": float(best[1]),
            "points": np.asarray(best[2], dtype=np.float32),
            "risk_profile": normalized_risk,
            "raw_risk_profile": risk_profile,
            "mean_risk": float(risk_profile.mean()) if len(risk_profile) else 0.0,
            "max_risk": float(risk_profile.max()) if len(risk_profile) else 0.0,
            "candidates": [(float(cost), float(y)) for cost, y, _, _ in results],
        }

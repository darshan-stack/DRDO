"""Shared typed contracts for the FFEM pipeline."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class LidarFrame:
    timestamp: float
    points_xyz: np.ndarray
    intensity: np.ndarray | None = None
    ring: np.ndarray | None = None
    pose_world_sensor: np.ndarray = field(default_factory=lambda: np.eye(4))

@dataclass
class CellObservation:
    cell_id: str
    position_xy: tuple[float, float]
    heights: np.ndarray
    semantic_probs: np.ndarray
    motion_probability: float
    traversability_features: dict[str, float]
    range_m: float

@dataclass
class AdaptiveCell:
    cell_id: str
    level: int
    bounds_xy: tuple[float, float, float, float]
    elevation_mean: float
    elevation_variance: float
    semantic_probs: np.ndarray
    motion_probability: float
    traversability_cost: float
    vertical_slices: list[dict[str, Any]] = field(default_factory=list)
    last_update: float = 0.0

@dataclass
class RefinementEvent:
    timestamp: float
    cell_id: str
    old_level: int
    new_level: int
    reason: str
    attention_score: float

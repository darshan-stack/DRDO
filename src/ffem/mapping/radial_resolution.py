"""Distance-based variable resolution policy for PS 26053."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ResolutionBand:
    max_radius_m: float
    cell_size_m: float
    name: str


class RadialResolutionPolicy:
    """Deterministic radial prior with a memory-aware far field.

    Near the ego sensor the grid remains very fine. With range, the nominal
    cell size increases so low-information far-field terrain does not consume
    the same storage density as the critical near field. Semantic/motion
    attention can refine within a band without making the far field denser
    than that band's configured resolution.
    """

    def __init__(self, bands: tuple[ResolutionBand, ...] | None = None):
        # Memory-aware demo profile: preserve fine near-field detail while
        # making distant terrain materially cheaper to represent.
        self.bands = bands or (
            ResolutionBand(8.0, 0.05, "near_5cm"),
            ResolutionBand(18.0, 0.10, "mid_10cm"),
            ResolutionBand(40.0, 0.25, "mid_25cm"),
            ResolutionBand(100.0, 0.50, "far_50cm"),
        )

    def cell_size(self, x, y):
        r = float(np.hypot(x, y))
        for b in self.bands:
            if r <= b.max_radius_m:
                return b.cell_size_m
        return self.bands[-1].cell_size_m

    def band(self, x, y):
        r = float(np.hypot(x, y))
        for b in self.bands:
            if r <= b.max_radius_m:
                return b.name
        return self.bands[-1].name

    def validate(self):
        prev = 0.0
        for b in self.bands:
            if b.max_radius_m <= prev or b.cell_size_m <= 0:
                raise ValueError(
                    "Resolution bands must have increasing positive radii and cell sizes"
                )
            prev = b.max_radius_m
        return True

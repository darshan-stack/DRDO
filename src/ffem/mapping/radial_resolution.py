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
    """Deterministic radial prior, independent of semantic refinement.

    The policy gives a safe high-detail near field and bounds memory at range.
    Semantic/motion attention may refine within a band but never makes a
    far-field cell finer than the configured band unless explicitly enabled.
    """
    def __init__(self, bands: tuple[ResolutionBand,...] | None=None):
        self.bands=bands or (ResolutionBand(10.0,0.05,'near'),ResolutionBand(30.0,0.10,'mid_near'),ResolutionBand(60.0,0.25,'mid_far'),ResolutionBand(100.0,0.50,'far'))
    def cell_size(self,x,y):
        r=float(np.hypot(x,y))
        for b in self.bands:
            if r<=b.max_radius_m:return b.cell_size_m
        return self.bands[-1].cell_size_m
    def band(self,x,y):
        r=float(np.hypot(x,y))
        for b in self.bands:
            if r<=b.max_radius_m:return b.name
        return self.bands[-1].name
    def validate(self):
        prev=0.0
        for b in self.bands:
            if b.max_radius_m<=prev or b.cell_size_m<=0: raise ValueError('Resolution bands must have increasing positive radii and cell sizes')
            prev=b.max_radius_m
        return True

import numpy as np

from ffem.pipeline import FFEMConfig, AdaptiveElevationMap
from ffem.planning.local_planner import LocalRiskPlanner


def test_traversability_cost_increases_path_risk():
    cfg = FFEMConfig(max_level=0)
    mapping = AdaptiveElevationMap(cfg)
    cell_id = mapping._ensure_root(3.0, 0.0)
    cell = mapping.nodes[cell_id]
    cell.semantic_probs = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cell.attention = 0.0
    cell.motion_probability = 0.0
    cell.traversability = 0.0
    clear = LocalRiskPlanner._risk(cell)
    cell.traversability = 1.0
    rough = LocalRiskPlanner._risk(cell)
    assert rough > clear


def test_unknown_space_has_finite_risk():
    assert np.isfinite(LocalRiskPlanner._risk(None))

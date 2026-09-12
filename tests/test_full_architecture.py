import numpy as np

from ffem.pipeline import AdaptiveElevationMap, FFEMConfig


def test_explicit_four_child_hierarchy():
    mapping = AdaptiveElevationMap(FFEMConfig(max_level=2))
    root_id = mapping._ensure_root(2.0, 2.0)
    root = mapping.nodes[root_id]
    root.attention = 0.9

    assert mapping._split(root) is True
    assert len(root.children) == 4
    assert all(mapping.nodes[cid].parent == root_id for cid in root.children)
    assert all(mapping.nodes[cid].active for cid in root.children)
    assert root_id not in mapping.leaves
    assert len(mapping.leaves) == 4


def test_planning_feedback_increases_criticality_and_can_refine():
    cfg = FFEMConfig(max_level=2, refine_threshold=0.50, planning_weight=0.20)
    mapping = AdaptiveElevationMap(cfg)
    root_id = mapping._ensure_root(3.0, 1.0)
    root = mapping.nodes[root_id]
    root.attention = 0.1

    changes = mapping.apply_planning_feedback(
        np.array([[3.0, 1.0], [4.0, 1.0]], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        frame=1,
    )

    assert changes >= 1
    assert len(mapping.nodes[root_id].children) == 4
    assert any(mapping.nodes[cid].planning_criticality > 0 for cid in mapping.nodes[root_id].children)


def test_merge_requires_four_quiet_siblings():
    cfg = FFEMConfig(max_level=2, merge_threshold=0.25, dwell_frames=1)
    mapping = AdaptiveElevationMap(cfg)
    root_id = mapping._ensure_root(5.0, 1.0)
    root = mapping.nodes[root_id]
    root.attention = 0.8
    assert mapping._split(root)
    for cid in root.children:
        mapping.nodes[cid].attention = 0.0
    root.quiet_frames = 1
    assert mapping._merge(root) is True
    assert root_id in mapping.leaves
    assert len(root.children) == 0

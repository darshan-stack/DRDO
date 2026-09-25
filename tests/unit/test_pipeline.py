import numpy as np
from ffem.pipeline import FFEMConfig, FFEMPipeline

def test_pipeline_step_returns_expected_channels():
    result = FFEMPipeline(FFEMConfig(max_topology_changes=4), seed=2).step(0)
    assert result['points'].shape[1] == 3
    assert result['semantic_probs'].shape[0] == result['points'].shape[0]
    assert result['stats']['active_cells'] > 0

def test_dynamic_region_gets_attention_and_refinement():
    cfg = FFEMConfig(refine_threshold=0.05, max_topology_changes=100)
    pipe = FFEMPipeline(cfg, seed=7)
    result = pipe.step(0)
    assert result['stats']['moving_points'] > 0
    assert len(pipe.mapping.events) > 0

def test_elevation_map_limits_active_cells():
    cfg = FFEMConfig(max_active_cells=10, max_topology_changes=0)
    pipe = FFEMPipeline(cfg, seed=3)
    pipe.step(0)
    assert len(pipe.mapping.cells) <= 10


def test_pipeline_probabilities_are_normalized():
    result = FFEMPipeline(FFEMConfig(max_topology_changes=0), seed=4).step(0)
    sums = result["semantic_probs"].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_pipeline_reset_clears_temporal_map_state():
    pipe = FFEMPipeline(FFEMConfig(max_topology_changes=4), seed=5)
    pipe.step(0)
    assert pipe.mapping.nodes
    pipe.reset()
    assert not pipe.mapping.nodes
    assert not pipe.mapping.leaves
    assert not pipe.history

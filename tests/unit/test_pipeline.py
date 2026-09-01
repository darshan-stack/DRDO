import numpy as np
from ffem.pipeline import FFEMConfig, AdaptiveElevationMap, FFEMPipeline

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

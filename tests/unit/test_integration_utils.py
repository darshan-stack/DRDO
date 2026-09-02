import numpy as np
from ffem.ros2.transforms import transform_points
from ffem.evaluation.metrics import binary_precision_recall, runtime_summary

def test_transform_points_translation():
    out=transform_points(np.array([[1,2,3]],dtype=np.float32), np.array([[1,0,0,10],[0,1,0,20],[0,0,1,30],[0,0,0,1]],dtype=float))
    np.testing.assert_allclose(out, [[11,22,33]])

def test_binary_metrics_and_runtime_summary():
    m=binary_precision_recall(np.array([1,0,1]),np.array([1,1,0])); assert m['precision']==0.5 and m['recall']==0.5
    s=runtime_summary([{'total_ms':1,'map_ms':.5,'active_cells':2,'topology_changes':1},{'total_ms':3,'map_ms':1.5,'active_cells':4,'topology_changes':0}]); assert s['total_p95_ms']>=2.9 and s['frames']==2

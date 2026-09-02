"""Online metrics for FFEM simulation and real-data evaluation."""
from __future__ import annotations
import numpy as np

def binary_precision_recall(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred, truth = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    tp = float(np.sum(pred & truth)); fp = float(np.sum(pred & ~truth)); fn = float(np.sum(~pred & truth))
    return {'precision': tp / max(tp + fp, 1.0), 'recall': tp / max(tp + fn, 1.0), 'f1': 2*tp / max(2*tp + fp + fn, 1.0)}

def semantic_iou(pred: np.ndarray, truth: np.ndarray, classes: int) -> dict[str, float]:
    pred, truth = np.asarray(pred), np.asarray(truth); out = {}
    for c in range(classes):
        inter = np.sum((pred == c) & (truth == c)); union = np.sum((pred == c) | (truth == c))
        out[f'class_{c}_iou'] = float(inter / max(union, 1))
    out['mean_iou'] = float(np.mean(list(out.values()))) if out else 0.0
    return out

def runtime_summary(history: list[dict]) -> dict[str, float]:
    if not history: return {}
    total = np.asarray([float(h['total_ms']) for h in history]); map_ms = np.asarray([float(h['map_ms']) for h in history])
    return {'frames': float(len(history)), 'total_p50_ms': float(np.percentile(total, 50)), 'total_p95_ms': float(np.percentile(total, 95)), 'map_p50_ms': float(np.percentile(map_ms, 50)), 'map_p95_ms': float(np.percentile(map_ms, 95)), 'mean_active_cells': float(np.mean([h['active_cells'] for h in history])), 'mean_topology_changes': float(np.mean([h['topology_changes'] for h in history]))}

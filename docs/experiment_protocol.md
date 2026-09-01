# FFEM experiment protocol

## Baselines

Every result must compare the fixed-resolution 2.5D baseline, geometry-only adaptive cells, semantic-only adaptive cells, and full FFEM. Predictive dilation must be evaluated as a separate ablation.

## Required metrics

Report semantic IoU and per-class F1, moving-object precision/recall, track continuity, traversability accuracy, elevation RMSE, height-mode recall, semantic calibration error, dynamic ghost-trail volume, temporal boundary jitter, peak memory, active-cell count, topology changes per frame, dropped frames, and end-to-end P50/P95 latency.

## Runtime accounting

Record sensor ingestion, preprocessing, perception, attention-score computation, map fusion, split/merge, slice allocation, visualization, and total pipeline latency separately. FPS claims are invalid unless hardware, input rate, point count, model precision, and whether visualization is included are recorded.

## Research safeguards

Synthetic mock perception is only a software-integration test. It is not evidence of field performance. Real outdoor validation should include sparse long-range returns, uneven terrain, vegetation, stationary clutter, slow movers, occlusion, and multiple moving objects. Uncertainty must be calibrated before it is used as an allocation signal.

# Real perception pipeline

The perception backend is selectable through `model_backend`.

```text
fallback      -> deterministic smoke-test labels only
 torch_range  -> trained PyTorch range-image segmentation checkpoint
```

Train on SemanticKITTI after downloading the dataset:

```bash
python3 -m pip install --break-system-packages torch
python3 scripts/train_segmentation.py --data-root /path/to/SemanticKITTI --sequences 00 01 02 --epochs 20 --checkpoint models/checkpoints/range_segmentation.pt
```

Run inference with the trained checkpoint:

```bash
python3 scripts/run_replay.py --backend torch_range --checkpoint models/checkpoints/range_segmentation.pt --frames 300 --recording outputs/semantic_demo.rrd
```

For ROS 2, pass the same backend and checkpoint parameters:

```bash
ros2 run ffem_lidar_mapping ffem_node --ros-args \
  -p model_backend:=torch_range \
  -p checkpoint:=/absolute/path/range_segmentation.pt \
  -p input_topic:=/carla/ego_vehicle/lidar \
  -p map_frame:=map
```

The model expects a range image with depth and intensity channels. Its output is one probability vector per input point. SemanticKITTI labels are mapped into the seven FFEM classes defined in `semantic_kitti.py`. The fallback backend must not be used for quantitative research claims.

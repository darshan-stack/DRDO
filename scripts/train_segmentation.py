#!/usr/bin/env python3
"""Train the compact range-image segmentation model on labeled scans.

Requires torch. The script is intentionally explicit about missing datasets and
never silently falls back to mock labels.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from ffem.io.semantic_kitti import SemanticKITTIDataset
from ffem.perception.segmentation import RangeImageProjector, ProjectionConfig

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--sequences',nargs='+',default=['00']); ap.add_argument('--epochs',type=int,default=2); ap.add_argument('--checkpoint',default='models/checkpoints/range_segmentation.pt'); args=ap.parse_args()
 try: import torch; import torch.nn as nn; import torch.optim as optim
 except ImportError: raise SystemExit('PyTorch is required: install a compatible torch build for your CUDA/CPU environment.')
 ds=SemanticKITTIDataset(args.data_root,args.sequences); proj=RangeImageProjector(ProjectionConfig()); device='cuda' if torch.cuda.is_available() else 'cpu'; model=nn.Sequential(nn.Conv2d(2,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,7,1)).to(device); opt=optim.AdamW(model.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss(ignore_index=0)
 for epoch in range(args.epochs):
  model.train(); losses=[]
  for i in range(len(ds)):
   points,intensity,labels,_=ds[i]
   if labels is None: continue
   image=proj.project(points,intensity); x=np.stack([image['depth']/proj.cfg.max_range,image['intensity']],axis=0)[None]; target=np.zeros((proj.cfg.height,proj.cfg.width),dtype=np.int64); valid=image['point_index']>=0; target[valid]=labels[image['point_index'][valid]]
   logits=model(torch.from_numpy(x).float().to(device)); loss=loss_fn(logits,torch.from_numpy(target)[None].to(device)); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
  print(f'epoch={epoch+1} loss={np.mean(losses) if losses else float("nan"):.4f}')
 Path(args.checkpoint).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'classes':7,'projection':vars(proj.cfg)},args.checkpoint); print(f'saved {args.checkpoint}')
if __name__=='__main__':main()

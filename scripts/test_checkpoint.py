#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
from ffem.perception.factory import build_segmenter

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--sequence',default='00'); ap.add_argument('--frame',default='000000'); args=ap.parse_args()
    scan=f'{args.data_root}/sequences/{str(args.sequence).zfill(2)}/velodyne/{args.frame}.bin'
    points=np.fromfile(scan,dtype=np.float32).reshape(-1,4)
    seg,path=build_segmenter('auto','',7,32,1024,80.0)
    labels,probs=seg.predict(points[:,:3],points[:,3])
    print('checkpoint:',path); print('points:',len(points)); print('predicted:',len(labels)); print('classes:',np.unique(labels,return_counts=True)); print('prob_shape:',probs.shape); print('prob_sum_mean:',float(probs.sum(axis=1).mean()))
if __name__=='__main__': main()

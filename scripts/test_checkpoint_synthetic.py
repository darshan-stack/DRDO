#!/usr/bin/env python3
import numpy as np
from ffem.perception.factory import build_segmenter

def main():
    rng=np.random.default_rng(4); points=rng.normal(size=(2048,3)).astype(np.float32); points[:,0]+=20; intensity=rng.random(2048).astype(np.float32)
    seg,path=build_segmenter('auto','',7,32,1024,80.0)
    labels,probs=seg.predict(points,intensity)
    print('checkpoint:',path); print('points:',len(points)); print('predicted:',len(labels)); print('prob_shape:',probs.shape); print('classes:',np.unique(labels).tolist()); print('prob_sum_mean:',float(probs.sum(axis=1).mean()))
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Evaluate a checkpoint on labeled SemanticPOSS scans."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from ffem.io.semantic_poss import SemanticPOSSDataset, CLASS_NAMES
from ffem.perception.factory import build_segmenter

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--sequences',nargs='+',required=True); ap.add_argument('--checkpoint',default=''); ap.add_argument('--max-scans',type=int,default=0); ap.add_argument('--output',default='outputs/semanticposs_eval.json'); args=ap.parse_args()
    ds=SemanticPOSSDataset(args.data_root,args.sequences); seg,selected=build_segmenter('torch_range',args.checkpoint,7)
    cm=np.zeros((len(CLASS_NAMES),len(CLASS_NAMES)),dtype=np.int64); range_bins=[0,10,30,60,100]; by_range={f'{range_bins[i]}_{range_bins[i+1]}m': [0,0] for i in range(len(range_bins)-1)}; processed=0
    limit=len(ds) if args.max_scans<=0 else min(len(ds),args.max_scans)
    for i in range(limit):
        points,intensity,truth,_=ds[i]
        pred,_=seg.predict(points,intensity)
        for t,p in zip(truth,pred):
            if 0 <= int(t) < len(CLASS_NAMES) and 0 <= int(p) < len(CLASS_NAMES): cm[int(t),int(p)] += 1
        r=np.linalg.norm(points,axis=1)
        for j in range(len(range_bins)-1):
            m=(r>=range_bins[j])&(r<range_bins[j+1]); by_range[f'{range_bins[j]}_{range_bins[j+1]}m'][0]+=int(np.sum(m)); by_range[f'{range_bins[j]}_{range_bins[j+1]}m'][1]+=int(np.sum(pred[m]==truth[m])) if np.any(m) else 0
        processed+=1
    ious=[]
    for c in range(len(CLASS_NAMES)):
        inter=cm[c,c]; union=cm[c,:].sum()+cm[:,c].sum()-inter; ious.append(float(inter/union) if union else 0.0)
    report={'checkpoint':selected,'dataset':'semanticposs','sequences':args.sequences,'scans':processed,'class_names':list(CLASS_NAMES),'confusion_matrix':cm.tolist(),'per_class_iou':dict(zip(CLASS_NAMES,ious)),'mean_iou':float(np.mean(ious)),'overall_accuracy':float(np.trace(cm)/max(cm.sum(),1)),'accuracy_by_range':{k:{'points':v[0],'accuracy':float(v[1]/max(v[0],1))} for k,v in by_range.items()}}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()

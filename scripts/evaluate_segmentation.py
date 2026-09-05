#!/usr/bin/env python3
"""Formal SemanticPOSS evaluation for the trained FFEM range-image model."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from ffem.io.semantic_poss import SemanticPOSSDataset, CLASS_NAMES
from ffem.perception.factory import build_segmenter


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--sequences',nargs='+',required=True); ap.add_argument('--checkpoint',default=''); ap.add_argument('--max-scans',type=int,default=0); ap.add_argument('--output',default='outputs/semanticposs_eval.json'); args=ap.parse_args()
    ds=SemanticPOSSDataset(args.data_root,args.sequences)
    seg,selected=build_segmenter('torch_range',args.checkpoint,7)
    k=len(CLASS_NAMES); cm=np.zeros((k,k),dtype=np.int64); range_bins=[0,10,30,60,100]; by_range={f'{range_bins[i]}_{range_bins[i+1]}m':[0,0] for i in range(len(range_bins)-1)}
    processed=0; total_points=0
    limit=len(ds) if args.max_scans<=0 else min(len(ds),args.max_scans)
    for i in range(limit):
        points,intensity,truth,_=ds[i]; pred,_=seg.predict(points,intensity)
        truth=np.asarray(truth); pred=np.asarray(pred); valid=(truth>=0)&(truth<k)&(pred>=0)&(pred<k)
        np.add.at(cm,(truth[valid],pred[valid]),1); total_points += int(valid.sum())
        r=np.linalg.norm(points,axis=1)
        for j in range(len(range_bins)-1):
            m=(r>=range_bins[j])&(r<range_bins[j+1])&valid; by_range[f'{range_bins[j]}_{range_bins[j+1]}m'][0]+=int(m.sum()); by_range[f'{range_bins[j]}_{range_bins[j+1]}m'][1]+=int(np.sum(pred[m]==truth[m])) if np.any(m) else 0
        processed+=1
        if (i+1)%10==0 or i+1==limit: print(f'evaluated {i+1}/{limit}')
    tp=np.diag(cm).astype(float); fp=cm.sum(0)-tp; fn=cm.sum(1)-tp; support=cm.sum(1).astype(float)
    iou_den=tp+fp+fn; p_den=tp+fp; r_den=tp+fn
    iou=np.divide(tp,iou_den,out=np.zeros(k),where=iou_den>0); precision=np.divide(tp,p_den,out=np.zeros(k),where=p_den>0); recall=np.divide(tp,r_den,out=np.zeros(k),where=r_den>0)
    present=support>0
    report={'checkpoint':selected,'dataset':'semanticposs','sequences':args.sequences,'scans':processed,'points':total_points,'class_names':list(CLASS_NAMES),'confusion_matrix':cm.tolist(),'per_class_iou':dict(zip(CLASS_NAMES,iou.tolist())),'per_class_precision':dict(zip(CLASS_NAMES,precision.tolist())),'per_class_recall':dict(zip(CLASS_NAMES,recall.tolist())),'mean_iou':float(iou[present].mean()) if present.any() else 0.0,'macro_precision':float(precision[present].mean()) if present.any() else 0.0,'macro_recall':float(recall[present].mean()) if present.any() else 0.0,'overall_accuracy':float(tp.sum()/max(cm.sum(),1)),'accuracy_by_range':{q:{'points':v[0],'accuracy':float(v[1]/max(v[0],1))} for q,v in by_range.items()}}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(json.dumps({x:report[x] for x in ('mean_iou','macro_precision','macro_recall','overall_accuracy')},indent=2)); print(f'saved {out}')
if __name__=='__main__': main()

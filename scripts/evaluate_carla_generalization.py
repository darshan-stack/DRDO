#!/usr/bin/env python3
"""Quantitatively test SemanticPOSS-trained model on unseen CARLA LiDAR.

Two synchronized CARLA sensors are spawned on the existing hero vehicle:
- normal LiDAR supplies XYZ + intensity to the trained model
- semantic LiDAR supplies CARLA ground-truth semantic tags

A small voxel association maps semantic ground truth onto normal-LiDAR points.
The report contains mIoU, macro precision/recall, accuracy, matched-point
coverage and per-class IoU. This is an actual synthetic-domain generalization
test, not an unlabeled confidence proxy.
"""
from __future__ import annotations
import argparse, json, time
from collections import Counter
from pathlib import Path
import numpy as np

from ffem.io.semantic_poss import CLASS_NAMES
from ffem.perception.factory import build_segmenter

# CARLA semantic tags -> FFEM compact classes.
def carla_to_ffem(tag: int) -> int:
    if tag in (6, 7, 13, 18): return 1       # road/sidewalk/ground/terrain
    if tag == 8: return 2                     # vegetation
    if tag == 9: return 4                     # vehicle
    if tag == 4: return 5                     # pedestrian
    if tag in (1, 2, 5, 10, 11, 14, 15): return 3  # buildings/fence/pole/wall/sign/light/static
    return 6                                  # other/dynamic/water/etc.


def xyz(meas):
    a=np.frombuffer(meas.raw_data,dtype=np.float32)
    return a.reshape(-1,4)[:, :3] if a.size else np.empty((0,3),np.float32)

def semantic_xyz_tag(meas):
    a=np.frombuffer(meas.raw_data,dtype=np.float32)
    if not a.size: return np.empty((0,3),np.float32),np.empty((0,),np.int32)
    a=a.reshape(-1,6)
    return a[:, :3], a[:, 5].astype(np.int32)


def confusion(pred, gt, k=7):
    cm=np.zeros((k,k),dtype=np.int64); np.add.at(cm,(gt,pred),1); return cm

def metrics(cm):
    tp=np.diag(cm).astype(float); fp=cm.sum(0)-tp; fn=cm.sum(1)-tp; sup=cm.sum(1)
    iou_den=tp+fp+fn; p_den=tp+fp; r_den=tp+fn
    iou=np.divide(tp,iou_den,out=np.zeros(len(tp)),where=iou_den>0); p=np.divide(tp,p_den,out=np.zeros(len(tp)),where=p_den>0); r=np.divide(tp,r_den,out=np.zeros(len(tp)),where=r_den>0)
    valid=sup>0
    return {'mIoU':float(iou[valid].mean()) if valid.any() else 0.0,'macro_precision':float(p[valid].mean()) if valid.any() else 0.0,'macro_recall':float(r[valid].mean()) if valid.any() else 0.0,'accuracy':float(tp.sum()/max(cm.sum(),1)),'per_class_iou':dict(zip(CLASS_NAMES,iou.tolist())),'per_class_precision':dict(zip(CLASS_NAMES,p.tolist())),'per_class_recall':dict(zip(CLASS_NAMES,r.tolist())),'support':sup.astype(int).tolist(),'confusion_matrix':cm.astype(int).tolist()}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default='models/checkpoints/semanticposs_range_model.pt'); ap.add_argument('--host',default='127.0.0.1'); ap.add_argument('--port',type=int,default=2000); ap.add_argument('--frames',type=int,default=50); ap.add_argument('--voxel',type=float,default=0.12); ap.add_argument('--output',default='outputs/carla_generalization.json'); ap.add_argument('--max-points',type=int,default=5000); args=ap.parse_args()
    import carla
    client=carla.Client(args.host,args.port); client.set_timeout(10.0); world=client.get_world(); vehicles=list(world.get_actors().filter('vehicle.*'))
    if not vehicles: raise SystemExit('No vehicle found in CARLA. Start native stack first.')
    hero=next((v for v in vehicles if v.attributes.get('role_name')=='hero'),vehicles[0])
    bp=world.get_blueprint_library(); raw_bp=bp.find('sensor.lidar.ray_cast'); sem_bp=bp.find('sensor.lidar.ray_cast_semantic')
    attrs={'range':'80','channels':'64','points_per_second':'600000','rotation_frequency':'20','upper_fov':'10','lower_fov':'-30','sensor_tick':'0.05'}
    for k,v in attrs.items():
        if raw_bp.has_attribute(k): raw_bp.set_attribute(k,v)
        if sem_bp.has_attribute(k): sem_bp.set_attribute(k,v)
    raw_sensor=world.spawn_actor(raw_bp,carla.Transform(carla.Location(z=2.4)),attach_to=hero)
    sem_sensor=world.spawn_actor(sem_bp,carla.Transform(carla.Location(z=2.4)),attach_to=hero)
    raw_box={}; sem_box={}; cm=np.zeros((7,7),np.int64); matched=raw_total=0; frames=0; start=time.time(); segmenter,_=build_segmenter('torch_range',args.checkpoint,7)
    def raw_cb(m): raw_box[int(m.frame)]=m
    def sem_cb(m): sem_box[int(m.frame)]=m
    raw_sensor.listen(raw_cb); sem_sensor.listen(sem_cb)
    print('Collecting synchronized CARLA raw + semantic LiDAR...')
    try:
        deadline=time.time()+max(20,args.frames*0.5+10)
        while frames<args.frames and time.time()<deadline:
            world.wait_for_tick(1.0)
            common=sorted(set(raw_box).intersection(sem_box))
            while common and frames<args.frames:
                fid=common.pop(0); r=raw_box.pop(fid); s=sem_box.pop(fid); rp=xyz(r); sp,stag=semantic_xyz_tag(s); raw_total+=len(rp)
                if len(rp)>args.max_points: idx=np.linspace(0,len(rp)-1,args.max_points,dtype=int); rp=rp[idx]; inten=np.frombuffer(r.raw_data,dtype=np.float32).reshape(-1,4)[idx,3]
                else: inten=np.frombuffer(r.raw_data,dtype=np.float32).reshape(-1,4)[:,3] if len(rp) else np.empty((0,),np.float32)
                if len(rp)==0 or len(sp)==0: continue
                # Voxel association: majority semantic tag per voxel.
                keys=np.floor(sp/args.voxel).astype(np.int64); vox={}
                for q,t in zip(keys,stag): vox.setdefault(tuple(q.tolist()),[]).append(int(t))
                gt=[]; pred=[]
                for point,intv in zip(rp,inten):
                    vals=vox.get(tuple(np.floor(point/args.voxel).astype(np.int64).tolist()))
                    if not vals: continue
                    gt.append(carla_to_ffem(Counter(vals).most_common(1)[0][0]));
                if not gt: continue
                keep=[]
                for point in rp:
                    vals=vox.get(tuple(np.floor(point/args.voxel).astype(np.int64).tolist()))
                    keep.append(bool(vals))
                keep=np.asarray(keep); pp=rp[keep]; ii=inten[keep]; pred_lab,_=segmenter.predict(pp,ii); gt_arr=np.asarray(gt,np.int32)
                cm += confusion(pred_lab,gt_arr); matched+=len(gt_arr); frames+=1; print(f'frame={frames}/{args.frames} matched={len(gt_arr)}')
    finally:
        raw_sensor.stop(); sem_sensor.stop(); raw_sensor.destroy(); sem_sensor.destroy()
    res=metrics(cm); res.update({'dataset':'CARLA unseen-domain ground truth','frames':frames,'checkpoint':str(Path(args.checkpoint).resolve()),'matched_points':matched,'raw_points_seen':raw_total,'match_coverage':float(matched/max(raw_total,1)),'voxel_m':args.voxel,'note':'Ground truth comes from CARLA semantic LiDAR; normal CARLA LiDAR is the model input. Association is voxel-based.'})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(res,indent=2)); print(json.dumps({k:res[k] for k in ('mIoU','macro_precision','macro_recall','accuracy','match_coverage')},indent=2)); print(f'saved {out}')
if __name__=='__main__': main()

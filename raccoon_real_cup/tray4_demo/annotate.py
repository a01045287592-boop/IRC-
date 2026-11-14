"""Annotate only the currently approaching object before its grasp.

Raw RGB is never changed.  Derived annotations deliberately stop tracking an
object at its GRASP phase; the next segment starts tracking the next target.
"""
import argparse
from pathlib import Path
from core import read_json,atomic_json,jsonl,JsonLog

def annotate_episode(folder):
    import cv2
    from vision import load_model,detect,annotations
    folder=Path(folder).expanduser().resolve();cfg=read_json(folder/'config.json');model=load_model(cfg)
    rows=list(jsonl(folder/'frames.jsonl'));temp=folder/'annotations.jsonl.partial'
    steps={r['frame_index']:r for r in jsonl(folder/'steps.jsonl')}
    if temp.exists():temp.unlink() # Only an incomplete derived output; raw data untouched.
    log=JsonLog(temp);counts={};done=False
    try:
        for i,row in enumerate(rows):
            img=cv2.imread(str(folder/row['file']))
            if img is None:raise OSError('읽을 수 없는 RGB: '+row['file'])
            ann=annotations(detect(model,img,cfg))
            step=steps.get(row['frame_index'])
            if step and step['phase'] != 'GRASP':
                grasped = any(s['segment_id']==step['segment_id'] and s['phase']=='GRASP'
                              and s['frame_index'] <= step['frame_index'] for s in steps.values())
            else:
                grasped = True
            target = step['object_id'] if step and not grasped else None
            for role,obj in ann['objects'].items():
                if target is None:
                    obj.clear();obj['status']='not_applicable_after_grasp' if step else 'no_active_target'
                elif role != target:
                    obj.clear();obj['status']='not_active_target'
            ann['bbox_scope']=dict(target_object=target,phase=step['phase'] if step else None,
                                   rule='target only before GRASP; no bbox after GRASP')
            log.write('annotation',frame_index=row['frame_index'],image=row['file'],**ann)
            for obj in ann['objects'].values():counts[obj['status']]=counts.get(obj['status'],0)+1
            if i%50==0:print(f'후처리 YOLO {i+1}/{len(rows)}',flush=True)
        done=True
    finally:log.close()
    if done:
        temp.replace(folder/'annotations.jsonl')
        atomic_json(folder/'annotation_summary.json',dict(frames=len(rows),object_status_counts=counts,
                    source='offline YOLO segmentation + RGB material heuristic',used_for_control=False,
                    bbox_scope='only active target before GRASP; next target begins at next segment'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);a=p.parse_args();annotate_episode(a.episode)

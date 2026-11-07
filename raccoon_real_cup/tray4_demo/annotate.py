"""All raw frames receive a bbox/center row, including explicit missing detections."""
import argparse
from pathlib import Path
from core import read_json,atomic_json,jsonl,JsonLog

def annotate_episode(folder):
    import cv2
    from vision import load_model,detect,annotations
    folder=Path(folder);cfg=read_json(folder/'config.json');model=load_model(cfg)
    rows=list(jsonl(folder/'frames.jsonl'));temp=folder/'annotations.jsonl.partial'
    if temp.exists():temp.unlink() # Only an incomplete derived output; raw data untouched.
    log=JsonLog(temp);counts={};done=False
    try:
        for i,row in enumerate(rows):
            img=cv2.imread(str(folder/row['file']))
            if img is None:raise OSError('읽을 수 없는 RGB: '+row['file'])
            ann=annotations(detect(model,img,cfg))
            log.write('annotation',frame_index=row['frame_index'],image=row['file'],**ann)
            for obj in ann['objects'].values():counts[obj['status']]=counts.get(obj['status'],0)+1
            if i%50==0:print(f'후처리 YOLO {i+1}/{len(rows)}',flush=True)
        done=True
    finally:log.close()
    if done:
        temp.replace(folder/'annotations.jsonl')
        atomic_json(folder/'annotation_summary.json',dict(frames=len(rows),object_status_counts=counts,
                    source='offline YOLO segmentation + RGB material heuristic',used_for_control=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);a=p.parse_args();annotate_episode(a.episode)

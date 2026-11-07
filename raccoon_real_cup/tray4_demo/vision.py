"""Initial instance segmentation/order and offline frame annotations only."""
import math
import cv2
import numpy as np
from core import KINDS, ROLES, selection

def center(mask):
    y,x=np.nonzero(mask)
    return [float(x.mean()),float(y.mean())] if len(x) else None

def material(image,mask,cfg):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    white=(hsv[:,:,1]<=60)&(hsv[:,:,2]>=170)
    y=np.nonzero(mask)[0]
    if not len(y):return 'uncertain_cup',None
    region=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
    h=y.max()-y.min()+1
    region[:int(y.min()+h*.2)]=False;region[int(y.min()+h*.85)+1:]=False
    if region.sum()<20:return 'uncertain_cup',None
    ratio=float(white[region].mean())
    return ('glass' if ratio<=cfg['glass_max'] else 'plastic' if ratio>=cfg['plastic_min'] else 'uncertain_cup'),ratio

def load_model(cfg):
    from pathlib import Path
    from ultralytics import YOLO
    path=Path(cfg['model']).expanduser()
    if not path.is_file():raise FileNotFoundError('모델 파일 없음: '+str(path))
    model=YOLO(str(path))
    names=set(model.names.values() if isinstance(model.names,dict) else model.names)
    if model.task!='segment' or not {'bowl','cup','tray','tissue'}<=names:
        raise ValueError('bowl/cup/tray/tissue segmentation 모델이 필요합니다.')
    return model

def detect(model,image,cfg):
    r=model.predict(image,conf=cfg['confidence'],imgsz=640,retina_masks=True,verbose=False)[0]
    if r.masks is None:return []
    out=[];h,w=image.shape[:2]
    for b,m in zip(r.boxes,r.masks.data.cpu().numpy()):
        name=model.names[int(b.cls[0])]
        if name not in ('bowl','cup','tray','tissue'):continue
        if m.shape!=(h,w):m=cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)
        mask=m>.5
        if not mask.any():continue
        kind={'bowl':'plate'}.get(name,name);ratio=None
        if kind=='cup':kind,ratio=material(image,mask,cfg)
        out.append(dict(kind=kind,mask=mask,center=center(mask),box=list(map(float,b.xyxy[0].tolist())),
                        confidence=float(b.conf[0]),white_ratio=ratio))
    keep=[]
    for d in sorted(out,key=lambda x:-x['confidence']):
        if any(d['kind']==k['kind'] and (d['mask']&k['mask']).sum()>=.7*min(d['mask'].sum(),k['mask'].sum()) for k in keep):continue
        keep.append(d)
    return keep

def inventory(dets):
    trays=[d for d in dets if d['kind']=='tray' and d['confidence']>=.5]
    if not trays:raise ValueError('앞 트레이를 찾지 못했습니다.')
    tray=max(trays,key=lambda d:d['box'][3]);m=tray['mask']
    ys,xs=np.nonzero(m);band=max(int((ys.max()-ys.min()+1)*.1),1)
    bx=xs[ys>=ys.max()-band];x=(int(bx.min())+int(bx.max()))//2
    col=np.flatnonzero(m[:,x]);ref=[x,int(col[-1]) if len(col) else int(ys.max())]
    hull=cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32))
    area=np.zeros_like(m,np.uint8);cv2.fillConvexPoly(area,hull,1)
    objects=[]
    for d in dets:
        if d['kind']=='tray':continue
        mask=d['mask'];yy=np.nonzero(mask)[0];base=mask.copy()
        base[:yy.max()-int((yy.max()-yy.min()+1)*.3)]=False
        if base.sum() and (base&(area>0)).sum()/base.sum()>=.6:objects.append(d)
    # CROSS_CLASS_CUP_PLATE_V1
    # 같은 컵을 접시로도 검출한 경우에만 낮은 신뢰도의 접시 후보를 제외.
    plates = [d for d in objects if d['kind'] == 'plate']
    cups = [d for d in objects if d['kind'] in ('plastic', 'glass')]

    if len(plates) == 2:
        duplicate_ids = set()

        for plate in plates:
            for cup in cups:
                overlap = int((plate['mask'] & cup['mask']).sum())
                plate_area = int(plate['mask'].sum())
                cup_area = int(cup['mask'].sum())

                if (plate_area and cup_area
                    and overlap / plate_area >= .95
                    and overlap / cup_area >= .95
                    and cup['confidence'] >= .75
                    and plate['confidence'] <= .75
                    and cup['confidence'] - plate['confidence'] >= .04
                    and any(
                        other is not plate
                        and other['confidence'] >= .85
                        and other['confidence'] - plate['confidence'] >= .15
                        for other in plates
                    )):
                    duplicate_ids.add(id(plate))

        if len(duplicate_ids) == 1:
            objects = [d for d in objects if id(d) not in duplicate_ids]

    return selection(objects,ref)

def draw_preview(image,dets,chosen,message):
    view=image.copy();rank={o['id']:i+1 for i,o in enumerate((chosen or {}).get('objects',[]))}
    for d in dets:
        if d['kind']=='tray':continue
        role=next((r for r,k in KINDS.items() if k==d['kind']),d['kind'])

        # PREVIEW_SELECTED_ONLY_V1
        # 순서에 선택된 객체만 화면에 표시.
        if chosen and not any(
            obj['id'] == role
            and all(
                abs(a-b) < 1e-6
                for a, b in zip(obj['bbox_xyxy'], d['box'])
            )
            for obj in chosen['objects']
        ):
            continue

        text=f"{rank.get(role,'?')} {role}"
        if d['white_ratio'] is not None:text+=f" W={d['white_ratio']:.2f}"
        x1,y1,x2,y2=map(int,d['box']);cv2.rectangle(view,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.putText(view,text,(max(0,x1-15),max(40,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,255,255),1)
    if chosen:cv2.drawMarker(view,tuple(chosen['reference_pixel']),(0,255,255),cv2.MARKER_CROSS,14,1)
    cv2.putText(view,message,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
    return view

def annotations(dets):
    """No hallucinated masks/boxes; unique semantic class is a heuristic ID."""
    result={r:dict(status='not_detected',bbox_xyxy=None,bbox_corners=None,center_pixel=None) for r in ROLES}
    raw=[]
    for d in dets:
        x1,y1,x2,y2=d['box']
        raw.append(dict(kind=d['kind'],bbox_xyxy=d['box'],bbox_corners=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                        center_pixel=d['center'],white_ratio=d.get('white_ratio'),confidence=d['confidence']))
    for r,k in KINDS.items():
        candidates=[d for d in raw if d['kind']==k]
        if len(candidates)==1:result[r]=dict(status='detected_semantic_id_unverified',**candidates[0])
        elif len(candidates)>1:result[r]['status']='ambiguous_multiple_detections'
    return dict(objects=result,detections=raw,id_method='one instance per semantic type; no verified tracker; occlusion is missing')

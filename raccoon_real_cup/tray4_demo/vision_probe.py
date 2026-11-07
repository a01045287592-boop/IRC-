#!/usr/bin/env python3
"""Diagnose detection from saved RGB images or a stationary camera, without a robot."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent


def tray_fraction(detection,tray_mask,cv2,np):
    ys,xs=np.nonzero(tray_mask)
    if not len(xs):return None
    hull=cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32))
    area=np.zeros_like(tray_mask,np.uint8)
    cv2.fillConvexPoly(area,hull,1)
    mask=detection['mask'];yy=np.nonzero(mask)[0]
    if not len(yy):return None
    base=mask.copy()
    base[:yy.max()-int((yy.max()-yy.min()+1)*.3)]=False
    return float((base&(area>0)).sum()/base.sum()) if base.sum() else None


def run_worker(args,cfg):
    import cv2
    import numpy as np
    from vision import load_model,detect,inventory,draw_preview

    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
    folder=args.out.expanduser()/stamp
    folder.mkdir(parents=True,exist_ok=False)
    images=[]
    if args.image:
        image=cv2.imread(str(args.image))
        if image is None:raise RuntimeError('이미지를 열 수 없습니다: '+str(args.image))
        images=[image]
    else:
        import pyrealsense2 as rs
        pipe=rs.pipeline();rc=rs.config();started=False
        if cfg.get('camera_serial'):rc.enable_device(cfg['camera_serial'])
        rc.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
        try:
            profile=pipe.start(rc);started=True
            serial=profile.get_device().get_info(rs.camera_info.serial_number)
            print('카메라 연결:',serial,'/ 물체를 움직이지 말고 잠시 기다려 주세요.',flush=True)
            for index in range(46):
                color=pipe.wait_for_frames(2000).get_color_frame()
                if not color:raise RuntimeError('컬러 프레임을 받지 못했습니다.')
                if index in (15,30,45):images.append(np.asanyarray(color.get_data()).copy())
        finally:
            if started:pipe.stop()
    for i,image in enumerate(images):
        if not cv2.imwrite(str(folder/f'rgb_{i+1}.png'),image):raise OSError('원본 이미지 저장 실패')
    print('원본 이미지',len(images),'장 확보. 신뢰도별 검출을 확인합니다.',flush=True)
    model=load_model(cfg)
    results=[]
    for index,image in enumerate(images,1):
        for threshold in (.40,.25,.10):
            local=dict(cfg,confidence=threshold)
            dets=detect(model,image,local)
            trays=[d for d in dets if d['kind']=='tray' and d['confidence']>=.5]
            tray=max(trays,key=lambda d:d['box'][3]) if trays else None
            candidates=[]
            for d in dets:
                if d['kind']=='tray':continue
                ratio=tray_fraction(d,tray['mask'],cv2,np) if tray else None
                candidates.append(dict(kind=d['kind'],confidence=round(d['confidence'],4),
                    white_ratio=None if d['white_ratio'] is None else round(d['white_ratio'],4),
                    tray_base_fraction=None if ratio is None else round(ratio,4),
                    bbox_xyxy=[round(v,1) for v in d['box']]))
            try:
                selection=inventory(dets)
                status='네 물체 선택 가능';order=selection['order']
            except ValueError as exc:
                status=str(exc);order=None
            row=dict(frame=index,confidence_threshold=threshold,status=status,order=order,candidates=candidates)
            results.append(row)
            print(f'\n프레임 {index}, 신뢰도 {threshold:.2f}: {status}',flush=True)
            for d in candidates:
                print(' ',d['kind'],'신뢰도=',d['confidence'],'흰색비율=',d['white_ratio'],
                      '트레이내부비율=',d['tray_base_fraction'],'상자=',d['bbox_xyxy'],flush=True)
            overlay=draw_preview(image,dets,None,f'debug conf={threshold:.2f}')
            if not cv2.imwrite(str(folder/f'debug_{index}_{int(threshold*100):02d}.png'),overlay):
                raise OSError('진단 이미지 저장 실패')
    (folder/'results.json').write_text(json.dumps(dict(source_setup=str(args.setup.resolve()),
        thresholds=[.4,.25,.1],frames=len(images),results=results),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('\n진단 결과:',folder,flush=True)
    print('이 결과는 카메라/YOLO 검사이며 로봇을 연결하거나 움직이지 않았습니다.',flush=True)


def main():
    p=argparse.ArgumentParser(description='플라스틱 컵 인식 실패를 프레임과 신뢰도별로 확인')
    p.add_argument('--setup',type=Path,required=True)
    p.add_argument('--image',type=Path,help='카메라 대신 저장된 원본 이미지 사용')
    p.add_argument('--out',type=Path,default=ROOT.parents[1]/'raccoon_dataset/tray4_vision_debug')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    cfg=json.loads(args.setup.read_text(encoding='utf-8'))
    if args.worker:
        run_worker(args,cfg)
        return
    # Use the same YOLO environment and CUDA library path as the normal runner.
    from tray4 import vision_env
    py=Path(cfg['vision_python']).expanduser()
    if not py.is_file():raise FileNotFoundError('YOLO Python 없음: '+str(py))
    command=[str(py),str(Path(__file__).resolve()),'--setup',str(args.setup),
             '--out',str(args.out),'--worker']
    if args.image:command+=['--image',str(args.image)]
    result=subprocess.run(command,env=vision_env())
    raise SystemExit(result.returncode)


if __name__=='__main__':main()

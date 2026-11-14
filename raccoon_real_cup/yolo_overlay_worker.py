
# NEAREST_DEBUG_VIEW_V1
def draw_nearest_debug(view, dets, reference, key, index):
    fresh = nearest.front_tray_point(dets)
    lines = [
        "REF: " + str(reference)
        + (" CURRENT" if fresh is not None else " OLD/MISSING")
    ]

    # 검출된 트레이 범위: 청록색
    for obj in dets:
        if obj["name"] == rs_seg.TRAY:
            x1, y1, x2, y2 = map(int, obj["box"])
            cv2.rectangle(view, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # 실제 거리 계산 기준점: 노란 십자
    if reference is not None:
        ref = tuple(int(v) for v in reference)
        cv2.drawMarker(
            view, ref, (0, 255, 255),
            cv2.MARKER_CROSS, 20, 2
        )

    bowls = 0
    for obj in dets:
        if obj["name"] == rs_seg.TRAY:
            continue
        bowls += int(obj["name"] == "bowl")
        center = obj.get("center")
        if center is None:
            continue

        center = tuple(int(v) for v in center)
        eligible = obj.get("on_tray") is not None
        label = f'{obj["name"]}:{obj.get("num")}'
        distance = (
            ((obj["center"][0] - reference[0]) ** 2
             + (obj["center"][1] - reference[1]) ** 2) ** 0.5
            if reference is not None else None
        )
        distance_text = f"{distance:.1f}px" if distance is not None else "NO REF"
        state = "IN" if eligible else "OUT"
        chosen = "*" if (obj["name"], obj.get("num")) == key else ""
        lines.append(f"{chosen}{label} {distance_text} {state}")

        color = (0, 255, 0) if eligible else (0, 165, 255)
        cv2.circle(view, center, 4, color, -1)
        cv2.putText(
            view, label, (center[0] + 5, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1
        )
        if reference is not None:
            cv2.line(view, ref, center, color, 1)

    if bowls == 0:
        lines.append("BOWL NOT DETECTED")

    # 画面左上の診断表示。保存済み画像には描かない。
    for n, line in enumerate(lines):
        y = 48 + n * 18
        cv2.putText(view, line, (5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 3)
        cv2.putText(view, line, (5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)

    if index % 20 == 0:
        print("NEAREST DEBUG | " + " | ".join(lines), flush=True)

import argparse, json, os, sys, time
from pathlib import Path
import cv2
from ultralytics import YOLO
sys.path.insert(0, str(Path.home()/'yolo_ws'))
import rs_seg
import rs_seg_nearest as nearest

p=argparse.ArgumentParser()
p.add_argument('episode', type=Path)
p.add_argument('--task', choices=['plate','tissue'], required=True)
a=p.parse_args()
episode=a.episode.resolve(); events=episode/'events.jsonl'; out=episode/'yolo_overlay'
frames=out/'frames'; frames.mkdir(parents=True, exist_ok=False)
selection=out/'selection.json'; stop=out/'stop_requested'; done=out/'worker_done.json'
model_path=Path.home()/'yolo_ws/runs/segment/angle3/weights/best.pt'
if not model_path.is_file(): raise SystemExit(f'YOLO model not found: {model_path}')
model=YOLO(str(model_path)); print('YOLO classes:',model.names,flush=True)
labeler=rs_seg.InstanceLabeler(); picker=nearest.NearestPicker(); original_pick=picker.pick
locked=[None]; visible=[False]; reference=None; count=0; switches=0
track={'last_box':None,'last_time':None,'lost':False}

def pick(dets, ref, now):
    if locked[0] is None:
        return original_pick(dets,ref,now)

    if track['lost']:
        return None

    obj=next(
        (d for d in dets
         if (d.get('name'),d.get('num'))==locked[0]),
        None
    )

    if obj is not None and track['last_box'] is not None:
        x1,y1,x2,y2=obj['box']
        a,b,c,d=track['last_box']

        delta=(
            ((x1+x2-a-c)/2)**2
            + ((y1+y2-b-d)/2)**2
        )**0.5

        dt=max(0.0, now-track['last_time'])

        # 짧은 시간에 너무 멀리 바뀐 검출은 같은 물체로 확정하지 않음
        if dt > 0.5 or delta > 25+120*dt:
            obj=None

    if obj is None:
        if (
            track['last_time'] is not None
            and now-track['last_time'] > 0.5
        ):
            track['lost']=True
        return None

    track['last_box']=tuple(obj['box'])
    track['last_time']=now
    visible[0]=True
    return obj

picker.pick=pick

# 대상이 보이지 않을 때 옛 마스크를 빈 공간에 그리지 않음
original_outline=picker.outline_mask

def safe_outline(target):
    if target is None:
        return None
    return original_outline(target)

picker.outline_mask=safe_outline
cv2.namedWindow('YOLO target - S confirm / Q cancel',cv2.WINDOW_NORMAL)
print('빨간 대상과 로봇에 티칭한 위치를 확인하고 창에서 S를 누르세요. Q는 취소.',flush=True)
with events.open('r',encoding='utf-8') as source, (out/'frames.jsonl').open('x',encoding='utf-8') as log:
    pending=''; last_seen_frame=-1
    while True:
        chunk=source.read()
        if chunk: pending+=chunk
        if '\n' not in pending:
            if stop.exists(): break
            cv2.waitKey(1); time.sleep(.02); continue
        line,pending=pending.split('\n',1)
        try: ev=json.loads(line)
        except json.JSONDecodeError: continue
        if ev.get('kind')!='camera_frame': continue
        index=ev['frame_index']; name=f'{index:06d}.png'; img=cv2.imread(str(episode/ev['file']))
        if img is None: raise RuntimeError(f'cannot read frame {ev["file"]}')
        dets=rs_seg.detect(model,img,.4,640)
        visible[0]=False
        vis,reference=nearest.render(img,dets,labeler,picker,reference,ev['host_arrival_ns']/1e9)
        key=locked[0] or picker.key
        if locked[0] is None and key is not None:
            visible[0]=any((d.get('name'),d.get('num'))==key for d in dets)
        target_visible=bool(visible[0]); overlay_stale=False
        if not cv2.imwrite(str(frames/name),vis): raise OSError('overlay write failed')
        row={'frame_index':index,'host_arrival_ns':ev['host_arrival_ns'],'overlay_image':'frames/'+name,
             'target_class':key[0] if key else None,'target_number':key[1] if key else None,
             'target_visible':target_visible,'overlay_stale':overlay_stale,'locked':locked[0] is not None, 'tracking_lost':track['lost']}
        log.write(json.dumps(row,ensure_ascii=False)+'\n'); log.flush(); count+=1
        view=vis.copy(); delay=(time.monotonic_ns()-ev['host_arrival_ns'])/1e9
        display_state = 'LOST' if locked[0] and not target_visible else ('LOCKED' if locked[0] else 'READY')
        cv2.putText(view,f'{display_state} {key} lag {delay:.1f}s',
                    (5,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
        draw_nearest_debug(view, dets, reference, key, index)
        cv2.imshow('YOLO target - S confirm / Q cancel',view)
        pressed=cv2.waitKey(1)&255
        if pressed in (ord('q'),27):
            (out/'cancel').touch(); break
        if pressed==ord('s') and locked[0] is None:
            expected={'plate':'bowl','tissue':'tissue'}[a.task]
            if not key or not target_visible:
                print('대상이 보이지 않아 확인할 수 없습니다.',flush=True)
            elif key[0]!=expected:
                print(f'선택된 {key[0]}와 동작 {a.task}가 달라 시작하지 않습니다.',flush=True)
            elif delay>0.5:
                print(f'화면이 {delay:.1f}초 지연되어 선택하지 않았습니다.',flush=True)
            else:
                locked[0]=tuple(key)
                selected=next(
                    d for d in dets
                    if (d.get('name'),d.get('num'))==locked[0]
                )
                track['last_box']=tuple(selected['box'])
                track['last_time']=ev['host_arrival_ns']/1e9
                row={'target_class':key[0],'target_number':key[1],'frame_index':index,
                     'frame_host_arrival_ns':ev['host_arrival_ns'],'confirmed_ns':time.monotonic_ns()}
                temp=out/'selection.tmp';temp.write_text(json.dumps(row),encoding='utf-8');os.replace(temp,selection)
                print('선택 고정:',key,'기존 동작을 시작합니다.',flush=True)
cv2.destroyAllWindows()
done.write_text(json.dumps({'processed_frames':count,'locked_target':locked[0]}),encoding='utf-8')
print('YOLO 표시 저장 완료:',count,flush=True)

"""YOLO environment process; NEVER imports the robot SDK or sends motor commands."""
import argparse
import json
import math
import time
from pathlib import Path

from tray_core import atomic_json, digest, make_selection, validate_model_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('episode', type=Path)
    p.add_argument('--preview-only', action='store_true')
    a = p.parse_args(); out = a.episode
    cfg = json.loads((out/'config.json').read_text())
    # All frames come from the parent's one camera connection.
    import cv2
    from ultralytics import YOLO
    import tray_vision as vision
    model = YOLO(str(Path(cfg['model']['weights']).expanduser()))
    validate_model_names(model.names, cfg['model']['class_names'])
    if getattr(model, 'task', None) != 'segment':
        raise ValueError('segmentation 모델이 필요합니다.')
    atomic_json(out/'model_ready.json', {'names': model.names, 'task': model.task})
    win = 'TRAY V1 | S: confirm five objects / Q: cancel'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    stable = 0; previous = None; selection = None; message = ''
    processed = 0; total_recorded = 0; last_phase = None
    with (out/'camera.jsonl').open() as source, (out/'observations.jsonl').open('x') as log:
        while True:
            pos = source.tell(); line = source.readline()
            if not line or not line.endswith('\n'):
                source.seek(pos)
                if (out/'capture_done.json').is_file() and not line:
                    break
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), 27):
                    atomic_json(out/'cancel.json', {'reason': 'Q/ESC', 'ns': time.monotonic_ns()})
                time.sleep(.005)
                continue
            ev = json.loads(line); state = ev['state']
            img = cv2.imread(str(out/ev['file']))
            if img is None:
                raise OSError('읽지 못한 프레임: ' + ev['file'])
            lag = (time.monotonic_ns() - ev['host_arrival_ns'])/1e9
            if state['phase'] == 'waiting':
                dets = vision.detect(model, img, cfg)
                candidate = None; vis = img.copy()
                try:
                    objects, ref = vision.front_inventory(dets)
                    candidate = make_selection(objects, ref, ev['frame_index'],
                                               ev['host_arrival_ns'], digest(cfg))
                    centers = {o['id']: o['center'] for o in candidate['objects']}
                    unchanged = previous is not None and all(
                        math.dist(previous[k], centers[k]) <= 5 for k in centers)
                    stable = stable + 1 if unchanged else 1
                    previous = centers
                    ready = stable >= cfg['vision']['stable_frames']
                    atomic_json(out/'latest_inventory.json', dict(candidate, stable=ready))
                    first = candidate['objects'][0]
                    target = vision.match_target(dets, first, 1.)
                    if target is not None:
                        vis = vision.outline(vis, target['mask'])
                    # Setup labels exist ONLY before recording; no distances or lines.
                    for o in candidate['objects']:
                        x, y = map(int, o['center'])
                        cv2.putText(vis, f"{o['rank']} {o['id']}", (max(0,x-30),max(20,y-20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                    message = ('PREVIEW ONLY' if a.preview_only else 'S: confirm IDs + taught coordinates') if ready else 'Waiting for stable five objects'
                except ValueError:
                    stable = 0; previous = None
                    message = 'Need plate1 tissue1 plastic2 glass1 on FRONT tray'
                view = vis.copy()
                cv2.putText(view, message, (8,22), cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
                cv2.putText(view, 'plastic_1=left, plastic_2=right at START', (8,43),
                            cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
            else:
                if selection is None:
                    selection = json.loads((out/'selection.json').read_text())
                vis, ran, visible = vision.render_recorded(
                    img, ev, selection, lambda image: vision.detect(model,image,cfg),
                    cfg['vision']['association_gate_px'])
                view = vis  # no debug drawings/HUD on recording frames
                if state['recording']:
                    path = 'observation_frames/' + f"{ev['frame_index']:06d}.png"
                    if not cv2.imwrite(str(out/path), vis):
                        raise OSError('PNG 저장 실패')
                    row = dict(ev, image=path, raw_image=ev['file'], yolo_ran=ran,
                               target_overlay=visible, input_valid=(visible if ran else True))
                    log.write(json.dumps(row)+'\n'); log.flush()
                    total_recorded += 1
                atomic_json(out/'target_status.json', dict(object_id=state['object_id'],
                    revision=state['revision'], frame_ns=ev['host_arrival_ns'], visible=visible))
                if state['phase'] != last_phase:
                    print('화면:', state['phase'], state['object_id'], flush=True)
                    last_phase = state['phase']
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xff
            processed += 1
            atomic_json(out/'worker_status.json', {'processed': processed,
                'updated_ns': time.monotonic_ns(), 'frame_ns': ev['host_arrival_ns']})
            if key in (ord('q'), 27):
                atomic_json(out/'cancel.json', {'reason': 'Q/ESC', 'ns': time.monotonic_ns()})
            if key in (ord('s'), ord('S')) and state['phase'] == 'waiting' and not a.preview_only:
                if candidate and stable >= cfg['vision']['stable_frames'] and not (out/'selection.json').exists():
                    if (time.monotonic_ns()-ev['host_arrival_ns'])/1e9 > .5:
                        print('화면 지연으로 선택하지 않았습니다. 최신 화면에서 S를 누르세요.', flush=True)
                    else:
                        candidate['confirmed_ns'] = time.monotonic_ns()
                        atomic_json(out/'selection.json', candidate)
                        selection = candidate
                        print('5개 순서 고정:', [o['id'] for o in selection['objects']], flush=True)
    cv2.destroyAllWindows()
    atomic_json(out/'worker_done.json', {'processed': processed, 'recorded': total_recorded})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import sys
        if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
            atomic_json(Path(sys.argv[1])/'worker_error.json', {'error': repr(exc)})
        raise

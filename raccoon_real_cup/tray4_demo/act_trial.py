"""Bounded ACT hardware trial. Robot Python hosts camera; torch runs separately."""
import argparse
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from core import atomic_json, JsonLog

TRIAL_LIMIT_DEG = 50.0

def guarded_goal(action, observed, home):
    from planner import kin
    a = np.asarray(action, dtype=float)
    if a.shape != (8,) or not np.isfinite(a).all():
        raise ValueError('유효하지 않은 모델 출력')
    mode = ('horizontal', 'vertical', 'wrist')[int(a[5:].argmax())]
    kin.check_q(observed)
    for j in range(4):
        if abs(observed[j]-home[j]) > TRIAL_LIMIT_DEG:
            raise ValueError(f'관절 {j+1}: 시작 대비 실측 변화 {abs(observed[j]-home[j]):.1f}° > {TRIAL_LIMIT_DEG:g}°')
    goal = np.rad2deg(a[:4]).tolist()
    kin.check_q(goal)
    active = [3] if mode == 'wrist' else [0, 1, 2]
    for j in active:
        current_delta = abs(goal[j]-observed[j])
        start_delta = abs(goal[j]-home[j])
        if max(current_delta, start_delta) > TRIAL_LIMIT_DEG:
            raise ValueError(f'관절 {j+1}: 목표={goal[j]:.1f}°, 실측={observed[j]:.1f}°, '
                             f'시작={home[j]:.1f}°; 현재 대비 {current_delta:.1f}° / '
                             f'시작 대비 {start_delta:.1f}° (제한 {TRIAL_LIMIT_DEG:g}°)')
    return goal, mode


def worker(args):
    import torch
    from PIL import Image
    from act_train import ACTPolicy
    torch.set_num_threads(4)
    model = ACTPolicy.from_pretrained(str(args.model)).to('cpu').eval()
    folder = args.ipc
    atomic_json(folder/'ready.json', {'ready': True})
    last = -1
    while True:
        request = folder/'request.json'
        if not request.exists():
            time.sleep(.01); continue
        req = json.loads(request.read_text())
        if req['id'] == last:
            time.sleep(.01); continue
        last = req['id']
        with Image.open(folder/'frame.png') as image:
            rgb = np.array(image.convert('RGB').resize((96,96), Image.Resampling.BILINEAR), dtype=np.float32)/255
        batch = {'observation.images.front': torch.from_numpy(rgb.transpose(2,0,1).copy()).unsqueeze(0),
                 'observation.state': torch.tensor([req['state']], dtype=torch.float32)}
        with torch.inference_mode():
            action = model.predict_action_chunk(batch)[0,0].cpu().tolist()
        atomic_json(folder/'response.json', {'id':last, 'action':action})


def trial(args):
    import cv2
    import pyrealsense2 as rs
    from roboid import Raccoon
    from control import NativePort
    from planner import kin
    robot = port = pipe = process = log = timer = None
    stopped = threading.Event()
    gate = threading.Lock()
    args.out.mkdir(parents=True, exist_ok=True)
    def stop():
        with gate:
            stopped.set()
            if port is not None and args.execute:
                port.stop()
    try:
        with tempfile.TemporaryDirectory(prefix='act_trial_') as tmp:
            folder = Path(tmp)
            with (args.out/'inference.log').open('w') as worker_log:
                process = subprocess.Popen([str(args.policy_python), str(Path(__file__).resolve()),
                    '--worker', '--ipc', tmp, '--model', str(args.model.resolve())], stdout=worker_log, stderr=subprocess.STDOUT)
                deadline = time.monotonic()+60
                while not (folder/'ready.json').exists():
                    if process.poll() is not None or time.monotonic()>deadline:
                        raise RuntimeError('모델 로딩 실패: inference.log 확인')
                    time.sleep(.1)
                robot = Raccoon(port_name=args.port)
                deadline = time.monotonic()+8
                while not robot.is_connected():
                    if time.monotonic()>deadline: raise RuntimeError('로봇 연결 실패')
                    time.sleep(.1)
                log = JsonLog(args.out/'events.jsonl')
                port = NativePort(robot, 5, log)
                port.gripper = 0  # Trial requires an already-open, empty gripper.
                camera = rs.pipeline(); config = rs.config()
                if args.serial: config.enable_device(args.serial)
                config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
                camera.start(config); pipe = camera
                print('빈 집게를 미리 열고 시작 자세로 준비하세요. 카메라 창 Enter: 시작 / Q: 취소', flush=True)
                while True:
                    frame = pipe.wait_for_frames(1000).get_color_frame()
                    if not frame: continue
                    cv2.imshow('ACT trial - ENTER start / Q stop', np.asanyarray(frame.get_data()))
                    key = cv2.waitKey(20)&255
                    if key in (ord('q'),27): return
                    if key in (10,13): break
                home = port.read()
                kin.check_q(home)
                end = time.monotonic()+args.seconds
                timer = threading.Timer(args.seconds, stop); timer.start()
                index = 0
                while not stopped.is_set():
                    cycle = time.monotonic()
                    frame = pipe.wait_for_frames(500).get_color_frame()
                    if not frame: raise RuntimeError('카메라 프레임 없음')
                    bgr = np.asanyarray(frame.get_data()).copy()
                    observed = port.read()
                    if not cv2.imwrite(str(folder/'frame.png'), bgr): raise OSError('프레임 저장 실패')
                    atomic_json(folder/'request.json', {'id':index, 'state':np.deg2rad(observed).tolist()+[0.]})
                    response = None
                    while not stopped.is_set():
                        if process.poll() is not None: raise RuntimeError('추론 프로세스 종료')
                        if (folder/'response.json').exists():
                            candidate = json.loads((folder/'response.json').read_text())
                            if candidate['id']==index: response=candidate; break
                        if time.monotonic()-cycle > .5: raise RuntimeError('추론 지연 0.5초 초과')
                        time.sleep(.005)
                    if response is None: break
                    goal, mode = guarded_goal(response['action'], port.read(), home)
                    log.write('prediction', **response, goal_deg=goal, mode=mode)
                    with gate:
                        if stopped.is_set() or time.monotonic()>=end: break
                        if args.execute: port.apply(goal, 0, mode, port.read())
                    print(f'{index}: {mode} goal={np.round(goal,1).tolist()}', flush=True)
                    cv2.imshow('ACT trial - ENTER start / Q stop', bgr)
                    if cv2.waitKey(1)&255 in (ord('q'),27): break
                    index += 1
                    stopped.wait(max(0,.1-(time.monotonic()-cycle)))
    finally:
        stop()
        if timer: timer.cancel(); timer.join()
        if process:
            process.terminate(); process.wait(timeout=5)
        if pipe: pipe.stop()
        if robot: robot.dispose()
        if log: log.close()
        cv2.destroyAllWindows()
        print('시험 종료. 자동 복귀하지 않습니다.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('raccoon_dataset/tray4_act_lighting_v1/act_tray4_policy'))
    # Preserve the venv symlink path: resolve() selects the base interpreter.
    p.add_argument('--policy-python', type=Path,
                   default=Path(__file__).absolute().parents[2]/'.venv/bin/python')
    p.add_argument('--port', default='/dev/ttyACM2')
    p.add_argument('--serial')
    p.add_argument('--seconds', type=float, default=3.)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--out', type=Path, default=Path('raccoon_dataset/act_trials')/time.strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--ipc', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 0 < args.seconds <= 30: p.error('시험 시간은 0초 초과, 30초 이하')
    if args.worker: worker(args)
    else:
        try: trial(args)
        except (Exception, KeyboardInterrupt) as exc:
            print(f'중단: {exc}', flush=True)
            raise SystemExit(1)

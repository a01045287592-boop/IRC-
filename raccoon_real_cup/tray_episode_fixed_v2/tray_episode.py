#!/usr/bin/env python3
"""One tray / five objects / one recording. Default: check only, no hardware."""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from tray_core import (VERSION, ROLES, EpisodeState, atomic_json, config_errors,
                       digest, point, validate_selection)

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def yolo_env(python):
    env = os.environ.copy()
    venv = python.parent.parent
    libs = sorted((venv/'lib').glob('python*/site-packages/nvidia/cudnn/lib'))
    if libs:
        env['LD_LIBRARY_PATH'] = str(libs[0]) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env


def check(cfg):
    errors = config_errors(cfg)
    for msg in errors:
        print('미완료:', msg)
    if errors:
        print('CHECK: 설정 대기. 카메라/로봇 연결 없음.')
        return False
    import tray_motion as motion
    motion.preflight(cfg, list(ROLES), [0,-10,-140,60])
    print('CHECK PASS: 예시 시작 자세의 경로 검사. 실제 실행 전 실측 시작 자세와 YOLO 순서로 다시 검사합니다.')
    print('현재 배치 고정 좌표용. 순서만 실시간 영상으로 계산합니다. 실제 충돌/파지는 미검증입니다.')
    return True


class Session:
    def __init__(self, out, cfg):
        self.out, self.cfg = out, cfg
        self.state = EpisodeState()
        self.stop_camera = threading.Event()
        self.errors = []; self.frame_count = 0; self.recorded_count = 0
        self.event_lock = threading.Lock()
        self.events = (out/'events.jsonl').open('x', encoding='utf-8')
        (out/'camera.jsonl').touch(exist_ok=False)
        self.child = None; self.tap = None; self.last_packet_ns = None

    def log(self, kind, **values):
        if kind == 'joint_packet':
            self.last_packet_ns = values['read_end_ns']
        _, state = self.state.snapshot()
        row = dict(kind=kind, host_monotonic_ns=time.monotonic_ns(), state=state, **values)
        with self.event_lock:
            self.events.write(json.dumps(row, ensure_ascii=False)+'\n')
            self.events.flush()

    def transition(self, snapshot):
        self.log('episode_phase', phase_state=snapshot)

    def camera(self):
        pipe = None
        try:
            import numpy as np
            import pyrealsense2 as rs
            from PIL import Image
            pipe = rs.pipeline(); conf = rs.config()
            conf.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            profile = pipe.start(conf)
            atomic_json(self.out/'camera_ready.json', {'serial': profile.get_device().get_info(rs.camera_info.serial_number)})
            for _ in range(30):
                if self.stop_camera.is_set():
                    return
                pipe.wait_for_frames(5000)
            deadline = 0
            with (self.out/'camera.jsonl').open('a', encoding='utf-8') as frames:
                while not self.stop_camera.is_set():
                    fs = pipe.wait_for_frames(5000); color = fs.get_color_frame()
                    if not color:
                        continue
                    # Snapshot and host timestamp are acquired under the same state lock.
                    now, state = self.state.snapshot()
                    if now < deadline:
                        continue
                    missed = 0 if not deadline else max(0, (now-deadline)//100_000_000)
                    deadline = now + 100_000_000 if not deadline else deadline + (missed+1)*100_000_000
                    folder = 'rgb_frames' if state['recording'] else 'preview_frames'
                    name = folder + '/' + f'{self.frame_count:06d}.png'
                    Image.fromarray(np.asanyarray(color.get_data()).copy()).save(self.out/name)
                    ev = dict(kind='camera_frame', frame_index=self.frame_count, file=name,
                              host_arrival_ns=now, camera_timestamp_ms=color.get_timestamp(),
                              camera_frame_number=color.get_frame_number(), state=state,
                              skipped_schedule_slots=int(missed))
                    frames.write(json.dumps(ev)+'\n'); frames.flush()
                    self.frame_count += 1
                    if state['recording']:
                        self.recorded_count += 1
        except BaseException as exc:
            self.errors.append(repr(exc))
        finally:
            if pipe is not None:
                try:
                    pipe.stop()
                except Exception as exc:
                    self.errors.append(repr(exc))
            atomic_json(self.out/'capture_done.json', dict(frames=self.frame_count, recorded=self.recorded_count,
                                                          errors=self.errors))

    def health(self, packets=True):
        if self.errors:
            raise RuntimeError('카메라 오류: ' + str(self.errors))
        if (self.out/'cancel.json').is_file():
            raise RuntimeError('화면에서 중단 요청을 받았습니다.')
        if (self.out/'worker_error.json').is_file():
            raise RuntimeError('YOLO 오류: ' + str(load(self.out/'worker_error.json')))
        if self.child is not None and self.child.poll() is not None:
            raise RuntimeError('YOLO 프로세스 종료: ' + str(self.child.returncode))
        if packets and self.tap is not None:
            if self.tap.errors or self.tap.dropped:
                raise RuntimeError('관절 패킷 기록 오류/큐 누락')
            if self.last_packet_ns is not None and (time.monotonic_ns()-self.last_packet_ns)/1e9 > .5:
                raise RuntimeError('관절 패킷이 0.5초 넘게 갱신되지 않았습니다.')

    def wait(self, seconds):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            self.health()
            time.sleep(min(.05, max(0,deadline-time.monotonic())))


class RecordedRobot:
    COMMANDS = {'set_degree','open_gripper','close_gripper','lock_horz','lock_vert','unlock','stop'}
    def __init__(self, raw, session):
        self.raw, self.s = raw, session

    def __getattr__(self, name):
        method = getattr(self.raw, name)
        if name not in self.COMMANDS:
            return method
        def call(*args, **kw):
            if name != 'stop':
                self.s.health()
            if name == 'close_gripper':
                self.s.transition(self.s.state.close_command())
                print('파지 명령: 테두리 OFF / YOLO 추론 OFF', flush=True)
            elif name == 'open_gripper':
                self.s.transition(self.s.state.open_command())
            begin = time.monotonic_ns()
            self.s.log('robot_command_start', command=name, args=list(args), kwargs=kw,
                       call_start_ns=begin)
            try:
                result = method(*args, **kw)
            except BaseException as exc:
                self.s.log('robot_command_error', command=name, error=repr(exc), call_start_ns=begin)
                raise
            self.s.log('robot_command_end', command=name, call_start_ns=begin, call_end_ns=time.monotonic_ns())
            return result
        return call


def execute(session, cfg, selection, port):
    import tray_motion as motion
    from importlib.metadata import version
    if version('roboid') != motion.bench_module.SDK_VERSION:
        raise ValueError('기존 코드와 같은 roboid==1.7.3 환경이 필요합니다.')
    from roboid import Raccoon
    from packet_tap import PacketTap
    raw = None
    try:
        raw = Raccoon(port_name=port)
        session.tap = PacketTap(raw._roboid, session.log)
        deadline = time.monotonic()+8
        while session.last_packet_ns is None:
            session.health(packets=False)
            if time.monotonic() > deadline:
                raise RuntimeError('관절 수신 대기 시간 초과')
            time.sleep(.05)
        proxy = RecordedRobot(raw, session)
        class CheckedBench(motion.bench_module.Bench):
            def read(self):
                session.health()
                return super().read()
        bench = CheckedBench(proxy,cfg['speed'],ROOT/'unused_pose.json',2,2)
        home = bench.read()
        order = [o['id'] for o in selection['objects']]
        plans = motion.preflight(cfg, order, home)
        atomic_json(session.out/'motion_plan.json', dict(start_q=home, plans=plans))
        # Reject movement of any target while connecting/preflighting.
        latest = load(session.out/'latest_inventory.json')
        if not latest.get('stable') or (time.monotonic_ns()-latest['frame_ns'])/1e9 > .75:
            raise RuntimeError('로봇 연결 후 5개 대상의 최신 안정 화면이 없습니다. 이동하지 않았습니다.')
        latest_map = {o['id']: o for o in latest['objects']}
        import math
        if any(math.dist(o['center'], latest_map[o['id']]['center']) > 10 for o in selection['objects']):
            raise RuntimeError('선택 후 물체 위치가 바뀌었습니다. 이동하지 않았습니다.')
        session.transition(session.state.start(order))
        plastic_placed = 0
        for rank, role in enumerate(order, 1):
            # Replan from measured current posture. A failure stops the entire episode.
            c, stages = motion.plan_object(cfg, role, bench.read(), home, plastic_placed)
            snapshot = session.state.begin_object(role); session.transition(snapshot)
            print(f'대상 {rank}/5: {role} — 빨간 테두리 ON', flush=True)
            deadline = time.monotonic()+5
            while True:
                session.health()
                status_path = session.out/'target_status.json'
                status = load(status_path) if status_path.is_file() else {}
                if (status.get('object_id') == role and status.get('revision') == snapshot['revision']
                        and status.get('visible') and (time.monotonic_ns()-status['frame_ns'])/1e9 <= .5):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(role + ': 파지 위치에서 대상 재확인 실패. 다음 이동을 시작하지 않습니다.')
                time.sleep(.05)
            for stage in stages:
                session.health()
                session.log('motion_phase', object_id=role, name=stage['name'], stage=stage)
                print(role, stage['name'], flush=True)
                if stage['kind'] in ('move','rotate'):
                    motion.execute_stage(proxy,bench,stage,lambda event, **kw: session.log('motion_log',event=event,**kw))
                elif stage['kind'] == 'open':
                    proxy.open_gripper(); session.wait(.5)
                elif stage['kind'] == 'close':
                    proxy.close_gripper(); session.wait(.5)
                elif stage['kind'] == 'hold':
                    session.wait(c.get('hold_seconds',.5))
            session.log('object_commands_complete', object_id=role, physical_success='unverified')
            session.transition(session.state.complete_object())
            if role.startswith('plastic_'):
                plastic_placed += 1
        # Global episode start, even if intermediate replanning accumulated small drift.
        for stage in motion.home_stages(home, cfg['clearance_z_cm']):
            stage['engine'] = 'plate'
            motion.execute_stage(proxy,bench,stage,lambda event, **kw: session.log('motion_log',event=event,**kw))
        session.wait(.3)
        session.transition(session.state.finish())
    finally:
        if raw is not None:
            try:
                raw.stop()
            finally:
                try:
                    if session.tap:
                        session.tap.close()
                finally:
                    raw.dispose()


def run(cfg, config_path, preview_only, port, output_root):
    python = Path(cfg['model']['python']).expanduser()
    weights = Path(cfg['model']['weights']).expanduser()
    if not python.is_file() or not weights.is_file():
        raise ValueError('YOLO python 또는 모델 파일을 찾지 못했습니다.')
    out = output_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('rgb_frames','observation_frames','preview_frames'):
        (out/folder).mkdir()
    atomic_json(out/'config.json',cfg)
    meta = dict(version=VERSION, status='preview' if preview_only else 'waiting', success='unverified',
                objects_expected=5, inputs='observation_frames: red before close command; raw after',
                joint_unit='degree', wrist_unit='cm', time_base='host time.monotonic_ns',
                physical_grasp_verified=False, action_labels='not generated; see robot_command_start events',
                config_source=str(config_path.resolve()))
    atomic_json(out/'metadata.json',meta)
    session = Session(out,cfg)
    thread = threading.Thread(target=session.camera, name='camera', daemon=True)
    failure = None
    print('기록 폴더:',out,flush=True)
    try:
        command = [str(python),str(ROOT/'tray_worker.py'),str(out)]
        if preview_only:
            command.append('--preview-only')
        session.child = subprocess.Popen(command, env=yolo_env(python))
        # Validate new material classes before opening hardware.
        deadline = time.monotonic()+90
        while not (out/'model_ready.json').is_file():
            session.health(packets=False)
            if time.monotonic() > deadline:
                raise RuntimeError('모델 준비 시간 초과')
            time.sleep(.05)
        thread.start()
        print('현재 배치 고정 좌표 사용. 5개 물체와 순서 확인 후 S 시작 / Q 취소.',flush=True)
        while True:
            session.health(packets=False)
            if not preview_only and (out/'selection.json').is_file():
                selection = load(out/'selection.json')
                validate_selection(selection,cfg)
                execute(session,cfg,selection,port)
                meta['status'] = 'controller_returned_unverified'
                break
            time.sleep(.05)
    except (Exception, KeyboardInterrupt) as exc:
        failure = exc
        meta['status'] = 'preview_closed' if preview_only and (out/'cancel.json').is_file() else 'interrupted_or_failed'
        meta['error'] = repr(exc)
        session.transition(session.state.finish(failed=True))
        print('종료:',str(exc) or 'Ctrl+C',flush=True)
    finally:
        session.stop_camera.set()
        if thread.ident is not None:
            thread.join(timeout=7)
            if thread.is_alive():
                session.errors.append('camera thread did not stop')
        else:
            atomic_json(out/'capture_done.json',dict(frames=0,recorded=0,errors=session.errors))
        if session.child is not None:
            try:
                session.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                session.child.terminate()
                try:
                    session.child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    session.child.kill();session.child.wait()
        meta.update(camera_errors=session.errors, frames=session.recorded_count,
                    commands_completed=list(session.state.finished),
                    yolo_exit_code=session.child.returncode if session.child else None)
        if session.tap:
            meta['packets'] = dict(count=session.tap.count,dropped=session.tap.dropped,errors=session.tap.errors)
        done = load(out/'worker_done.json') if (out/'worker_done.json').is_file() else {}
        meta['overlay_complete'] = done.get('recorded') == session.recorded_count and session.child.returncode == 0
        if session.errors or (not preview_only and not meta['overlay_complete']):
            meta['status'] = 'recording_incomplete'
        atomic_json(out/'metadata.json',meta)
        session.events.close()
    print('저장:',out)
    print('상태:',meta['status'],'/ 실제 성공:',meta['success'])
    print('학습 입력용 이미지:',out/'observation_frames')
    print('관절 패킷·명령:',out/'events.jsonl')
    print('파지 명령 기준 전환이며 실제 파지 성공을 자동 판정한 결과는 아닙니다.')
    return 0 if meta['status'] in ('controller_returned_unverified','preview_closed') else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'tray_config.json')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--preview',action='store_true',help='카메라/YOLO만 실행, 로봇 연결 없음')
    modes.add_argument('--execute',action='store_true',help='5개 실제 동작과 연속 기록')
    modes.add_argument('--model-info',action='store_true')
    modes.add_argument('--set-point',nargs=4,metavar=('KEY','X','Y','Z'))
    modes.add_argument('--set-model')
    modes.add_argument('--set-stack-step-mm',type=float)
    p.add_argument('--port',default='/dev/ttyACM0')
    p.add_argument('--out',type=Path,default=Path.home()/'Raccoonbot_Cafe_Clean/raccoon_dataset/tray_raw')
    args=p.parse_args();cfg=load(args.config)
    if args.set_point or args.set_model or args.set_stack_step_mm is not None:
        updated=copy.deepcopy(cfg)
        if args.set_point:
            key,*values=args.set_point
            group, sep, name=key.partition('.')
            if group not in ('pickups','destinations') or not sep or name not in cfg[group]:
                p.error('KEY는 pickups.plate 또는 destinations.cup_pour 같은 기존 항목이어야 합니다.')
            values=list(map(float,values))
            if not point(values): p.error('유한한 손목 좌표 3개 필요')
            updated[group][name]=values
        elif args.set_model:
            path=Path(args.set_model).expanduser().resolve()
            if not path.is_file(): p.error('모델 파일이 없습니다.')
            updated['model']['weights']=str(path)
        else:
            if not 0<args.set_stack_step_mm<=30: p.error('0 초과 30mm 이하 필요')
            updated['plastic_stack_step_mm']=args.set_stack_step_mm
        backup=args.config.with_name(args.config.name+'.before_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        shutil.copy2(args.config,backup)
        atomic_json(args.config,updated)
        print('설정 저장:',args.config,'/ 로봇 연결 없음')
        return 0
    if args.model_info:
        if not cfg['model']['weights']: p.error('--set-model로 학습한 모델 경로부터 지정하세요.')
        python=Path(cfg['model']['python']).expanduser()
        return subprocess.call([str(python),str(ROOT/'tray_model_check.py'),str(args.config.resolve())],env=yolo_env(python))
    if not (args.execute or args.preview):
        return 0 if check(cfg) else 2
    if args.execute and not check(cfg):
        return 2
    if not cfg['model']['weights']:
        p.error('모델 경로를 --set-model로 지정하세요.')
    return run(cfg,args.config,args.preview,args.port,args.out.expanduser().resolve())


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError) as exc:
        raise SystemExit('실행하지 못했습니다: '+str(exc))

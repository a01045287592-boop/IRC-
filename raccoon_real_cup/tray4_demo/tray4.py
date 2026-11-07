#!/usr/bin/env python3
"""Terminal entry point: new -> teach four XYZs -> check -> run --execute."""
import argparse
import copy
import json
import math
import os
import select
import subprocess
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from core import (VERSION,ROLES,LABELS,SCHEMA,default_config,read_json,atomic_json,compatible_selections,
                  validate_config,finite_point,JsonLog,hash_data)
from planner import kin,plan_episode

ROOT=Path(__file__).resolve().parent

def vision_env():
    env=os.environ.copy();venv=Path.home()/'yolo_ws/.venv/lib'
    libs=list(venv.glob('python*/site-packages/nvidia/cudnn/lib'))
    if libs:env['LD_LIBRARY_PATH']=str(libs[0])+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env

def ask(prompt,guard=lambda:None):
    print(prompt,end='',flush=True)
    while True:
        guard()
        if select.select([sys.stdin],[],[],.1)[0]:
            line=sys.stdin.readline()
            if line=='':raise EOFError('터미널 입력 종료')
            return line.strip()

def wait_until(predicate,guard,timeout,label):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        guard();result=predicate()
        if result:return result
        time.sleep(.02)
    raise TimeoutError(label)


def run(args):
    cfg=read_json(args.setup);validate_config(cfg)
    if not args.execute:
        # Offline mode still reports an example plan without opening hardware.
        plan_episode(cfg,list(ROLES),[0.,-10.,-140.,60.])
        print('기구학/순서 검사만 완료했습니다. 실제 실행에는 --execute가 필요합니다.');return
    import numpy  # Verify export dependency before connecting hardware.
    if version('roboid')!=kin.SDK_VERSION:raise ValueError('roboid==1.7.3 환경에서 실행하세요.')
    py=Path(cfg['vision_python']).expanduser()
    if not py.is_file():raise FileNotFoundError('YOLO Python 없음: '+str(py))
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder=args.out.expanduser()/stamp;folder.mkdir(parents=True,exist_ok=False)
    atomic_json(folder/'config.json',cfg);atomic_json(folder/'schema.json',SCHEMA)
    atomic_json(folder/'camera_control.json',dict(mode='select'))
    meta=dict(version=VERSION,status='starting',success='unverified',episode_id=stamp,
              object_count=4,objects=[],config_hash=hash_data(cfg),simulated=False,
              empty_run=args.empty_run,sdk=kin.SDK_VERSION,instruction=cfg['instruction'])
    atomic_json(folder/'metadata.json',meta)
    log=JsonLog(folder/'events.jsonl');steps=JsonLog(folder/'steps.jsonl')
    worker_log=(folder/'camera_worker.log').open('x',encoding='utf-8')
    process=None;robot=None;port=None;sampler=None;failure=None
    print('에피소드 저장:',folder,flush=True)
    try:
        process=subprocess.Popen([str(py),str(ROOT/'camera_worker.py'),'--episode',str(folder)],
                                 env=vision_env(),stdout=worker_log,stderr=subprocess.STDOUT)
        def camera_guard():
            if (folder/'camera_error.json').exists():raise RuntimeError(read_json(folder/'camera_error.json')['error'])
            if process.poll() is not None:raise RuntimeError('카메라 프로세스 종료. camera_worker.log 확인')
        def ready():
            if not (folder/'ready.json').exists():return None
            row=read_json(folder/'ready.json')
            return row if row['ready'] and time.monotonic_ns()-row['host_ns']<2_000_000_000 else None
        print('처음에만 YOLO를 사용합니다. 네 물체가 보이도록 준비하세요. 터미널 Q Enter: 취소',flush=True)
        last_print=0
        while True:
            camera_guard();r=ready()
            if r:break
            if time.monotonic()-last_print>2:
                row=read_json(folder/'ready.json') if (folder/'ready.json').exists() else {}
                print('인식 중:',row.get('status') or '모델/카메라 준비 또는 순서 안정화 중',flush=True);last_print=time.monotonic()
            if select.select([sys.stdin],[],[],.1)[0]:
                entered=sys.stdin.readline()
                if entered=='':raise EOFError('터미널 입력 종료')
                if entered.strip().lower()=='q':raise KeyboardInterrupt()
        print(' → '.join(f'{i+1}. {LABELS[o["id"]]} ({o["distance_pixel"]:.1f}px)' for i,o in enumerate(r['selection']['objects'])))
        for role in ROLES:print(LABELS[role], '티칭 손목 cm:',cfg['pickups'][role])
        shown_selection=r['selection']
        reply=ask('물체/좌표 일치, 시작 자세 수평, 집게 비어 있음 확인 후 Enter: 시작 / Q: 취소 > ',camera_guard)
        if reply:raise KeyboardInterrupt('시작 취소')
        # WAIT_FOR_CONFIRMED_SCENE_V1
        print(
            '확인한 배치가 다시 안정될 때까지 기다립니다 (최대 30초).',
            flush=True,
        )

        def stable_confirmed():
            current = ready()
            if current and compatible_selections(
                shown_selection,
                current['selection'],
                cfg['stable_pixels'],
            ):
                return current
            return None

        r = wait_until(
            stable_confirmed,
            camera_guard,
            30,
            '확인한 배치가 안정되지 않았습니다. 다시 시작하세요.',
        )
        atomic_json(folder/'camera_control.json',dict(mode='freeze',selection=r['selection']))
        chosen=wait_until(lambda:read_json(folder/'selection.json') if (folder/'selection.json').exists() else None,
                          camera_guard,30,'순서 확정 응답 없음')
        meta['order']=chosen['order'];meta['status']='recording'
        from roboid import Raccoon
        from control import NativePort,Sampler,Controller,build_step
        robot=Raccoon(port_name=args.port)
        wait_until(robot.is_connected,camera_guard,8,'로봇 연결 실패')
        port=NativePort(robot,cfg['speed'],log);home=port.read()
        plans=plan_episode(cfg,chosen['order'],home)
        if args.empty_run:
            for obj in plans:
                for s in obj['stages']:
                    if s['kind']=='close':s['kind']='open'
        atomic_json(folder/'plan.json',dict(home_qpos_deg=home,objects=plans,
                    scope='joint-range/native-mode plan; no collision model'))
        meta['home_qpos_deg']=home;atomic_json(folder/'metadata.json',meta)
        sampler=Sampler(port,log,20);sampler.start()
        def guard():
            camera_guard()
            if sampler.error:raise RuntimeError(sampler.error)
        time.sleep(.2)
        guard()
        sampler.before(time.monotonic_ns(),cfg['max_joint_age_ms'])
        atomic_json(folder/'camera_control.json',dict(mode='record'))
        last_frame=-1
        for segment,obj in enumerate(plans):
            role=obj['id']
            actual=port.read()
            if max(abs(a-b) for a,b in zip(actual,home))>2.5 or math.dist(kin.wrist_fk(actual),kin.wrist_fk(home))>.5:
                raise RuntimeError('대기 중 시작 자세가 바뀌었습니다. 자동으로 이동하지 않습니다.')
            controller=Controller(port,obj['stages'],log,role)
            log.write('segment_start',segment_id=segment,object_id=role)
            previous_frame_ns=None
            # Do not reuse a frame captured before this segment was requested.
            not_before=time.monotonic_ns()
            while not controller.finished:
                def next_frame():
                    if not (folder/'latest_frame.json').exists():return None
                    f=read_json(folder/'latest_frame.json')
                    return f if f['frame_index']>last_frame and f['host_arrival_ns']>=not_before else None
                frame=wait_until(next_frame,guard,1.,'RGB 갱신이 1초 이상 끊겼습니다.')
                last_frame=frame['frame_index'];t=frame['host_arrival_ns'];now=time.monotonic_ns()
                if now-t>cfg['max_step_gap_ms']*1e6:raise RuntimeError('RGB 처리 지연 초과')
                if previous_frame_ns is not None and t-previous_frame_ns>cfg['max_step_gap_ms']*1e6:
                    raise RuntimeError('10Hz 동작 표본 간격이 200ms를 넘었습니다.')
                joint=sampler.before(t,cfg['max_joint_age_ms']);prev_grip=port.gripper
                result=controller.tick(joint['qpos_deg'],now/1e9)
                step=build_step(frame,joint,prev_grip,result,role,segment,time.monotonic_ns())
                steps.write('step',**step);previous_frame_ns=t
            log.write('segment_end',segment_id=segment,object_id=role)
            next_label=LABELS[plans[segment+1]['id']] if segment+1<len(plans) else '에피소드 저장'
            response=ask(f'{LABELS[role]} 동작·복귀 완료. 실제 성공이면 Y Enter → {next_label}; 실패/중단 N > ',guard).lower()
            success='user_confirmed' if response=='y' and not args.empty_run else ('empty_run' if args.empty_run and response=='y' else 'failed_or_unconfirmed')
            meta['objects'].append(dict(id=role,segment_id=segment,success=success))
            log.write('user_result',object_id=role,success=success)
            atomic_json(folder/'metadata.json',meta)
            if response!='y':raise RuntimeError('사용자가 실패/중단을 선택했습니다.')
        meta['status']='complete';meta['success']='empty_run' if args.empty_run else 'user_confirmed'
    except BaseException as exc:
        failure=exc;meta['status']='aborted';meta['success']='unverified';meta['error']=repr(exc)
        log.write('error',message=repr(exc),traceback=traceback.format_exc())
        print('중단:',exc,flush=True)
    finally:
        if port:
            try:port.stop()
            except Exception as exc:meta['stop_error']=repr(exc)
        if sampler:
            try:sampler.close()
            except Exception as exc:meta['sampler_close_error']=repr(exc)
        if robot:
            try:robot.dispose()
            except Exception as exc:meta['dispose_error']=repr(exc)
        atomic_json(folder/'camera_control.json',dict(mode='stop'))
        if process:
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate();meta['camera_forced_stop']=True
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
        worker_log.close();steps.close();log.close()
        atomic_json(folder/'metadata.json',meta)
    print('원본 기록:',folder,flush=True)
    if (folder/'frames.jsonl').exists() and (folder/'frames.jsonl').stat().st_size:
        try:
            print('로봇 연결 종료. 저장된 원본 영상에만 YOLO bbox/중심점 후처리를 합니다.',flush=True)
            subprocess.run([str(py),str(ROOT/'annotate.py'),str(folder)],env=vision_env(),check=True)
        except (Exception,KeyboardInterrupt) as exc:print('후처리 미완료:',exc,'\n나중에 annotate 명령으로 재실행할 수 있습니다.')
        from alignment import write_frame_alignment
        alignment_error=None
        try:
            alignment=write_frame_alignment(folder)
            print('프레임별 상태 연결:',alignment,flush=True)
        except Exception as exc:
            alignment_error=repr(exc);print('프레임별 상태 연결 실패:',exc,flush=True)
        from review import audit,export_episode
        report=audit(folder);atomic_json(folder/'audit.json',report)
        if alignment_error:
            report['errors'].append('frame alignment failed: '+alignment_error)
            report['data_checks_passed']=False
            atomic_json(folder/'audit.json',report)
        print('데이터 검사:', '통과' if report['data_checks_passed'] else '미통과',flush=True)
        for error in report['errors'][:12]:print(' -',error)
        if report['data_checks_passed']:print('학습 표본:',export_episode(folder))
    if failure:raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('new');a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('set-point');a.add_argument('--setup',type=Path,required=True);a.add_argument('id',choices=ROLES);a.add_argument('xyz',type=float,nargs=3)
    a=sub.add_parser('points');a.add_argument('--setup',type=Path,required=True)
    a=sub.add_parser('check');a.add_argument('--setup',type=Path,required=True)
    a=sub.add_parser('run');a.add_argument('--setup',type=Path,required=True);a.add_argument('--execute',action='store_true');a.add_argument('--empty-run',action='store_true')
    a.add_argument('--port',default='/dev/ttyACM0');a.add_argument('--out',type=Path,default=Path.home()/'Raccoonbot_Cafe_Clean/raccoon_dataset/tray4_raw')
    a=sub.add_parser('collect',help='매 에피소드마다 새 파일에 네 파지 좌표를 터미널로 입력')
    a.add_argument('--execute',action='store_true');a.add_argument('--empty-run',action='store_true')
    a.add_argument('--port',default='/dev/ttyACM0');a.add_argument('--out',type=Path,default=Path.home()/'Raccoonbot_Cafe_Clean/raccoon_dataset/tray4_raw')
    a=sub.add_parser('annotate');a.add_argument('episode',type=Path)
    a=sub.add_parser('audit');a.add_argument('episode',type=Path);a.add_argument('--export',action='store_true')
    sub.add_parser('self-test')
    a=p.parse_args()
    try:
        if a.cmd=='new':
            if a.out.exists():raise FileExistsError('기존 설정이 있습니다. 새 파일 이름을 사용하세요: '+str(a.out))
            atomic_json(a.out,default_config());print('설정 생성:',a.out.resolve());print('네 파지 위치만 입력하세요. 고정 목적지 여섯 곳은 제공한 최신 좌표입니다.')
        elif a.cmd in ('set-point','points'):
            cfg=read_json(a.setup);validate_config(cfg,False)
            roles=ROLES if a.cmd=='points' else [a.id]
            for r in roles:
                xyz=a.xyz if a.cmd=='set-point' else [float(v) for v in input(f'{LABELS[r]} 손목 x y z (cm, 공백 구분): ').replace(',',' ').split()]
                if not finite_point(xyz):raise ValueError('유한한 XYZ 세 값 필요')
                from planner import plate
                plate.ik(xyz,-90 if r in ('plate_1','tissue_1') else 0)
                cfg['pickups'][r]=xyz
            atomic_json(a.setup,cfg);print('파지 위치 저장:',a.setup.resolve(),' / 로봇 연결 없음')
        elif a.cmd=='check':
            import itertools
            cfg=read_json(a.setup);validate_config(cfg)
            count=0
            for order in itertools.permutations(ROLES):plan_episode(cfg,list(order),[0.,-10.,-140.,60.]);count+=1
            print(f'네 물체 {count}가지 순서: 기구학/단계 검사 통과. 예시 시작 자세 기준; 실제 충돌/파지는 미검증.')
            for name,xyz in cfg['destinations'].items():print(name,xyz)
            print('SDK/카메라 연결 없음. 실행 시 실제 시작 관절값으로 다시 검사합니다.')
        elif a.cmd=='run':run(a)
        elif a.cmd=='collect':
            cfg=default_config()
            print('컵 두 개는 정면(pitch≈0°), 접시·휴지는 아래(pitch≈-90°)를 향한 파지 자세의 손목 XYZ(cm)를 입력하세요.')
            for role in ROLES:
                point=[float(v) for v in input(f'{LABELS[role]} x y z: ').replace(',',' ').split()]
                if not finite_point(point):raise ValueError('XYZ 숫자 세 개 필요')
                cfg['pickups'][role]=point
            validate_config(cfg)
            a.setup=a.out.expanduser().parent/'tray4_setups'/('setup_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
            atomic_json(a.setup,cfg);print('이번 배치 설정:',a.setup)
            run(a)
        elif a.cmd=='annotate':
            cfg=read_json(a.episode/'config.json')
            subprocess.run([str(Path(cfg['vision_python']).expanduser()),str(ROOT/'annotate.py'),str(a.episode)],env=vision_env(),check=True)
        elif a.cmd=='audit':
            from review import audit,export_episode
            report=audit(a.episode);atomic_json(a.episode/'audit.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
            if a.export:print(export_episode(a.episode))
        elif a.cmd=='self-test':
            subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],cwd=ROOT,check=True)
    except (ValueError,FileNotFoundError,FileExistsError,KeyError) as exc:p.exit(2,str(exc)+'\n')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Collect one verified object segment as a standalone ACT episode."""
import argparse, json, math, os, select, shutil, subprocess, sys, time, traceback
from datetime import datetime
from pathlib import Path
from importlib.metadata import version
import numpy as np
from core import VERSION, ROLES, LABELS, SCHEMA, read_json, atomic_json, hash_data, JsonLog, compatible_selections, validate_config
from planner import kin, object_plan
from control import NativePort, Sampler, Controller, build_step

ROOT=Path(__file__).resolve().parent
def vision_env():
    env=os.environ.copy();libs=list((Path.home()/'yolo_ws/.venv/lib').glob('python*/site-packages/nvidia/cudnn/lib'))
    if libs:env['LD_LIBRARY_PATH']=str(libs[0])+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env
def ask(prompt,guard):
    print(prompt,end='',flush=True)
    while True:
        guard()
        if select.select([sys.stdin],[],[],.1)[0]:
            line=sys.stdin.readline()
            if line=='':raise EOFError('터미널 입력 종료')
            return line.strip()
def wait_until(pred,guard,timeout,label):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        guard();value=pred()
        if value:return value
        time.sleep(.02)
    raise TimeoutError(label)
def export_partial(folder,horizon=10):
    folder=Path(folder);rows=[]
    for line in open(folder/'steps.jsonl',encoding='utf-8'):
        x=json.loads(line)
        if x.get('event')=='step' and x.get('train_valid'):rows.append(x)
    if not rows:raise ValueError('부분 에피소드에 유효한 step이 없습니다.')
    n=len(rows);actions=np.empty((n,horizon,8),np.float32);pads=np.ones((n,horizon),bool)
    for i,row in enumerate(rows):
        for j in range(horizon):
            k=min(i+j,n-1);actions[i,j]=rows[k]['action'];pads[i,j]=i+j>=n
    out=folder/'training';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'episode.npz',state=np.asarray([r['observation_state'] for r in rows],np.float32),action=np.asarray([r['action'] for r in rows],np.float32),image=np.asarray([r['image'] for r in rows]),segment_id=np.zeros(n,np.int32),object_id=np.asarray([r['object_id'] for r in rows]),frame_index=np.asarray([r['frame_index'] for r in rows],np.int64))
    np.savez_compressed(out/f'act_h{horizon}.npz',image=np.asarray([r['image'] for r in rows]),observation_state=np.asarray([r['observation_state'] for r in rows],np.float32),action=actions,action_is_pad=pads,segment_id=np.zeros(n,np.int32),object_id=np.asarray([r['object_id'] for r in rows]),frame_index=np.asarray([r['frame_index'] for r in rows],np.int64))
    atomic_json(out/'schema.json',dict(format='raccoon_partial_act_v1',episode=folder.name,object_id=rows[0]['object_id'],observation_state_dim=5,action_dim=8,action_horizon=horizon,split_rule='split parent partial episodes, never frames'))
    return out/'episode.npz',len(rows)
def run(args):
    cfg=read_json(args.setup);validate_config(cfg)
    if args.object not in ROLES:raise ValueError('object must be one of '+','.join(ROLES))
    if not args.execute:print('부분 에피소드 기구학 검사만 하려면 --execute 없이 실행하세요.');object_plan(cfg,args.object,[0.,-10.,-140.,60.],[0.,-10.,-140.,60.]);return
    if version('roboid')!=kin.SDK_VERSION:raise ValueError('roboid==1.7.3 환경이 필요합니다.')
    py=Path(cfg['vision_python']).expanduser();stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder=args.out.expanduser()/stamp;folder.mkdir(parents=True)
    atomic_json(folder/'config.json',cfg);atomic_json(folder/'schema.json',SCHEMA);atomic_json(folder/'camera_control.json',dict(mode='select'))
    meta=dict(version=VERSION,status='starting',success='unverified',episode_id=stamp,object_count=1,objects=[],target=args.object,config_hash=hash_data(cfg),simulated=False,sdk=kin.SDK_VERSION,instruction=cfg['instruction'])
    atomic_json(folder/'metadata.json',meta);log=JsonLog(folder/'events.jsonl');steps=JsonLog(folder/'steps.jsonl');worker_log=(folder/'camera_worker.log').open('x');process=robot=port=sampler=None;failure=None
    print('부분 에피소드 저장:',folder,flush=True)
    try:
        process=subprocess.Popen([str(py),str(ROOT/'camera_worker_partial.py'),'--episode',str(folder),'--target',args.object],env=vision_env(),stdout=worker_log,stderr=subprocess.STDOUT)
        def camera_guard():
            if (folder/'camera_error.json').exists():raise RuntimeError(read_json(folder/'camera_error.json')['error'])
            if process.poll() is not None:raise RuntimeError('카메라 프로세스 종료')
        def ready():
            if not (folder/'ready.json').exists():return None
            r=read_json(folder/'ready.json');return r if r.get('ready') and time.monotonic_ns()-r['host_ns']<2_000_000_000 else None
        print(f'{LABELS[args.object]}가 목표로 보이고, 해당 단계의 주변 배치를 준비하세요. 다른 물체가 함께 보여도 됩니다. Q: 취소',flush=True)
        r=wait_until(ready,camera_guard,120,'선택 물체 인식 실패')
        print('인식 완료:',LABELS[args.object],r['selection']['objects'][0]['distance_pixel'],'px',flush=True)
        reply=input('배치 확인 후 Enter: 시작 / Q: 취소 > ').strip().lower()
        if reply=='q':raise KeyboardInterrupt('시작 취소')
        shown=r['selection'];atomic_json(folder/'camera_control.json',dict(mode='freeze',selection=shown))
        r=wait_until(lambda: read_json(folder/'selection.json') if (folder/'selection.json').exists() else None,camera_guard,30,'배치 안정화 실패')
        meta['status']='recording';meta['selection']=r;atomic_json(folder/'metadata.json',meta)
        from roboid import Raccoon
        robot=Raccoon(port_name=args.port);wait_until(robot.is_connected,camera_guard,8,'로봇 연결 실패');port=NativePort(robot,cfg['speed'],log);home=port.read();stages=object_plan(cfg,args.object,home,home)
        atomic_json(folder/'plan.json',dict(home_qpos_deg=home,object_id=args.object,stages=stages,scope='partial native-mode plan'))
        meta['home_qpos_deg']=home;atomic_json(folder/'metadata.json',meta);sampler=Sampler(port,log,20);sampler.start();time.sleep(.2)
        # The partial camera worker remains frozen until the robot sampler is ready.
        # Switch it explicitly to raw-frame recording before waiting for frames.
        atomic_json(folder/'camera_control.json',dict(mode='record'))
        def guard():
            camera_guard()
            if sampler.error:raise RuntimeError(sampler.error)
        last_frame=-1;controller=Controller(port,stages,log,args.object);not_before=time.monotonic_ns()
        while not controller.finished:
            frame=wait_until(lambda: (read_json(folder/'latest_frame.json') if (folder/'latest_frame.json').exists() else None),guard,1,'RGB 갱신 중단')
            if frame['frame_index']<=last_frame or frame['host_arrival_ns']<not_before:continue
            last_frame=frame['frame_index'];joint=sampler.before(frame['host_arrival_ns'],cfg['max_joint_age_ms']);prev=port.gripper;result=controller.tick(joint['qpos_deg'],time.monotonic());steps.write('step',**build_step(frame,joint,prev,result,args.object,0,time.monotonic_ns()))
        reply=input(f'{LABELS[args.object]} 성공이면 Y, 실패면 N > ').strip().lower()
        if reply!='y':raise RuntimeError('사용자가 실패/중단을 선택했습니다.')
        meta['objects']=[dict(id=args.object,segment_id=0,success='user_confirmed')];meta['status']='complete';meta['success']='user_confirmed';atomic_json(folder/'metadata.json',meta)
    except BaseException as exc:
        failure=exc;meta['status']='aborted';meta['error']=repr(exc);atomic_json(folder/'metadata.json',meta);log.write('error',message=repr(exc),traceback=traceback.format_exc());print('중단:',exc,flush=True)
    finally:
        if port:
            try:port.stop()
            except Exception:pass
        if sampler:
            try:sampler.close()
            except Exception:pass
        if robot:
            try:robot.dispose()
            except Exception:pass
        atomic_json(folder/'camera_control.json',dict(mode='stop'))
        if process:
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.terminate()
        worker_log.close();steps.close();log.close()
    if failure is None and meta.get('status')=='complete' and (folder/'steps.jsonl').exists():
        try:
            path,count=export_partial(folder);print('부분 ACT 표본:',path,'steps=',count,flush=True)
        except Exception as exc:print('부분 표본 생성 실패:',exc,flush=True)
    elif failure is not None:
        print('실패/중단 에피소드이므로 학습 표본을 저장하지 않습니다.',flush=True)
        shutil.rmtree(folder,ignore_errors=True)
    if failure:raise SystemExit(1)
    print('부분 에피소드 완료:',folder,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',choices=('run',));ap.add_argument('--object',choices=ROLES,required=True);ap.add_argument('--setup',type=Path,required=True);ap.add_argument('--port',default='/dev/ttyACM0');ap.add_argument('--execute',action='store_true');ap.add_argument('--out',type=Path,default=Path.home()/'Raccoonbot_Cafe_Clean/raccoon_dataset/tray4_raw_partial');args=ap.parse_args();run(args)

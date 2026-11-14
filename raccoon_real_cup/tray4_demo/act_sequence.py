"""One-start, object-routed ACT runner. No taught trajectories are replayed.

Completion is a heuristic: observed close/release, return home, and target
absent from a continuously detected tray. It does not prove correct disposal.
"""
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
from core import atomic_json, read_json, JsonLog, ROLES, KINDS

ROOT = Path(__file__).absolute().parents[2]


class Sequence:
    def __init__(self, home, roles=ROLES):
        self.home = np.asarray(home)
        self.roles = tuple(roles)
        self.done = []
        self.target = None
        self.closed = False
        self.released = False
        self.away = False
        self.stable = 0

    def choose(self, visible):
        # Plate then tissue; cups ordered by distance to front tray edge.
        remaining = [r for r in self.roles if r not in self.done]
        if any(r not in visible for r in remaining):
            raise RuntimeError('남은 물체를 모두 확실하게 인식하지 못했습니다.')
        for r in ('plate_1', 'tissue_1'):
            if r in remaining:
                self.target = r; break
        else:
            self.target = min(remaining, key=lambda r: visible[r]['distance']) if remaining else None
        self.closed = self.released = self.away = False
        self.stable = 0
        return self.target

    def observe(self, q, gripper, visible, tray_ok):
        self.away |= bool(np.max(np.abs(np.asarray(q)-self.home)) > 8)
        self.closed |= gripper == 1
        self.released |= self.closed and gripper == 0
        candidate = (tray_ok and self.target not in visible and self.away and
                     self.released and gripper == 0 and np.max(np.abs(np.asarray(q)-self.home)) < 4)
        self.stable = self.stable+1 if candidate else 0
        if self.stable >= 10:
            self.done.append(self.target)
            self.target = None
            return True
        return False


def validate_policies(manifest, roles=ROLES):
    if not set(roles) <= set(manifest):
        raise ValueError('정책 목록에 네 물체 ID가 모두 필요합니다.')
    for role, location in manifest.items():
        run = Path(location).expanduser().absolute()
        report = read_json(run/'training_report.json')
        if not (run/'act_tray4_policy/model.safetensors').is_file():
            raise ValueError(f'{role}: 모델 파일 없음')
        # Do not silently route the same mixed policy as four object policies.
        paths = report['train_episode_paths']+report['validation_episode_paths']
        if not paths: raise ValueError('빈 학습 목록')
        for p in paths:
            schema = read_json(Path(p)/'training/schema.json')
            if schema.get('object_id') != role:
                raise ValueError(f'{role}: 해당 물체 전용으로 학습된 정책이 아닙니다: {p}')


def serve(args):
    import cv2
    if args.worker == 'vision':
        import torch
        torch.set_num_threads(4)
        from vision import load_model, detect
        cfg = read_json(args.setup); model = load_model(cfg)
        # This worker's YOLO environment cannot load its GPU cuDNN libraries.
        model.to('cpu')
        model.overrides['device'] = 'cpu'
        detect(model, np.zeros((480,640,3), dtype=np.uint8), cfg)
    else:
        import torch
        from PIL import Image
        from act_train import ACTPolicy
        torch.set_num_threads(4)
        phase_specs = {r: read_json(Path(p)/'training_report.json').get('phase_spec') for r,p in read_json(args.policies).items()}
        policies = {r: ACTPolicy.from_pretrained(str(Path(p)/'act_tray4_policy')).to('cpu').eval()
                    for r,p in read_json(args.policies).items()}
    atomic_json(args.ipc/'ready.json', {})
    last = -1
    while True:
        if not (args.ipc/'request.json').exists(): time.sleep(.01); continue
        request = read_json(args.ipc/'request.json')
        if request['id'] == last: time.sleep(.005); continue
        last = request['id']
        if args.worker == 'vision':
            detections = detect(model, cv2.imread(request['image']), cfg)
            trays = [d for d in detections if d['kind']=='tray' and d['confidence']>=.5]
            visible = {}
            if trays:
                tray = max(trays, key=lambda d:d['box'][3])
                mask = tray['mask']; ys,xs = np.nonzero(mask)
                hull = cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32))
                area = np.zeros_like(mask,np.uint8); cv2.fillConvexPoly(area,hull,1)
                ref = [(xs.min()+xs.max())/2,ys.max()]
                for role,kind in KINDS.items():
                    candidates=[]
                    for d in detections:
                        if d['kind']!=kind: continue
                        yy=np.nonzero(d['mask'])[0]; base=d['mask'].copy()
                        base[:yy.max()-max(1,int((yy.max()-yy.min()+1)*.3))]=False
                        if base.sum() and (base&(area>0)).sum()/base.sum()>=.6: candidates.append(d)
                    if len(candidates)==1:
                        d=candidates[0]; visible[role]={'box':d['box'],'distance':float(np.linalg.norm(np.array(d['center'])-ref))}
            result={'visible':visible,'tray_ok':bool(trays)}
        else:
            with Image.open(request['image']) as source:
                image=np.asarray(source.convert('RGB').resize((96,96),Image.Resampling.BILINEAR),np.float32)/255
            state = request['state']
            if phase_specs[request['target']]:
                from act_phase import feature
                state = state + feature(request['phase'], phase_specs[request['target']]).tolist()
            batch={'observation.images.front':torch.from_numpy(image.transpose(2,0,1).copy()).unsqueeze(0),
                   'observation.state':torch.tensor([state],dtype=torch.float32)}
            with torch.inference_mode():
                chunk=policies[request['target']].predict_action_chunk(batch)[0].cpu().numpy()
            if not np.isfinite(chunk).all(): raise ValueError('비정상 예측')
            result={'actions':chunk.tolist()}
        atomic_json(args.ipc/'response.json', {'id':last,**result})


class RPC:
    def __init__(self,args,kind,folder,python):
        self.folder=folder; folder.mkdir(); self.index=0
        self.kind=kind; self.log_path=args.out/(kind+'.log')
        self.log=(args.out/(kind+'.log')).open('w')
        self.process=subprocess.Popen([str(python),str(Path(__file__).absolute()),'--worker',kind,
            '--ipc',str(folder),'--setup',str(args.setup),'--policies',str(args.policies)],stdout=self.log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+90
        while not (folder/'ready.json').exists():
            if self.process.poll() is not None or time.monotonic()>deadline:
                self.close()
                raise RuntimeError(f'{kind} 로딩 실패: {args.out}')
            time.sleep(.1)
    def call(self,payload):
        self.index+=1
        atomic_json(self.folder/'request.json',{'id':self.index,**payload})
        deadline=time.monotonic()+2
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f'{self.kind} 프로세스 종료 (코드 {self.process.returncode}): {self.log_path}')
            if (self.folder/'response.json').exists():
                result=read_json(self.folder/'response.json')
                if result['id']==self.index:return result
            if time.monotonic()>deadline:raise RuntimeError('작업 프로세스 응답 시간 초과')
            time.sleep(.005)
    def close(self):
        self.process.terminate(); self.process.wait(timeout=5); self.log.close()


def run(args):
    # Validate before camera or hardware access.
    validate_policies(read_json(args.policies), [args.target] if args.target else ROLES)
    from act_phase import PhaseProgress
    specs={r:read_json(Path(p)/'training_report.json').get('phase_spec') for r,p in read_json(args.policies).items()}
    if any(not spec for spec in specs.values()):
        raise ValueError('단계 조건부 정책이 필요합니다. 기존 5차원 모델은 실행할 수 없습니다.')
    progress=None
    import cv2
    import pyrealsense2 as rs
    from roboid import Raccoon
    from control import NativePort
    from planner import kin
    import threading
    args.out.mkdir(parents=True,exist_ok=False)
    workers=[]; camera=robot=port=timer=None
    log=JsonLog(args.out/'events.jsonl'); stop=threading.Event(); gate=threading.Lock()
    def halt():
        with gate:
            stop.set()
            if port and args.execute:port.stop()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for kind,python in [('vision',Path(read_json(args.setup)['vision_python']).expanduser()),('policy',ROOT/'.venv/bin/python')]:
                workers.append(RPC(args,kind,Path(tmp)/kind,python))
            vision,policy=workers
            robot=Raccoon(port_name=args.port)
            deadline=time.monotonic()+8
            while not robot.is_connected():
                if time.monotonic()>deadline: raise RuntimeError('로봇 연결 실패')
                time.sleep(.1)
            port=NativePort(robot,10,log); port.gripper=0
            cam=rs.pipeline(); config=rs.config(); cfg=read_json(args.setup)
            if cfg.get('camera_serial'):config.enable_device(cfg['camera_serial'])
            config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30); cam.start(config); camera=cam
            seq=Sequence(port.read(), [args.target] if args.target else ROLES); armed=False; stable=0; last_boxes=None; target_since=None
            print('집게가 열린 빈 시작 자세로 준비. 카메라 Enter 한 번 시작 / Q 중단',flush=True)
            index=0
            while not stop.is_set():
                frame=camera.wait_for_frames(1000).get_color_frame()
                if not frame:raise RuntimeError('카메라 없음')
                bgr=np.asanyarray(frame.get_data()).copy(); image=args.out/f'{index:06d}.png'
                if not cv2.imwrite(str(image),bgr):raise OSError('영상 저장 실패')
                seen=vision.call({'image':str(image.absolute())})
                q=port.read(); visible=seen['visible']
                if not armed:
                    boxes={r:visible[r]['box'] for r in visible}
                    same=(last_boxes is not None and set(seq.roles)<=set(boxes) and set(seq.roles)<=set(last_boxes)
                          and all(np.max(np.abs(np.array(boxes[r])-last_boxes[r]))<8 for r in seq.roles))
                    stable=stable+1 if same else 0; last_boxes=boxes
                    if stable>=5:seq.choose(visible)
                elif seq.observe(q,port.gripper,visible,seen['tray_ok'] and progress.finished):
                    log.write('completion_heuristic',target=seq.done[-1],note='removal/home/release; correct placement unverified')
                    if len(seq.done)==len(seq.roles):print('선택 대상 제거·복귀 조건 충족. 종료.');break
                    seq.choose(visible);target_since=time.monotonic();progress=PhaseProgress(specs[seq.target])
                view=bgr.copy()
                if seq.target in visible:
                    x1,y1,x2,y2=map(int,visible[seq.target]['box']);cv2.rectangle(view,(x1,y1),(x2,y2),(0,255,0),3)
                cv2.putText(view,f'{seq.target or "WAIT"} | {"RUN" if armed else "ENTER when ready"}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
                cv2.imshow('Object ACT - Enter start / Q stop',view);key=cv2.waitKey(1)&255
                if key in (ord('q'),27):break
                if not armed:
                    if key in (10,13) and stable>=5:
                        seq.home=np.asarray(q);armed=True;target_since=time.monotonic();progress=PhaseProgress(specs[seq.target])
                        timer=threading.Timer(args.max_seconds,halt);timer.start()
                    index+=1;continue
                if time.monotonic()-target_since>args.object_timeout:raise RuntimeError('대상 완료 시간 초과')
                if progress.finished:
                    index+=1;continue
                if progress.observe(q):
                    log.write('phase_advance',target=seq.target,index=progress.index)
                    print(seq.target, 'FINISHED' if progress.finished else progress.stage['name'],flush=True)
                    index+=1;continue
                result=policy.call({'phase':progress.stage['name'],'image':str(image.absolute()),'target':seq.target,'state':np.deg2rad(q).tolist()+[float(port.gripper)]})
                # Re-observe after each phase-conditioned action; do not cross phase boundaries.
                for action in result['actions'][:1]:
                    a=np.asarray(action);goal=np.rad2deg(a[:4]).tolist();kin.check_q(goal)
                    mode=('horizontal','vertical','wrist')[int(a[5:8].argmax())];grip=int(a[4]>=.5)
                    progress.accept(goal,mode,grip)
                    observed=port.read();active=[3] if mode=='wrist' else [0,1,2]
                    # A taught wrist rotation can span 90 degrees. Approach the validated
                    # predicted endpoint in bounded increments, never change its mode.
                    command_goal=list(observed)
                    for j in active:
                        command_goal[j]+=float(np.clip(goal[j]-observed[j],-5,5))
                    log.write('policy_step',phase=progress.stage['name'],target=seq.target,image=str(image),observed_deg=observed,command_goal_deg=command_goal,previous_gripper=port.gripper,action=action)
                    gripper_changed = grip != port.gripper
                    with gate:
                        if stop.is_set():break
                        if args.execute:
                            port.apply(command_goal,grip,mode,observed)
                            seq.closed |= grip == 1
                            seq.released |= seq.closed and grip == 0
                    if stop.wait(.5 if gripper_changed else .1):break
                index+=1
    finally:
        halt()
        if timer:timer.cancel();timer.join()
        for w in workers:w.close()
        if camera:camera.stop()
        if robot:robot.dispose()
        log.close();cv2.destroyAllWindows()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policies',type=Path,default=ROOT/'raccoon_dataset/tray4_phase_act_v1/policies.json')
    p.add_argument('--setup',type=Path,default=ROOT/'raccoon_dataset/tray4_setups/setup_20261108_onepour_reteach.json')
    p.add_argument('--port',default='/dev/ttyACM2')
    p.add_argument('--target',choices=ROLES,help='시험할 물체 하나만 지정; 생략하면 전체 순서 실행')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--check',action='store_true')
    p.add_argument('--max-seconds',type=float,default=180)
    p.add_argument('--object-timeout',type=float,default=60)
    p.add_argument('--out',type=Path,default=ROOT/'raccoon_dataset/act_sequence_trials'/time.strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--worker',choices=['vision','policy'],help=argparse.SUPPRESS)
    p.add_argument('--ipc',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if not 0<a.max_seconds<=300 or not 0<a.object_timeout<=120:p.error('전체 300초 / 대상 120초 이내')
    if a.worker:serve(a)
    elif a.check:validate_policies(read_json(a.policies), [a.target] if a.target else ROLES);print('정책 구성 검사 통과')
    else:run(a)

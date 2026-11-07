"""Feedback-gated native-mode execution. No perception-based motion corrections."""
import collections
import copy
import math
import threading
import time
from core import MODES, encode_action
from planner import kin

class NativePort:
    """Nonblocking equivalent of roboid 1.7.3 angle/gripper writes.

    The supplied SDK's open/close methods wait 500ms and lock_horz/vert may
    wait for a joint event. Here the public write API sends the same device
    commands; Controller implements those settle waits without blocking capture.
    Only this object writes motors; sampler uses the same short-lived lock.
    """
    def __init__(self,robot,speed,log):
        self.robot=robot;self.speed=speed;self.log=log;self.lock=threading.RLock()
        self.mode=None;self.goal=None;self.gripper=None;self.failed=False;self.command_index=-1
    def read(self):
        with self.lock:
            if not self.robot.is_connected():raise RuntimeError('로봇 연결이 끊겼습니다. 자동 재개하지 않습니다.')
            return kin.check_q(list(self.robot.encoder(-1)))
    def apply(self,goal,gripper,mode,observed):
        kin.check_q(goal);kin.check_q(observed)
        if self.failed:raise RuntimeError('정지한 연결에는 다시 명령하지 않습니다.')
        if mode not in MODES or gripper not in (0,1):raise ValueError('명령 모드 오류')
        changed_mode=self.mode!=mode
        mask=[False,False,False,True] if mode=='wrist' else [True,True,True,False]
        changed_goal=self.goal is None or any(abs(goal[j]-self.goal[j])>1e-8 for j in range(4) if mask[j])
        if changed_mode and mode!='wrist':
            if abs(kin.pitch(observed)-(0 if mode=='horizontal' else -90))>4:
                raise RuntimeError('자세 고정을 바꾸기 전 pitch 오차가 4도를 넘습니다.')
        writes=[]
        def write(*args):
            result=self.robot.write(*args)
            if not result:raise RuntimeError('SDK가 장치 명령을 거부했습니다: '+repr(args))
            if args[0]==self.robot.WRITE_EXT and result!=8:raise RuntimeError('집게 명령 8바이트 기록 실패')
            writes.append(list(args))
        start=time.monotonic_ns()
        try:
            with self.lock:
                if not self.robot.is_connected():raise RuntimeError('명령 전 로봇 연결 끊김')
                r=self.robot
                if changed_mode or changed_goal:
                    mode_value={'horizontal':r.JOINT_MODE_ANGLE_HORZ,
                                'vertical':r.JOINT_MODE_ANGLE_VERT,'wrist':r.JOINT_MODE_ANGLE}[mode]
                    write(r.JOINT_MODE,mode_value)
                    write(r.JOINT_SPEED,self.speed)
                    if changed_mode and mode!='wrist':
                        # Mirrors the SDK lock's q4=encoder value initialization.
                        write(r.JOINT_ANGLE,3,float(observed[3]))
                    for j,used in enumerate(mask):
                        if used:write(r.JOINT_ANGLE,j,float(goal[j]))
                if gripper!=self.gripper:
                    write(r.WRITE_EXT,[0x21,0x00,int(gripper),0,0,0,0,0])
        except BaseException as exc:
            self.failed=True
            self.log.write('command_error',call_start_ns=start,call_end_ns=time.monotonic_ns(),
                           completed_writes=writes,error=repr(exc))
            try:self.stop()
            except Exception:pass
            raise
        end=time.monotonic_ns()
        self.mode=mode;self.goal=list(goal);self.gripper=gripper
        if writes:
            self.command_index+=1
            self.log.write('command',command_index=self.command_index,call_start_ns=start,call_end_ns=end,goal_deg=list(goal),
                           mode=mode,gripper_command=gripper,joint_command_mask=list(map(int,mask)),
                           speed=self.speed,sdk_writes=writes)
        return bool(writes)
    def stop(self):
        self.failed=True
        with self.lock:
            try:
                # Explicit velocity mode + zero velocity; preserve gripper command.
                r=self.robot;r.write(r.JOINT_MODE,r.JOINT_MODE_VELOCITY)
                for j in range(4):r.write(r.JOINT_VELOCITY,j,0)
            finally:self.log.write('stop_requested')

class Sampler:
    def __init__(self,port,log,hz=20):
        self.port=port;self.log=log;self.hz=hz;self.samples=collections.deque(maxlen=400)
        self.lock=threading.Lock();self.stop_event=threading.Event();self.error=None
        self.thread=threading.Thread(target=self._run,name='joint-20hz',daemon=True)
    def start(self):self.thread.start()
    def _run(self):
        deadline=time.monotonic();period=1/self.hz
        try:
            while not self.stop_event.is_set():
                if self.stop_event.wait(max(0,deadline-time.monotonic())):break
                a=time.monotonic_ns();q=self.port.read();b=time.monotonic_ns()
                row=dict(read_start_ns=a,read_end_ns=b,qpos_deg=q,source='SDK encoder cache read; sensor timestamp unavailable')
                with self.lock:self.samples.append(row)
                self.log.write('joint_observation',**row)
                deadline+=period
                if time.monotonic()>deadline:
                    n=int((time.monotonic()-deadline)/period)+1;deadline+=n*period
                    self.log.write('joint_deadline_missed',count=n)
        except BaseException as exc:
            self.error=repr(exc);self.log.write('joint_error',message=self.error)
    def before(self,frame_ns,max_age_ms):
        if self.error:raise RuntimeError(self.error)
        with self.lock:
            item=next((x for x in reversed(self.samples) if x['read_end_ns']<=frame_ns),None)
        if item is None or (frame_ns-item['read_end_ns'])/1e6>max_age_ms:
            raise RuntimeError('현재 영상 이전의 신선한 관절 표본이 없습니다.')
        return copy.deepcopy(item)
    def close(self):
        self.stop_event.set();self.thread.join(timeout=2)
        if self.thread.is_alive():raise RuntimeError('관절 기록 스레드 종료 실패')

class Controller:
    def __init__(self,port,stages,log,role):
        self.port=port;self.stages=stages;self.log=log;self.role=role
        self.index=-1;self.stage=None;self.entered=None;self.deadline=None
        self.history=[];self.goal=None;self.mode=None;self.gripper=0 if port.gripper is None else port.gripper
        self.finished=False
    def _enter(self,q,now):
        self.index+=1;self.history=[];self.entered=now
        if self.index==len(self.stages):
            self.finished=True;self.log.write('segment_motion_complete',object_id=self.role);return
        self.stage=self.stages[self.index];s=self.stage
        if s['kind'] in ('move','rotate'):
            self.goal=list(s['target']);self.mode=s['mode']
            self.deadline=now+min(60.,max(12.,max(abs(a-b) for a,b in zip(q,self.goal))*.6))
        elif s['kind'] in ('open','close'):self.gripper=int(s['kind']=='close')
        elif s['kind']!='hold':raise ValueError('알 수 없는 단계 '+str(s))
        self.log.write('phase',object_id=self.role,phase=s['name'],kind=s['kind'],index=self.index)
        print(self.role, s['name'],flush=True)
        if self.goal is None:raise ValueError('첫 단계는 이동이어야 합니다.')
    def _arrived(self,q):
        s=self.stage;self.history.append(list(q));self.history=self.history[-5:]
        pos=math.dist(kin.wrist_fk(q),kin.wrist_fk(self.goal))
        pe=abs(kin.pitch(q)-s['pitch'])
        if s['kind']=='move' and pe>6:raise RuntimeError('이동 중 pitch 오차가 6도를 넘습니다.')
        if len(self.history)<5:return False
        spread=max(max(r[j] for r in self.history)-min(r[j] for r in self.history) for j in range(4))
        joint=max(abs(a-b) for a,b in zip(q,self.goal))
        return joint<=2 and pos<=s['position_tolerance_cm'] and pe<=s['pitch_tolerance_deg'] and spread<=.5
    def tick(self,q,now):
        if self.finished:raise RuntimeError('이미 끝난 구간')
        if self.stage is None:self._enter(q,now)
        else:
            s=self.stage
            arrived=(self._arrived(q) if s['kind'] in ('move','rotate') else now-self.entered>=s['seconds'])
            if s['kind'] in ('move','rotate') and now>self.deadline and not arrived:
                raise RuntimeError('목표 도착 시간 초과: '+s['name'])
            if arrived:
                self.log.write('phase_done',object_id=self.role,phase=s['name'],qpos_deg=list(q))
                self._enter(q,now)
        if not self.finished:self.port.apply(self.goal,self.gripper,self.mode,q)
        return dict(action=encode_action(self.goal,self.gripper,self.mode),
                    mode=self.mode,gripper_command=self.gripper,
                    command_index=self.port.command_index,phase=self.stage['name'],finished=self.finished)

def build_step(frame,joint,previous_gripper,result,role,segment,decision_ns):
    known=previous_gripper in (0,1)
    return dict(frame_index=frame['frame_index'],image=frame['file'],frame_host_ns=frame['host_arrival_ns'],
                joint_read_ns=joint['read_end_ns'],decision_ns=decision_ns,
                observation_state=[math.radians(x) for x in joint['qpos_deg']]+[float(previous_gripper or 0)],
                observation_state_mask=[1,1,1,1,int(known)],gripper_measured=None,
                action=result['action'],phase=result['phase'],object_id=role,segment_id=segment,
                mode=result['mode'],command_index=result['command_index'],train_valid=known)

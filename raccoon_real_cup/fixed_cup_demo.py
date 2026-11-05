"""Fixed-coordinate cup pick -> pour/shake -> place. roboid 1.7.3.
Default: offline plan only. --execute enables motion; --empty-run skips closing.
Coordinates: wrist, cm, robot base frame. No camera or collision detector.
"""
import argparse
import json
import math
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from cup_pick_real import (Bench, SDK_VERSION, check_q, wrist_fk, horizontal_ik,
                           pitch, execution_plan)
from smooth_cup_motion import execute_phase


def validate(cfg):
    for key in ('grasp_wrist_cm','pour_wrist_cm','place_wrist_cm'):
        p=cfg[key]
        if not isinstance(p,list) or len(p)!=3 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in p):
            raise ValueError(f'{key}: finite x,y,z required')
        horizontal_ik(p)
    for key in ('clearance_z_cm','approach_cm','pour_pitch_deg','speed','shake_cycles','hold_seconds'):
        if not isinstance(cfg[key],(int,float)) or not math.isfinite(cfg[key]):
            raise ValueError(f'{key}: invalid number')
    if not 1<=cfg['speed']<=40 or int(cfg['speed'])!=cfg['speed']:raise ValueError('speed 1..40 integer')
    if not .5<=cfg['approach_cm']<=3:raise ValueError('approach_cm 0.5..3')
    if not -90<=cfg['pour_pitch_deg']<=-30:raise ValueError('pour_pitch_deg -90..-30')
    if not 1<=cfg['shake_cycles']<=3 or int(cfg['shake_cycles'])!=cfg['shake_cycles']:raise ValueError('shake_cycles 1..3 integer')
    if not .5<=cfg['hold_seconds']<=3:raise ValueError('hold_seconds .5..3')
    z=cfg['clearance_z_cm']
    if z<max(cfg['grasp_wrist_cm'][2],cfg['place_wrist_cm'][2])+2 or z>20:
        raise ValueError('clearance must be >= grasp/place +2cm, and <=20cm')


def radial_back(p, distance):
    r=math.hypot(*p[:2])
    if r<=distance+1:raise ValueError('approach too close to base')
    return [p[0]*(r-distance)/r,p[1]*(r-distance)/r,p[2]]


def plan(cfg,start):
    validate(cfg);check_q(start)
    if abs(pitch(start))>3:
        raise ValueError('시작 자세가 수평이 아닙니다. cup_pick_real.py의 horz로 먼저 확인하세요.')
    current=list(start); steps=[]
    def line(label,p):
        nonlocal current
        initial=wrist_fk(current)
        n=max(1,math.ceil(math.dist(initial,p)/.1))
        if n>600:raise ValueError('path exceeds bench scale')
        dense=[];previous=current
        for i in range(1,n+1):
            q=horizontal_ik([a+(b-a)*i/n for a,b in zip(initial,p)])
            if max(abs(a-b) for a,b in zip(previous,q))>5:
                raise ValueError(f'{label}: IK jump or singularity')
            dense.append(q);previous=q
        commands=execution_plan(current,dense)
        steps.append(dict(name=label,kind='move',targets=commands,wrist_cm=p))
        current=dense[-1]
    def rotate(label,degrees):
        nonlocal current
        goal=current.copy();goal[3]+=degrees-pitch(current);check_q(goal)
        n=max(1,math.ceil(abs(goal[3]-current[3])/5))
        commands=[]
        for i in range(1,n+1):
            q=current.copy();q[3]+=(goal[3]-current[3])*i/n;check_q(q);commands.append(q)
        steps.append(dict(name=label,kind='rotate',targets=commands,pitch_deg=degrees))
        current=goal
    def event(name,kind):steps.append(dict(name=name,kind=kind))
    g=cfg['grasp_wrist_cm']; dest=cfg['place_wrist_cm']; wet=cfg['pour_wrist_cm']
    z=cfg['clearance_z_cm'];pre=radial_back(g,cfg['approach_cm']);back=radial_back(dest,cfg['approach_cm'])
    here=wrist_fk(current)
    # Raise first; no sideways sweep at the initial low height.
    line('INITIAL_RAISE',[here[0],here[1],max(z,here[2])])
    event('OPEN','open')
    line('ABOVE_APPROACH',[pre[0],pre[1],z])
    line('LOWER_BEFORE_CUP',pre)
    line('FRONT_APPROACH',g)
    event('GRASP','close')
    line('LIFT_UPRIGHT',[g[0],g[1],z])
    line('ABOVE_POUR',[wet[0],wet[1],max(z,wet[2])])
    line('POUR_POSITION',wet)
    for i in range(int(cfg['shake_cycles'])):
        rotate(f'TILT_DOWN_{i+1}',cfg['pour_pitch_deg'])
        event(f'DRAIN_HOLD_{i+1}','hold')
        rotate(f'TILT_UP_{i+1}',0)
        event(f'UPRIGHT_HOLD_{i+1}','hold')
    line('LEAVE_POUR',[wet[0],wet[1],max(z,wet[2])])
    line('RIGHT_TRANSIT',[0.0,math.hypot(dest[0],dest[1]),z])
    line('ABOVE_PLACE',[dest[0],dest[1],z])
    line('LOWER_TO_PLACE',dest)
    event('RELEASE','open')
    line('RETREAT_FROM_CUP',back)
    line('FINISH_RAISED',[back[0],back[1],z])
    return steps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent
    parser.add_argument('--config',type=Path,default=root/'fixed_cup_config.json')
    parser.add_argument('--port',default='/dev/ttyACM0')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--empty-run',action='store_true',help='motion but keep gripper open; no cup')
    parser.add_argument('--step',action='store_true',help='Enter before each phase')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());validate(cfg)
    if not args.execute:
        steps=plan(cfg,[0,-10,-140,60])
        print('PLAN ONLY: no SDK connection. Example start [0,-10,-140,60].')
        for s in steps:
            print(s['name'],s['kind'],s.get('wrist_cm',s.get('pitch_deg','')))
        print('Wrist cm configuration:',args.config)
        print('Real start and all targets are checked again before execution.')
        return
    if version('roboid')!=SDK_VERSION:parser.error('Install roboid==1.7.3')
    from roboid import Raccoon
    robot=None;log=None
    try:
        robot=Raccoon(port_name=args.port)
        time.sleep(.5)
        b=Bench(robot,int(cfg['speed']),root/'cup_pose.json',2,2)
        start=b.read()
        steps=plan(cfg,start)  # complete preflight before open or motion
        logdir=root/'real_logs';logdir.mkdir(exist_ok=True)
        path=logdir/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.jsonl')
        log=path.open('w');t0=time.monotonic()
        def record(event,**kw):
            log.write(json.dumps(dict(time_s=time.monotonic()-t0,event=event,**kw))+ '\n');log.flush()
        record('start',config=cfg,measured_q=start,empty_run=args.empty_run)
        print('LOG:',path)
        print('실제 이동 시작. 빈 그리퍼 모드:',args.empty_run,'Ctrl+C: 정지 요청')
        for s in steps:
            print('PHASE:',s['name'],s.get('wrist_cm',s.get('pitch_deg','')),flush=True)
            if args.step:input('주변 공간을 확인하고 Enter: ')
            # If manually displaced during the pause, reject the next large jump.
            record('phase',name=s['name'])
            if s['kind'] in ('move','rotate'):
                execute_phase(robot, b, s['targets'], s, record)
                record('phase_done',measured_q=b.read())
            elif s['kind']=='open':robot.open_gripper()
            elif s['kind']=='close':
                if args.empty_run:print('EMPTY RUN: close 생략')
                else:robot.close_gripper()
                time.sleep(.5)
            elif s['kind']=='hold':time.sleep(cfg['hold_seconds'])
        record('sequence_complete',measured_q=b.read())
        print('동작 명령 완료. 실제 파지/비우기/배치 성공 여부는 별도 확인하세요.')
    except (KeyboardInterrupt,EOFError):print('\n중단. 컵을 잡았다면 지지 상태를 확인하세요.')
    except Exception as exc:
        print('STOP:',exc)
        if log:log.write(json.dumps(dict(event='error',message=str(exc)))+'\n');log.flush()
        raise SystemExit(1)
    finally:
        if robot:
            robot.stop();time.sleep(.2);robot.dispose()
        if log:log.close()

if __name__=='__main__':main()

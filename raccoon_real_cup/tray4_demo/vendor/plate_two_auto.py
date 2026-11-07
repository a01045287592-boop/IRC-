"""RaccoonBot two-plate test. roboid 1.7.3; wrist cm; no vision/collision sensing.
--init creates an editable configuration with unset physical heights.
Default reads configuration and checks an offline plan. --execute moves the robot.
"""
import argparse
import json
import math
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from cup_pick_real import Bench, SDK_VERSION, check_q, wrist_fk, pitch


def ik(wrist, desired_pitch):
    if len(wrist)!=3 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in wrist):
        raise ValueError('모든 손목 x, y, z를 cm 숫자로 설정하세요.')
    x,y,z=wrist;radius=math.hypot(x,y)
    if radius<1 or z<0:raise ValueError('손목 위치가 유효하지 않습니다.')
    h=z-8.25;cosine=(radius*radius+h*h-200)/200
    if not -1<=cosine<=1:raise ValueError(f'도달 불가능한 손목 좌표: {wrist}')
    c=-math.acos(cosine)
    b=math.atan2(-radius,h)-math.atan2(10*math.sin(c),10+10*math.cos(c))
    q=[math.degrees(math.atan2(-x,y)),math.degrees(b),math.degrees(c)]
    q.append(desired_pitch-90-q[1]-q[2])
    return check_q(q)


def number(value,label):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'{label}을(를) 실측 후 숫자로 입력하세요.')
    return float(value)


def validate(cfg):
    cups=cfg.get('plate_grasp_wrist_cm')
    if not isinstance(cups,list) or len(cups)!=2:raise ValueError('접시 파지 좌표가 정확히 2개 필요합니다.')
    for i,p in enumerate(cups,1):
        if not isinstance(p,list) or len(p)!=3:raise ValueError(f'접시 {i} 좌표는 [x,y,z] 형식입니다.')
        for v in p:number(v,f'접시 {i} 파지 좌표')
        ik(p,-90)
    stack=cfg.get('plate_stack_wrist_cm')
    if not isinstance(stack,list) or len(stack)!=3:raise ValueError('적층 좌표는 [x,y,z] 형식입니다.')
    for v in stack:number(v,'접시 적층 좌표')
    rise=number(cfg.get('stack_rise_mm'),'접시 한 장당 적층 높이 증가량(mm)')
    if not 0<rise<=20:raise ValueError('적층 높이 증가는 0 초과 20mm 이하로 입력하세요.')
    clearance=number(cfg.get('clearance_z_cm'),'이송 높이')
    if clearance<max(c[2] for c in cups)+2 or clearance<stack[2]+rise/10+2 or clearance>25:
        raise ValueError('이송 높이는 파지/적층 높이보다 2cm 이상 높고 25cm 이하가 필요합니다.')
    pour=cfg.get('pour_wrist_cm')
    if not isinstance(pour,list) or len(pour)!=3:raise ValueError('배출 좌표 형식이 잘못됐습니다.')
    for v in pour:number(v,'배출 손목 좌표')
    angle=number(cfg.get('pour_pitch_deg'),'배출 기울기')
    if not 0<=angle<=20:raise ValueError('배출 그리퍼 각도는 수평 0도부터 위쪽 20도까지만 가능합니다.')
    cycles=number(cfg.get('tilt_cycles'),'기울이기 횟수')
    if int(cycles)!=cycles or not 1<=cycles<=3:raise ValueError('기울이기는 1~3회입니다.')
    hold=number(cfg.get('hold_seconds'),'유지 시간')
    if not .5<=hold<=3:raise ValueError('유지 시간은 0.5~3초입니다.')
    for p in (pour,[pour[0],pour[1],clearance]):ik(p,-90)
    ik(pour,angle)
    for z in (stack[2],stack[2]+rise/10):ik([stack[0],stack[1],z],-90)
    return cfg


def plan(cfg,start):
    validate(cfg);check_q(start)
    if abs(pitch(start))>3:raise ValueError('시작 전 그리퍼를 수평으로 맞춰주세요.')
    q=list(start);stages=[];height=cfg['clearance_z_cm']
    def move(name,p,angle=-90):
        nonlocal q
        goal=ik(p,angle)
        stages.append({'name':name,'kind':'move','xyz':list(p),'pitch':angle,'target':goal})
        q=goal
    def rotate(name,angle):
        nonlocal q
        goal=q.copy();goal[3]=angle-90-q[1]-q[2];check_q(goal)
        stages.append({'name':name,'kind':'rotate','pitch':angle,'target':goal})
        q=goal
    def event(name,kind,plate=None):stages.append({'name':name,'kind':kind,'plate':plate})
    here=wrist_fk(q)
    move('INITIAL_RAISE',[here[0],here[1],max(here[2],height)],0)
    move('ROTATION_CLEARANCE',[0.,10.,height],0)
    event('OPEN','open')
    rotate('POINT_GRIPPER_DOWN',-90)
    for i in range(2):
        source=cfg['plate_grasp_wrist_cm'][i]
        out=list(cfg['plate_stack_wrist_cm']);out[2]+=i*cfg['stack_rise_mm']/10
        wet=cfg['pour_wrist_cm']
        event(f'PLATE_{i+1}_START','marker',i+1)
        # Transit via a high central position to avoid long chords close to the base.
        move('ABOVE_FRONT',[0.,10.,height])
        move('ABOVE_PLATE',[source[0],source[1],height])
        move('LOWER_TO_PLATE',source)
        event('GRASP','close')
        move('LIFT_DOWNWARD',[source[0],source[1],height])
        move('TO_DISPOSAL_ABOVE',[wet[0],wet[1],height])
        if any(abs(a-b)>1e-8 for a,b in zip([wet[0],wet[1],height],wet)):
            move('DISPOSAL_POSITION',wet)
        for j in range(int(cfg['tilt_cycles'])):
            rotate(f'TIP_UP_{j+1}',cfg['pour_pitch_deg'])
            event(f'EMPTY_HOLD_{j+1}','hold')
            rotate(f'POINT_DOWN_{j+1}',-90)
            event(f'RESET_HOLD_{j+1}','hold')
        move('LEAVE_DISPOSAL',[wet[0],wet[1],height])
        move('STACK_TRANSIT',[0.,10.,height])
        move('ABOVE_PLATE_STACK',[out[0],out[1],height])
        move('LOWER_TO_STACK',out)
        event('RELEASE','open')
        move('LIFT_OPEN_GRIPPER',[out[0],out[1],height])
    return stages


def execute_motion(robot,bench,stage,record):
    start=bench.read();goal=stage['target'];check_q(goal)
    is_rotation=stage['kind']=='rotate'
    fine=stage['name'] in ('LOWER_TO_PLATE','LOWER_TO_STACK')
    tolerance=.3 if fine else .5
    angle_tol=3.0 if is_rotation else 2.5
    timeout=min(60.,max(12.,max(abs(a-b) for a,b in zip(start,goal))*.6))
    try:
        if is_rotation:
            robot.unlock();bench._plate_mode=None
            robot.set_degree(4,goal[3],bench.speed)
            mode='wrist'
        else:
            mode='horizontal' if stage['pitch']==0 else 'vertical'
            if getattr(bench,'_plate_mode',None)!=mode:
                if abs(pitch(start)-stage['pitch'])>4:
                    raise RuntimeError('그리퍼 방향이 자세 전환 기준과 4도 이상 다릅니다.')
                if mode=='horizontal':robot.lock_horz()
                else:robot.lock_vert()
                bench._plate_mode=mode
            robot.set_degree(goal[0],goal[1],goal[2],None,bench.speed)
        record('command',phase=stage['name'],mode=mode,observation_q=start,target_q=goal)
        print('목표 1회 전송:',mode,[round(v,2) for v in goal],flush=True)
        deadline=time.monotonic()+timeout;history=[]
        while time.monotonic()<deadline:
            q=bench.read();history.append(q);history=history[-5:]
            joint=max(abs(a-b) for a,b in zip(q,goal))
            pos=math.dist(wrist_fk(q),wrist_fk(goal));p_err=abs(pitch(q)-stage['pitch'])
            if not is_rotation and p_err>6:
                raise RuntimeError(f'이동 중 그리퍼 방향 오차 {p_err:.1f}도')
            if len(history)==5:
                spread=max(max(row[j] for row in history)-min(row[j] for row in history) for j in range(4))
                if joint<=2 and pos<=tolerance and p_err<=angle_tol and spread<=.5:
                    print('단계 완료 — 위치 오차 %.1fmm, 방향 오차 %.2f도'%(pos*10,p_err),flush=True)
                    return
            time.sleep(.1)
        raise RuntimeError('목표 도착 시간 초과')
    except (Exception,KeyboardInterrupt):
        robot.stop()
        try:
            actual=bench.read()
            print('MOTION_STOP',stage['name'],'TARGET',[round(x,3) for x in goal],
                  'ACTUAL',[round(x,3) for x in actual],flush=True)
            record('motion_stop',phase=stage['name'],target_q=goal,observation_q=actual)
        except Exception:pass
        raise


def write_template(root,path):
    if path.exists():raise FileExistsError(f'{path} 파일이 이미 있어 덮어쓰지 않았습니다.')
    cup_path=root/'fixed_cup_config.json'
    cup=json.loads(cup_path.read_text())
    placement=cup['place_wrist_cm'];wet=cup['pour_wrist_cm']
    cfg={
        '_readme':'손목 x,y,z 단위 cm. null 값을 실측 숫자로 바꿔야 실행 가능합니다.',
        '_note':'아래 x,y 예시는 위치 보장 아님. 컵 적층 손목 좌표에서 y+5cm이며 그리퍼 끝 사이 5cm 보장 아님.',
        'plate_grasp_wrist_cm':[[0.0,10.0,1.5],[-5.0,10.0,1.5]],
        'plate_stack_wrist_cm':[round(placement[0],4),round(placement[1]+5,4),None],
        'stack_rise_mm':None,
        'pour_wrist_cm':list(wet),
        'clearance_z_cm':15.0,
        'pour_pitch_deg':10.0,
        'tilt_cycles':2,
        'hold_seconds':1.0,
    }
    path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    print('설정 파일 생성:',path)
    print('요청한 파지 z=1.5cm를 손목 기준으로 임시 입력했습니다. 집게 끝 높이와 혼동하지 마세요.')
    print('적층 z, 적층 높이 증가량(mm)을 실측하여 입력하세요. x,y도 실제 배치에 맞춰 조정하세요.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent
    parser.add_argument('--config',type=Path,default=root/'plate_config.json')
    parser.add_argument('--init',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--empty-run',action='store_true')
    parser.add_argument('--step',action='store_true')
    parser.add_argument('--port',default='/dev/ttyACM0')
    parser.add_argument('--speed',type=int,default=40)
    args=parser.parse_args()
    if args.empty_run and not args.execute:parser.error('--empty-run에는 --execute가 필요합니다.')
    if not 1<=args.speed<=40:parser.error('속도는 1~40입니다.')
    if args.init:
        try:write_template(root,args.config)
        except (OSError,KeyError,ValueError) as e:parser.error(str(e))
        return
    try:
        cfg=json.loads(args.config.read_text())
        stages=plan(cfg,[0,-10,-140,60])
    except (OSError,ValueError,KeyError,TypeError) as e:parser.error(str(e))
    print('접시 1,2 파지 손목(cm):',cfg['plate_grasp_wrist_cm'])
    print('적층 1 손목(cm):',cfg['plate_stack_wrist_cm'])
    out=cfg['plate_stack_wrist_cm'].copy();out[2]+=cfg['stack_rise_mm']/10
    print('적층 2 손목(cm):',out)
    print('배출 손목(cm):',cfg['pour_wrist_cm'],'배출 그리퍼 pitch:',cfg['pour_pitch_deg'])
    print('오프라인 endpoint 기구학 통과. 충돌/실제 파지/음식물은 검증하지 않습니다.')
    if not args.execute:
        for s in stages:print(s['name'],s.get('xyz',s.get('pitch','')))
        print('계획만 출력. 로봇에 연결하지 않았습니다.')
        return
    if version('roboid')!=SDK_VERSION:parser.error('roboid==1.7.3 필요')
    from roboid import Raccoon
    robot=None;log=None
    try:
        robot=Raccoon(port_name=args.port);time.sleep(.5)
        bench=Bench(robot,args.speed,root/'cup_pose.json',2,2)
        stages=plan(cfg,bench.read())
        logdir=root/'real_logs';logdir.mkdir(exist_ok=True)
        logpath=logdir/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_two_plates.jsonl')
        log=logpath.open('w');t0=time.monotonic()
        def record(event,**kw):
            log.write(json.dumps(dict(time_s=time.monotonic()-t0,event=event,**kw))+'\n');log.flush()
        record('start',config=cfg,empty_run=args.empty_run,speed=args.speed)
        print('LOG:',logpath,'빈 그리퍼:',args.empty_run,flush=True)
        for stage in stages:
            print('PHASE:',stage['name'],stage.get('xyz',stage.get('pitch','')),flush=True)
            if args.step:input('주변 확인 후 Enter: ')
            record('phase',name=stage['name'])
            if stage['kind'] in ('move','rotate'):
                execute_motion(robot,bench,stage,record)
            elif stage['kind']=='open':
                robot.open_gripper();record('gripper_command',command='open')
            elif stage['kind']=='close':
                if args.empty_run:print('EMPTY RUN: close 생략')
                else:robot.close_gripper();record('gripper_command',command='close')
                time.sleep(.5)
            elif stage['kind']=='hold':time.sleep(cfg['hold_seconds'])
        record('sequence_complete')
        print('접시 2개 동작 명령 완료. 실제 파지·음식물 제거·적층은 확인해줘.')
    except (KeyboardInterrupt,EOFError):print('\n중단 요청. 그리퍼와 접시 상태를 확인해줘.')
    except Exception as e:
        print('STOP:',e)
        if log:log.write(json.dumps({'event':'error','message':str(e)})+'\n');log.flush()
        raise SystemExit(1)
    finally:
        if robot is not None:
            robot.stop();time.sleep(.2);robot.dispose()
        if log:log.close()

if __name__=='__main__':main()



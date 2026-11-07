"""One tissue/ring demonstration using the existing RaccoonBot motion functions.
No motion unless --execute. --init writes an editable task configuration once.
All coordinates are wrist coordinates in cm, not gripper tip coordinates.
"""
import argparse
import json
import math
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from cup_pick_real import Bench, SDK_VERSION, check_q, pitch, wrist_fk
from plate_two_auto import ik, execute_motion

ROOT = Path(__file__).resolve().parent


def point(value, name):
    if not isinstance(value, list) or len(value) != 3 or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in value
    ):
        raise ValueError(f'{name}은 측정한 숫자 [x, y, z]로 입력하세요.')
    return [float(x) for x in value]


def make_plan(start, cfg, task):
    q = list(check_q(start))
    if abs(pitch(q)) > 3:
        raise ValueError('시작 자세를 수평(pitch 약 0도)으로 맞춰주세요.')
    grasp = point(cfg.get('grasp_wrist_cm'), '파지 손목 좌표')
    drop = point(cfg.get('drop_wrist_cm'), '배출 손목 좌표')
    h = cfg.get('clearance_z_cm')
    if isinstance(h, bool) or not isinstance(h, (int, float)) or not math.isfinite(h):
        raise ValueError('clearance_z_cm을 숫자로 입력하세요.')
    if h < grasp[2] + 2 or not 0 < h <= 25:
        raise ValueError('이송 높이는 파지 손목 z보다 2cm 이상 높고 25cm 이하여야 합니다.')
    stages = []
    def move(name, xyz, direction):
        nonlocal q
        goal = ik(xyz, direction)
        stages.append(dict(name=name, kind='move', xyz=list(xyz), pitch=direction, target=goal))
        q = goal
    def rotate(name, direction):
        nonlocal q
        goal = q.copy(); goal[3] = direction - 90 - q[1] - q[2]; check_q(goal)
        stages.append(dict(name=name, kind='rotate', pitch=direction, target=goal))
        q = goal
    def event(name, kind):
        stages.append(dict(name=name, kind=kind))
    here = wrist_fk(q)
    move('INITIAL_RAISE', [here[0], here[1], max(here[2], h)], 0)
    move('ROTATION_CLEARANCE', [0.0, 10.0, h], 0)
    event('OPEN', 'open')
    rotate('POINT_GRIPPER_DOWN', -90)
    move('ABOVE_OBJECT', [grasp[0], grasp[1], h], -90)
    move('LOWER_TO_OBJECT', grasp, -90)
    event('GRASP', 'close')
    move('LIFT_OBJECT', [grasp[0], grasp[1], h], -90)
    if task == 'tissue':
        rotate('FACE_FORWARD', 0)
        move('DROP_TRANSIT', [0.0, 10.0, h], 0)
        move('ABOVE_BIN', drop, 0)
    else:
        move('DROP_TRANSIT', [0.0, 10.0, h], -90)
        move('ABOVE_BIN', drop, -90)
    event('RELEASE', 'open')

    if task == 'tissue':
        # 배출 후 그리퍼를 들어 올리고 중앙을 거쳐 시작 위치로 복귀
        move('RETURN_ABOVE_BIN', [drop[0], drop[1], max(h, drop[2])], 0)
        move('RETURN_CENTER', [0.0, 10.0, h], 0)
        move('RETURN_START', [here[0], here[1], here[2]], 0)

    return stages


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', required=True, choices=('tissue','ring'))
    p.add_argument('--config', type=Path)
    p.add_argument('--init', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--empty-run', action='store_true')
    p.add_argument('--step', action='store_true')
    p.add_argument('--port', default='/dev/ttyACM0')
    args=p.parse_args()
    config=args.config or ROOT / f'{args.kind}_config.json'
    if args.init:
        template={'_comment':'손목 좌표 cm. 파지 좌표는 기존 포크 예시입니다. 배출 [0,0,0]은 임시값입니다. 티칭하여 변경하세요.',
                  'grasp_wrist_cm':[0.093,15.202,9.4],
                  'drop_wrist_cm':([0.0,0.0,0.0] if args.kind == 'tissue' else [-math.hypot(0.093,15.202),0.0,15.0]),
                  'clearance_z_cm':15.0}
        try:
            with config.open('x', encoding='utf-8') as f:
                json.dump(template,f,ensure_ascii=False,indent=2);f.write('\n')
        except FileExistsError:
            p.error(f'기존 파일은 덮어쓰지 않습니다: {config}')
        print('설정 생성:',config)
        print('파지 좌표는 기존 포크의 예시입니다. 배출 좌표를 확인하거나 티칭해 입력하세요.')
        return
    if args.empty_run and not args.execute: p.error('--empty-run에는 --execute가 필요합니다.')
    try:
        cfg=json.loads(config.read_text(encoding='utf-8'))
        cup=json.loads((ROOT/'fixed_cup_config.json').read_text(encoding='utf-8'))
        speed=cup['speed']
        if isinstance(speed,bool) or not isinstance(speed,int) or not 1 <= speed <= 40:
            raise ValueError('fixed_cup_config.json의 speed는 1~40의 정수여야 합니다.')
        stages=make_plan([0,-10,-140,60],cfg,args.kind)
    except (OSError, ValueError, KeyError, TypeError) as e: p.error(str(e))
    print('종류:',args.kind,'파지 손목(cm):',cfg['grasp_wrist_cm'],'배출 손목(cm):',cfg['drop_wrist_cm'])
    print('speed:',speed,'기구학 endpoint 확인; 충돌이나 실제 파지는 확인하지 않습니다.')
    for s in stages: print('PHASE:',s['name'],s.get('xyz',s.get('pitch','')))
    if not args.execute:
        print('계획만 확인했습니다. 로봇에는 연결하지 않았습니다.')
        return
    if version('roboid') != SDK_VERSION: p.error(f'roboid {SDK_VERSION} 환경이 필요합니다.')
    from roboid import Raccoon
    robot=None; log=None
    try:
        robot=Raccoon(port_name=args.port);time.sleep(.5)
        bench=Bench(robot,speed,ROOT/'cup_pose.json',2,2)
        stages=make_plan(bench.read(),cfg,args.kind)
        logdir=ROOT/'real_logs';logdir.mkdir(exist_ok=True)
        logpath=logdir/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+args.kind+'.jsonl')
        log=logpath.open('x',encoding='utf-8');start=time.monotonic()
        def record(event,**values):
            log.write(json.dumps({'time_s':time.monotonic()-start,'event':event,**values},ensure_ascii=False)+'\n');log.flush()
        record('start',config=cfg,empty_run=args.empty_run,speed=speed)
        print('LOG:',logpath,'빈 그리퍼:',args.empty_run)
        for stage in stages:
            print('PHASE:',stage['name'],stage.get('xyz',stage.get('pitch','')),flush=True)
            if args.step:input('주변 확인 후 Enter: ')
            record('phase',name=stage['name'])
            if stage['kind'] in ('move','rotate'): execute_motion(robot,bench,stage,record)
            elif stage['kind']=='open': robot.open_gripper();record('gripper_command',command='open');time.sleep(.5)
            elif stage['kind']=='close':
                if args.empty_run: print('EMPTY RUN: close 생략')
                else: robot.close_gripper();record('gripper_command',command='close')
                time.sleep(.5)
        record('sequence_complete')
        print('동작 명령 완료. 실제 파지와 배출 상태를 확인하세요.')
    except (KeyboardInterrupt,EOFError):
        print('중단 요청.');raise SystemExit(1)
    finally:
        if robot is not None:
            try: robot.stop();time.sleep(.2)
            finally: robot.dispose()
        if log is not None: log.close()

if __name__ == '__main__': main()

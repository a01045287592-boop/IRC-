"""Three fixed-position cups: grasp, tilt, place at increasing stack heights.
Coordinates are wrist (cm) and must be measured on the real workbench.
Default is an offline plan; --execute enables motion. No vision or grasp verification.
"""
import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from cup_pick_real import Bench, SDK_VERSION, wrist_fk, horizontal_ik, execution_plan
from fixed_cup_demo import plan
from smooth_cup_motion import execute_phase


def make_configs(base, cups, rise_mm):
    if len(cups) != 3:
        raise ValueError('컵 파지 위치 --cup을 정확히 세 개 입력하세요.')
    if not math.isfinite(rise_mm) or not 0 < rise_mm <= 20:
        raise ValueError('실측 컵 한 장당 쌓임 증가량은 0보다 크고 20mm 이하여야 합니다.')
    if any(len(c)!=3 or any(not math.isfinite(v) for v in c) for c in cups):
        raise ValueError('컵 좌표가 유효하지 않습니다.')
    if any(math.dist(a,b)<.1 for i,a in enumerate(cups) for b in cups[i+1:]):
        raise ValueError('세 컵의 파지 위치가 중복됩니다. 고정된 서로 다른 3개 위치가 필요합니다.')
    configs=[]
    for index, cup in enumerate(cups):
        cfg=copy.deepcopy(base)
        cfg['grasp_wrist_cm']=list(cup)
        cfg['place_wrist_cm'][2]=base['place_wrist_cm'][2]+index*rise_mm/10
        configs.append(cfg)
    return configs


def return_to_front(start, height):
    initial = wrist_fk(start)
    destination = [0.0, 10.0, height]
    n = max(1, math.ceil(math.dist(initial, destination) / .1))
    previous = list(start)
    dense = []
    for i in range(1, n + 1):
        point = [a + (b - a) * i / n for a, b in zip(initial, destination)]
        q = horizontal_ik(point)
        if max(abs(a-b) for a,b in zip(previous,q)) > 5:
            raise ValueError('복귀 경로의 관절 변화가 큽니다.')
        dense.append(q)
        previous = q
    targets = execution_plan(start, dense)
    return {'name':'RIGHT_TRANSIT', 'kind':'move', 'targets':targets,
            'wrist_cm':destination}


def plan_one(cfg, start, return_first):
    if return_first:
        transition = return_to_front(start, cfg['clearance_z_cm'])
        return [transition] + plan(cfg, transition['targets'][-1])
    return plan(cfg, start)


def preflight(configs, start):
    current=list(start); output=[]
    for i,cfg in enumerate(configs):
        steps=plan_one(cfg,current,i>0)
        last=next((s['targets'][-1] for s in reversed(steps)
                   if s.get('targets')),None)
        if last is None:raise RuntimeError('빈 경로입니다.')
        output.append(steps);current=list(last)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent
    parser.add_argument('--cup',nargs=3,type=float,action='append',required=True,
                        metavar=('X_CM','Y_CM','Z_CM'),help='컵 파지 손목 좌표; 총 3회 입력')
    parser.add_argument('--stack-step-mm',type=float,required=True,
                        help='컵 1개씩 포갰을 때 파지 위치 증가량 (실측 mm)')
    parser.add_argument('--config',type=Path,default=root/'fixed_cup_config.json')
    parser.add_argument('--port',default='/dev/ttyACM0')
    parser.add_argument('--execute',action='store_true',help='실제 로봇 동작')
    parser.add_argument('--empty-run',action='store_true',help='실제 이동, 그리퍼 닫지 않음')
    parser.add_argument('--step',action='store_true',help='각 단계 전 Enter (선택)')
    args=parser.parse_args()
    if args.empty_run and not args.execute:parser.error('--empty-run에는 --execute가 필요합니다')
    try:
        base=json.loads(args.config.read_text())
        configs=make_configs(base,args.cup,args.stack_step_mm)
        plans=preflight(configs,[0,-10,-140,60])
    except (OSError,ValueError,KeyError,TypeError) as exc:
        parser.error(str(exc))
    for i,cfg in enumerate(configs,1):
        print(f"컵 {i} 파지 손목(cm): {[round(v,3) for v in cfg['grasp_wrist_cm']]}",
              f"배치 손목(cm): {[round(v,3) for v in cfg['place_wrist_cm']]}")
    print('기구학 검사: 예시 초기 자세 기준 3개 경로 통과. 충돌/파지/적층은 검증하지 않습니다.')
    if not args.execute:
        print('계획만 출력했습니다. 로봇에 연결하지 않았습니다.')
        return
    if version('roboid')!=SDK_VERSION:parser.error('roboid==1.7.3 필요')
    from roboid import Raccoon
    robot=None;log=None;holding_command=False
    try:
        robot=Raccoon(port_name=args.port);time.sleep(.5)
        bench=Bench(robot,int(base['speed']),root/'cup_pose.json',2,2)
        start=bench.read()
        preflight(configs,start)  # reject all 3 before issuing any movement
        logdir=root/'real_logs';logdir.mkdir(exist_ok=True)
        logpath=logdir/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_three_cups.jsonl')
        log=logpath.open('w');t0=time.monotonic()
        def record(event,**values):
            log.write(json.dumps(dict(time_s=time.monotonic()-t0,event=event,**values))+'\n')
            log.flush()
        record('start',configs=configs,empty_run=args.empty_run)
        print('LOG:',logpath,'빈 그리퍼:',args.empty_run,flush=True)
        for i,cfg in enumerate(configs,1):
            # Re-plan from the actual end of the previous cup, not the ideal position.
            steps=plan_one(cfg,bench.read(),i>1)
            print(f'\n===== 컵 {i}/3 시작 =====',flush=True)
            for s in steps:
                print(f"컵 {i} PHASE: {s['name']}",s.get('wrist_cm',s.get('pitch_deg','')),flush=True)
                if args.step:input('주변 확인 후 Enter: ')
                record('phase',cup=i,name=s['name'])
                if s['kind'] in ('move','rotate'):
                    execute_phase(robot,bench,s['targets'],s,record)
                    record('phase_done',cup=i,name=s['name'],measured_q=bench.read())
                elif s['kind']=='open':
                    robot.open_gripper();holding_command=False
                    record('gripper_command',cup=i,command='open')
                elif s['kind']=='close':
                    if args.empty_run:
                        print('EMPTY RUN: close 생략',flush=True)
                    else:
                        robot.close_gripper();holding_command=True
                        record('gripper_command',cup=i,command='close')
                    time.sleep(.5)
                elif s['kind']=='hold':time.sleep(cfg['hold_seconds'])
            record('cup_cycle_complete',cup=i)
        record('all_cycles_complete')
        print('세 컵 동작 명령 완료. 실제 파지/배출/적층 성공은 별도 확인하세요.')
    except (KeyboardInterrupt,EOFError):
        print('\n중단 요청. 그리퍼와 컵 상태를 확인하세요.')
    except Exception as exc:
        print('STOP:',exc,flush=True)
        if log:
            log.write(json.dumps(dict(event='error',message=str(exc),holding_command=holding_command))+'\n');log.flush()
        raise SystemExit(1)
    finally:
        if robot is not None:
            robot.stop();time.sleep(.2);robot.dispose()
        if log:log.close()

if __name__=='__main__':main()



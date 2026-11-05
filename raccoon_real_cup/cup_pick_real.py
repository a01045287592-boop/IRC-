"""Raccoon cup pick bench test. SDK 1.7.3 only; cm / degrees.
No simulator coordinates, perception, collision checking, or grasp-success sensor.
"""
import argparse
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

LIMITS = ((-120, 120), (-90, 30), (-150, 0), (-105, 105))
SDK_VERSION = '1.7.3'
CONTROL_VERSION = '3.0-bench-motion'


def check_q(q):
    if len(q) != 4 or any(not math.isfinite(v) for v in q):
        raise ValueError('관절각이 유효하지 않습니다.')
    for i, (v, (lo, hi)) in enumerate(zip(q, LIMITS), 1):
        if not lo <= v <= hi:
            raise ValueError(f'관절 {i}: {v:.2f}도는 [{lo}, {hi}] 밖입니다.')
    return list(q)


def wrist_fk(q):
    """SDK 1.7.3 wrist convention. +y forward at joint1=0; +z upward."""
    a, b, c, _ = map(math.radians, q)
    radius = -10 * (math.sin(b) + math.sin(b+c))
    return [-math.sin(a)*radius, math.cos(a)*radius,
            8.25 + 10*math.cos(b) + 10*math.cos(b+c)]


def horizontal_ik(p):
    """Elbow branch used by SDK; reject unreachable points rather than clamp."""
    if len(p) != 3 or any(not math.isfinite(v) for v in p):
        raise ValueError('유효하지 않은 좌표')
    x, y, z = p
    radius = math.hypot(x, y)
    if radius < 1 or z < 0:
        raise ValueError('베이스 축 근처 또는 음수 높이는 이 테스트에서 제외합니다.')
    h = z - 8.25
    cosine = (radius*radius + h*h - 200) / 200
    if not -1 <= cosine <= 1:
        raise ValueError('손목 목표가 링크 길이로 도달 불가능합니다.')
    c = -math.acos(cosine)
    b = math.atan2(-radius, h) - math.atan2(10*math.sin(c), 10+10*math.cos(c))
    a = math.atan2(-x, y)
    q = [math.degrees(a), math.degrees(b), math.degrees(c)]
    q.append(-90-q[1]-q[2])
    check_q(q)
    if math.dist(wrist_fk(q), p) > 1e-6:
        raise ValueError('기구학 역검증 실패')
    return q


def pitch(q):
    return 90 + sum(q[1:])


def line_plan(start_q, target):
    check_q(start_q)
    if abs(pitch(start_q)) > 3:
        raise ValueError('먼저 horz로 수평 자세를 맞추세요.')
    start = wrist_fk(start_q)
    n = max(1, math.ceil(math.dist(start, target) / .1))
    if n > 50:
        raise ValueError('한 경로는 최대 5cm입니다. 가까운 위치부터 맞추세요.')
    path = []
    previous = start_q
    for i in range(1, n+1):
        p = [a+(b-a)*i/n for a, b in zip(start, target)]
        q = horizontal_ik(p)
        if max(abs(a-b) for a, b in zip(previous, q)) > 5:
            raise ValueError('관절 변화가 큰 경로입니다. 다른 시작 자세를 사용하세요.')
        path.append(q)
        previous = q
    return path


def targets(grasp, retreat_cm, lift_cm):
    horizontal_ik(grasp)
    r = math.hypot(*grasp[:2])
    pre = [grasp[0]*(r-retreat_cm)/r, grasp[1]*(r-retreat_cm)/r, grasp[2]]
    lifted = [grasp[0], grasp[1], grasp[2]+lift_cm]
    line_plan(horizontal_ik(grasp), pre)
    line_plan(horizontal_ik(pre), grasp)
    line_plan(horizontal_ik(grasp), lifted)
    return pre, lifted


def execution_plan(start_q, dense):
    """Coalesce checked IK samples into <=5mm and <=5deg command segments."""
    output=[]; previous=start_q; i=0
    while i<len(dense):
        last=i
        while last+1<len(dense):
            q=dense[last+1]
            if math.dist(wrist_fk(previous),wrist_fk(q))>.500001 or max(abs(a-b) for a,b in zip(previous,q))>5:
                break
            last+=1
        q=dense[last]
        if max(abs(a-b) for a,b in zip(previous,q))>5:
            raise ValueError('Execution segment exceeds 5 degrees')
        output.append(q);previous=q;i=last+1
    return output


class Bench:
    def __init__(self, robot, speed, config, retreat_cm, lift_cm):
        self.r, self.speed, self.config = robot, speed, config
        self.retreat_cm, self.lift_cm = retreat_cm, lift_cm
        self.grasp = None
        self.held = False  # commanded/assumed state, NOT a grasp measurement
        self.motion_started = False
        self.faulted = False

    def read(self):
        if not self.r.is_connected():
            raise RuntimeError('로봇 연결이 끊겼습니다.')
        return check_q(list(self.r.encoder(-1)))

    def move_q(self, target, cartesian=False, position_tolerance_cm=.3):
        check_q(target)
        start_q = self.read()
        self.motion_started = True
        self.r.unlock()
        self.r.set_degree(*target, self.speed)
        started = time.monotonic()
        history = []
        q = start_q
        while time.monotonic() - started < 6:
            q = self.read()
            history.append(q)
            history = history[-7:]
            # A 1mm waypoint can already be within 1 degree before moving.
            # Do not overwrite it immediately while the servo is still responding.
            if time.monotonic() - started >= .7 and len(history) == 7:
                spread = max(max(s[j] for s in history)-min(s[j] for s in history)
                             for j in range(4))
                error = max(abs(a-b) for a,b in zip(q,target))
                position_ok = math.dist(wrist_fk(q),wrist_fk(target)) <= position_tolerance_cm
                pitch_ok = abs(pitch(q)-pitch(target)) <= 2
                if error <= 2.0 and spread <= .5 and (not cartesian or (position_ok and pitch_ok)):
                    return
            time.sleep(.05)
        self.r.stop()
        errors = [a-b for a,b in zip(q,target)]
        worst = max(range(4), key=lambda j: abs(errors[j]))
        print('MOTION_TIMEOUT (단위: degree)', flush=True)
        for label, values in (('START', start_q), ('TARGET', target),
                              ('ACTUAL', q), ('ERROR', errors)):
            print(label, [round(v,3) for v in values], flush=True)
        print('WRIST_ERROR_MM:', round(10*math.dist(wrist_fk(q),wrist_fk(target)),3),
              'PITCH_DEG:', round(pitch(q),3), flush=True)
        raise RuntimeError(f'6초 내 도착/정착 조건 미충족. 최대 오차 관절 {worst+1}: '
                           f'{errors[worst]:+.3f}도. 자동 재시도 없이 정지.')

    def move_line(self, target):
        path = line_plan(self.read(), target)  # validate whole segment before sending
        print('이동 목표 손목(cm):', [round(v,3) for v in target], flush=True)
        for q in execution_plan(self.read(), path):
            self.move_q(q, cartesian=True)
        if math.dist(wrist_fk(self.read()), target) > .3:
            raise RuntimeError('최종 위치 오차가 3mm를 초과했습니다.')

    def status(self):
        q = self.read()
        print('관절(deg):', [round(v,2) for v in q])
        print('손목(cm):', [round(v,3) for v in wrist_fk(q)], 'pitch:', round(pitch(q),2))

    def horizontal(self):
        q = self.read()
        target = q.copy()
        target[3] = -90-q[1]-q[2]
        check_q(target)
        # Gradual wrist rotation; cup must be absent during setup.
        n = max(1, math.ceil(abs(target[3]-q[3])/2))
        for i in range(1,n+1):
            waypoint = q.copy()
            waypoint[3] += (target[3]-q[3])*i/n
            self.move_q(waypoint)

    def need_target(self):
        if self.grasp is None:
            raise ValueError('먼저 mark 또는 load를 실행하세요.')
        return targets(self.grasp, self.retreat_cm, self.lift_cm)

    def near(self, p):
        if math.dist(wrist_fk(self.read()), p) > .5:
            raise ValueError('예상 시작 위치에서 5mm 이상 떨어졌습니다. jog로 위치를 맞추세요.')
        if abs(pitch(self.read())) > 3:
            raise ValueError('수평 자세가 아닙니다.')

    def command(self, text):
        fields = text.split()
        if not fields:
            return True
        cmd = fields[0]
        if self.faulted and cmd not in ('status','help','stop','quit','exit','ack'):
            raise ValueError('정지 상태입니다. status로 확인하고 문제를 해소한 뒤 ack를 입력하세요.')
        if cmd == 'ack':
            self.read()
            self.faulted=False
            print('오류 잠금 해제. 자동 재시도하지 않습니다.')
            return True
        if cmd in ('quit','exit'):
            if self.held:
                print('컵을 잡은 상태입니다. lower로 내려놓거나 컵을 받친 뒤 종료하세요.')
            return False
        if self.held and cmd not in ('status','lower','stop','help'):
            raise ValueError('컵 파지 명령 이후에는 lower/status/stop/quit만 사용하세요.')
        if cmd == 'help':
            print(HELP)
        elif cmd == 'status':
            self.status()
        elif cmd == 'stop':
            self.r.stop()
            print('정지했습니다. 컵 지지 상태를 직접 확인하세요.')
        elif cmd == 'horz':
            self.horizontal()
            self.status()
        elif cmd == 'joint' and len(fields)==3:
            joint, delta = int(fields[1]), float(fields[2])
            if joint not in (1,2,3,4) or not math.isfinite(delta) or abs(delta)>2:
                raise ValueError('joint 번호 1~4, 변화량 최대 ±2도')
            q = self.read()
            q[joint-1] += delta
            self.move_q(q)
            self.status()
        elif cmd == 'jog' and len(fields)==3:
            axis, delta = fields[1], float(fields[2])
            if axis not in ('x','y','z') or not math.isfinite(delta) or abs(delta)>.5:
                raise ValueError('jog 축 x/y/z, 거리 최대 ±0.5cm')
            p = wrist_fk(self.read())
            p['xyz'.index(axis)] += delta
            self.move_line(p)
            self.status()
        elif cmd == 'open':
            self.r.open_gripper()
        elif cmd == 'mark':
            q = self.read()
            if abs(pitch(q))>3:
                raise ValueError('수평으로 맞춘 뒤 mark 하세요.')
            p = wrist_fk(q)
            pre, lifted = targets(p,self.retreat_cm,self.lift_cm)
            record = dict(sdk=SDK_VERSION,units='cm',reference='wrist',grasp=p,
                          retreat_cm=self.retreat_cm,lift_cm=self.lift_cm,measured_q=q)
            if self.config.exists():
                backup=self.config.with_name(self.config.name+f'.{time.time_ns()}.bak')
                backup.write_bytes(self.config.read_bytes())
            self.config.parent.mkdir(parents=True,exist_ok=True)
            self.config.write_text(json.dumps(record,indent=2))
            self.grasp=p
            print('저장:',self.config,'\n접근 위치:',pre,'\n상승 위치:',lifted)
        elif cmd == 'load':
            data=json.loads(self.config.read_text())
            if data['sdk']!=SDK_VERSION or data['reference']!='wrist' or data['units']!='cm':
                raise ValueError('저장 파일의 SDK/좌표 기준이 다릅니다.')
            # Use this session offsets, not unvalidated file controls.
            targets(data['grasp'],self.retreat_cm,self.lift_cm)
            self.grasp=data['grasp']
            print('목표만 불러왔습니다. 자동 이동하지 않습니다:',self.grasp)
        elif cmd == 'retreat':
            pre,_=self.need_target()
            self.near(self.grasp)
            self.move_line(pre)
        elif cmd == 'pick':
            pre,lifted=self.need_target()
            self.near(pre)
            # Preflight approach and lift before the gripper changes state.
            line_plan(self.read(),self.grasp)
            line_plan(horizontal_ik(self.grasp),lifted)
            self.r.open_gripper()
            self.move_line(self.grasp)
            self.held=True
            self.r.close_gripper()
            time.sleep(.5)
            self.move_line(lifted)
            print('파지/상승 명령 완료. 실제 파지 성공은 눈으로 확인하세요. lower: 내려놓기')
        elif cmd == 'lower':
            if not self.held:
                raise ValueError('이번 세션에서 pick 명령을 실행한 뒤 사용하세요.')
            pre,lifted=self.need_target()
            self.near(lifted)
            line_plan(self.read(),self.grasp)
            line_plan(horizontal_ik(self.grasp),pre)
            self.move_line(self.grasp)
            self.r.open_gripper()
            self.held=False
            self.move_line(pre)
        else:
            raise ValueError('알 수 없는 명령. help를 입력하세요.')
        return True


HELP='''명령 (각 줄을 입력하고 Enter):
 status          관절각 / 손목 좌표 읽기
 horz            현재 손목 위치에서 그리퍼를 천천히 수평으로 회전
 joint 2 -1      2번 관절 -1도 (초기 자세 조정용, 수평 유지 안 됨)
 jog y 0.2       수평 유지하며 손목 y 방향 +2mm (x/y/z 가능)
 open            그리퍼 열기
 mark            현재 수평 자세를 컵 파지 위치로 저장
 retreat         저장된 파지 위치에서 베이스 쪽으로 2cm 후퇴
 pick            접근 위치 → 정면 접근 → 닫기 → 2cm 상승
 lower           원래 높이로 내림 → 열기 → 후퇴
 load            이전 파지 위치만 불러오기 (자동 이동 없음)
 stop            정지
 quit            종료
컵은 mark까지 치워두고, retreat 후 저장한 파지 위치에 놓으세요.
손목 좌표는 컵 중심 좌표가 아닙니다. Ctrl+C는 정지를 요청하고 종료합니다.
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',help='동글 시리얼 포트; 생략하면 SDK 자동 탐색')
    p.add_argument('--read',action='store_true',help='연결 후 상태만 읽고 종료')
    p.add_argument('--speed',type=int,default=10)
    p.add_argument('--retreat-cm',type=float,default=2.0)
    p.add_argument('--lift-cm',type=float,default=2.0)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('cup_pose.json'))
    args=p.parse_args()
    print('Controller:', CONTROL_VERSION, flush=True)
    if not 1<=args.speed<=20 or not .5<=args.retreat_cm<=3 or not .5<=args.lift_cm<=3:
        p.error('speed 1~20, retreat/lift 0.5~3cm 범위만 사용합니다.')
    if version('roboid') != SDK_VERSION:
        p.error('전용 환경에 roboid==1.7.3을 설치하세요. 다른 버전은 연결하지 않습니다.')
    from roboid import Raccoon
    robot=None
    try:
        # Explicit keyword avoids the string-positional constructor bug in this SDK.
        robot=Raccoon(port_name=args.port)
        deadline=time.monotonic()+5
        while not robot.is_connected():
            if time.monotonic()>deadline:
                raise RuntimeError('연결 실패. 동글/전원/포트 권한을 확인하세요.')
            time.sleep(.1)
        time.sleep(.5)
        bench=Bench(robot,args.speed,args.config,args.retreat_cm,args.lift_cm)
        bench.status()
        if args.read:
            return
        print(HELP)
        while True:
            try:
                if not bench.command(input('raccoon> ').strip()):
                    break
            except RuntimeError as exc:
                robot.stop()
                bench.faulted=True
                print('STOP:',exc,'\n이동 잠금: status/help/quit 또는 확인 후 ack')
            except (ValueError,FileNotFoundError,json.JSONDecodeError) as exc:
                robot.stop()
                print('입력/계획 거부:',exc)
    except (KeyboardInterrupt,EOFError):
        print('\n정지를 요청합니다. 컵을 잡고 있다면 직접 지지하세요.')
    except Exception as exc:
        print('STOP:',exc)
        raise SystemExit(1)
    finally:
        if robot is not None:
            if not args.read:
                robot.stop()
                time.sleep(.2)
            robot.dispose()


if __name__=='__main__':
    main()

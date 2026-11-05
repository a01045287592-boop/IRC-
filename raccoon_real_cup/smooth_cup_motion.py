"""Continuous per-phase joint references; no per-waypoint settle wait.
Actual servo tracking, collision avoidance and stop reliability are hardware-dependent.
"""
import math
import time
from cup_pick_real import check_q, pitch, wrist_fk

DT = .1
LINEAR_CM_S = 3.2
ANGULAR_DEG_S = 60.0
JOINT_LEAD_DEG = 6.
WRIST_LEAD_CM = .8
LEAD_TIMEOUT_S = 8.
FINAL_TIMEOUT_S = 8.


def metrics(actual, target):
    return (max(abs(a-b) for a,b in zip(actual,target)),
            math.dist(wrist_fk(actual),wrist_fk(target)),
            abs(pitch(actual)-pitch(target)))


def references(start, targets):
    current=list(start)
    for q in targets:
        check_q(q)
        d=max(abs(a-b) for a,b in zip(q,current))
        xyz=math.dist(wrist_fk(q),wrist_fk(current))
        n=max(1,math.ceil(d/(ANGULAR_DEG_S*DT)),math.ceil(xyz/(LINEAR_CM_S*DT)))
        if n>500:
            raise ValueError('경로 구간이 비정상적으로 깁니다.')
        for i in range(1,n+1):
            point=[a+(b-a)*i/n for a,b in zip(current,q)]
            check_q(point)
            yield point
        current=list(q)


def execute_phase(robot, bench, targets, phase, record):
    """SDK native horizontal lock for moves; one q4 target for tilt phases."""
    if not targets:
        return
    start = bench.read()
    goal = list(targets[-1])
    check_q(goal)
    if max(abs(a-b) for a,b in zip(start,targets[0])) > 8:
        raise RuntimeError('계획 경로 시작점에서 8도 이상 벗어났습니다.')
    is_rotate = phase['kind'] == 'rotate'
    transit = is_rotate or phase['name'] in (
        'INITIAL_RAISE','ABOVE_APPROACH','LIFT_UPRIGHT','ABOVE_POUR',
        'POUR_POSITION','LEAVE_POUR','RIGHT_TRANSIT','ABOVE_PLACE','FINISH_RAISED')
    position_tolerance = .5 if transit else .3
    pitch_tolerance = 3. if is_rotate else 2.
    timeout = min(60., max(12., max(abs(a-b) for a,b in zip(start,goal))*.6))
    history = []
    try:
        if is_rotate:
            robot.unlock()
            bench._native_horizontal_active = False
            # Tilt only: do not command other joints in the pouring phase.
            robot.set_degree(4, goal[3], bench.speed)
            control = 'wrist'
        else:
            # lock_horz can itself move joint 4. Enter only from near-horizontal.
            if not getattr(bench, '_native_horizontal_active', False):
                if abs(pitch(start)) > 3:
                    raise RuntimeError('수평 고정 전 기울기가 3도를 넘었습니다.')
                robot.lock_horz()
                bench._native_horizontal_active = True
            # In horizontal lock, command joints 1-3 once; the robot controls joint 4.
            robot.set_degree(goal[0], goal[1], goal[2], None, bench.speed)
            control = 'horizontal'
        record('command_native', mode=control, observation_q=start, target_q=goal)
        print('목표 1회 전송:',control,[round(v,2) for v in goal],flush=True)
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            q=bench.read()
            joint_err,xyz_err,pitch_err=metrics(q,goal)
            if not is_rotate and abs(pitch(q))>6:
                raise RuntimeError('이송 중 수평 기울기가 6도를 넘었습니다.')
            history.append(q);history=history[-5:]
            if len(history)==5:
                spread=max(max(row[j] for row in history)-min(row[j] for row in history) for j in range(4))
                if joint_err<=2 and xyz_err<=position_tolerance and pitch_err<=pitch_tolerance and spread<=.5:
                    print('단계 완료 — 관절 최대 오차 %.2f°, 손목 %.1fmm, 기울기 %.2f°' %
                          (joint_err,xyz_err*10,pitch_err),flush=True)
                    return
            time.sleep(.1)
        raise RuntimeError('목표 도착 시간 초과')
    except (Exception, KeyboardInterrupt):
        robot.stop()
        try:
            actual=bench.read()
            print('MOTION_STOP',phase['name'],'TARGET',[round(v,3) for v in goal],
                  'ACTUAL',[round(v,3) for v in actual],flush=True)
            record('motion_stop', name=phase['name'],target_q=goal,observation_q=actual)
        except Exception:
            pass
        raise



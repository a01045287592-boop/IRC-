import ast
import shutil
from datetime import datetime
from pathlib import Path

path = Path.home() / 'Raccoonbot_Openvla/raccoon_real_cup/smooth_cup_motion.py'
source = path.read_text()

new_function = '''def execute_phase(robot, bench, targets, phase, record):
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
'''

tree = ast.parse(source)
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_phase']
if len(nodes) != 1:
    raise SystemExit('execute_phase 함수를 찾지 못해서 변경하지 않았어.')
node = nodes[0]
old = ast.get_source_segment(source, node)
if 'robot.set_degree(*ref,bench.speed)' not in old and 'SDK native horizontal lock for moves' not in old:
    raise SystemExit('기존 연속 전송 방식이 아니라서 변경하지 않았어.')
lines=source.splitlines(keepends=True)
updated=''.join(lines[:node.lineno-1])+new_function+'\n'+''.join(lines[node.end_lineno:])
ast.parse(updated)
if updated == source:
    print('이미 적용되어 있어. 변경하지 않았어.')
    raise SystemExit(0)
stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
backup=path.with_name(path.name+'.backup_before_native_'+stamp)
shutil.copy2(path,backup)
path.write_text(updated)
print('적용 완료: 이동=수평 고정 및 목표 1회 / 붓기=4번 관절 목표 1회')
print('기존 파일 백업:',backup)


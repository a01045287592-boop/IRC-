import ast
import shutil
from datetime import datetime
from pathlib import Path

root = Path.home() / 'Raccoonbot_Openvla/raccoon_real_cup'
path = root / 'smooth_cup_motion.py'
source = path.read_text()

new_function = '''def execute_phase(robot, bench, targets, phase, record):
    """One joint target per phase. Actual joint path is hardware-controlled."""
    if not targets:
        return
    actual = bench.read()
    start = actual[:]
    goal = list(targets[-1])
    check_q(goal)
    if max(abs(a-b) for a,b in zip(actual, targets[0])) > 8:
        raise RuntimeError('계획 시작점에서 8도 이상 벗어났습니다.')
    timeout = min(60., max(12., max(abs(a-b) for a,b in zip(start, goal)) * .6))
    transit = phase['kind']=='rotate' or phase['name'] in (
        'INITIAL_RAISE','ABOVE_APPROACH','LIFT_UPRIGHT','ABOVE_POUR',
        'POUR_POSITION','LEAVE_POUR','RIGHT_TRANSIT','ABOVE_PLACE','FINISH_RAISED')
    position_tolerance = .5 if transit else .3
    pitch_tolerance = 3.0 if phase['kind']=='rotate' else 2.0
    q_history = []
    try:
        robot.unlock()
        # Unlike previous streaming, send the final target once.
        robot.set_degree(*goal, bench.speed)
        record('command_direct', observation_q=actual, target_q=goal)
        print('목표 1회 전송:', [round(q,2) for q in goal], flush=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            actual = bench.read()  # checks connection, valid joints
            joint_err, xyz_err, pitch_err = metrics(actual, goal)
            if phase['kind'] == 'move' and abs(pitch(actual)) > 6:
                raise RuntimeError('이송 중 수평 기울기가 6도를 넘었습니다.')
            q_history.append(actual)
            q_history = q_history[-5:]
            if len(q_history) == 5:
                spread = max(max(q[j] for q in q_history)-min(q[j] for q in q_history) for j in range(4))
                if joint_err <= 2.0 and xyz_err <= position_tolerance and pitch_err <= pitch_tolerance and spread <= .5:
                    print('단계 완료 — 관절 최대 오차 %.2f°, 손목 %.1fmm, 기울기 %.2f°' %
                          (joint_err, xyz_err*10, pitch_err), flush=True)
                    return
            time.sleep(.1)
        raise RuntimeError('목표 도착 시간 초과')
    except (Exception, KeyboardInterrupt):
        robot.stop()
        try:
            actual = bench.read()
            print('MOTION_STOP', phase['name'], 'TARGET', [round(v,3) for v in goal],
                  'ACTUAL', [round(v,3) for v in actual], flush=True)
            record('motion_stop', name=phase['name'], target_q=goal, observation_q=actual)
        except Exception:
            pass
        raise
'''

tree = ast.parse(source)
items = [node for node in tree.body
         if isinstance(node, ast.FunctionDef) and node.name == 'execute_phase']
if len(items) != 1:
    raise SystemExit('execute_phase 함수를 하나로 찾지 못해 변경하지 않았어.')

node = items[0]
lines = source.splitlines(keepends=True)
updated = ''.join(lines[:node.lineno-1]) + new_function + '\n' + ''.join(lines[node.end_lineno:])
ast.parse(updated)

stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
backup = path.with_name(path.name + '.backup_' + stamp)
shutil.copy2(path, backup)
path.write_text(updated)
print('수정 완료: 각 이동 단계마다 최종 목표 1회 전송')
print('목표 좌표/그리퍼 동작/속도/컵 개수 설정은 유지')
print('이전 코드 백업:', backup)


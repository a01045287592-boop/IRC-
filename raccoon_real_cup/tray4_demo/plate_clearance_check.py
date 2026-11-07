#!/usr/bin/env python3
"""Supervised plate-only check using the existing planner and controller.

Default: offline plan only. --execute connects the robot; Enter authorizes one
stage at a time. No camera/YOLO and no training-episode success classification.
"""
import argparse
import copy
import math
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

from core import ROLES, JsonLog, atomic_json, hash_data, read_json
from control import Controller, NativePort, Sampler
from planner import kin, plan_episode
from tray4 import ask, wait_until

ROOT = Path(__file__).resolve().parent
LABELS = {
    'INITIAL_RAISE': '시작 위치에서 손목 들어 올리기',
    'ROTATION_CLEARANCE': '회전 준비 위치로 이동',
    'OPEN': '집게 열기',
    'POINT_GRIPPER_DOWN': '집게를 아래로 돌리기',
    'ABOVE_FRONT': '앞쪽 경유점',
    'ABOVE_PLATE': '접시 위로 이동',
    'LOWER_TO_PLATE': '접시 파지 위치로 내리기',
    'GRASP': '접시 잡기',
    'LIFT_DOWNWARD': '접시 들어 올리기',
    'TO_DISPOSAL_ABOVE': '접시를 든 채 배출 위치 위로 이동 — 컵 간격 확인',
    'DISPOSAL_POSITION': '배출 높이로 내리기',
    'ROTATE_TO_STACK_SIDE': '접시 놓는 쪽으로 회전',
    'LOWER_TO_STACK': '접시 놓는 위치로 내리기',
    'RELEASE': '집게를 열어 접시 놓기',
    'LIFT_OPEN_GRIPPER': '빈 집게 들어 올리기',
    'RETURN_CENTER': '중앙 경유점으로 복귀',
    'RETURN_HORIZONTAL': '집게를 정면으로 돌리기',
    'RETURN_START': '시작 위치로 복귀',
    'RETURN_EXACT_HOME': '시작 관절 자세로 맞추기',
}


def plate_stages(cfg, home):
    plans = plan_episode(cfg, list(ROLES), home)
    return next(obj['stages'] for obj in plans if obj['id'] == 'plate_1')


def verify_pose(actual, expected):
    if (max(abs(a-b) for a, b in zip(actual, expected)) > 2.5
            or math.dist(kin.wrist_fk(actual), kin.wrist_fk(expected)) > .5):
        raise RuntimeError('확인한 자세와 현재 자세가 다릅니다. 자동으로 이동하지 않습니다.')


def sampler_error_guard(sampler):
    if sampler.error:
        raise RuntimeError(sampler.error)


def sampler_guard(sampler):
    sampler_error_guard(sampler)
    # Reject old sampler reads; these still come from the SDK's encoder cache.
    return sampler.before(time.monotonic_ns(), 100.)


def execute_stage(port, sampler, stage, log):
    controller = Controller(port, [stage], log, 'plate_1')
    # Open/close/hold stages retain the previous persistent goal and mode.
    controller.goal = copy.deepcopy(port.goal)
    controller.mode = port.mode
    while not controller.finished:
        joint = sampler_guard(sampler)
        controller.tick(joint['qpos_deg'], time.monotonic())
        if not controller.finished:
            time.sleep(.1)
    return port.read()


def describe(stage):
    label = LABELS.get(stage['name'], {'rotate':'손목 회전', 'hold':'자세 유지'}.get(stage['kind'], stage['kind']))
    text = stage['name'] + ' / ' + label
    if 'target' in stage:
        xyz = [round(x, 3) for x in kin.wrist_fk(stage['target'])]
        text += f" / 손목 cm: {xyz} / pitch: {stage['pitch']:.2f}도"
    return text


def drive_stages(port, sampler, stages, home, log):
    expected = list(home)
    for index, stage in enumerate(stages, 1):
        print(f'\n[{index}/{len(stages)}] 다음 단계: {describe(stage)}', flush=True)
        if stage['name'] == 'TO_DISPOSAL_ABOVE':
            print('접시 아랫면과 컵 윗면의 실제 간격을 확인하세요. Enter를 누르면 이송합니다.', flush=True)
        reply = ask('배치와 이동 공간 확인 후 Enter: 이 단계 실행 / Q: 중단 > ',
                    lambda: sampler_guard(sampler))
        if reply:
            raise KeyboardInterrupt('사용자가 단계 실행을 취소했습니다.')
        sampler_guard(sampler)
        verify_pose(port.read(), expected)
        log.write('check_stage_authorized', phase=stage['name'], index=index)
        expected = execute_stage(port, sampler, stage, log)
        print('단계 도착. 실제 손목 cm:', [round(x, 3) for x in kin.wrist_fk(expected)], flush=True)


def minimum_heights(events):
    """Encoder FK estimates only, including 20 Hz samples during each phase."""
    current = None
    minima = {}
    for row in events:
        if row['event'] == 'phase':
            current = row['phase']
        elif row['event'] == 'segment_motion_complete':
            current = None
        elif row['event'] == 'joint_observation' and current is not None:
            z = kin.wrist_fk(row['qpos_deg'])[2]
            minima[current] = min(minima.get(current, z), z)
    return minima


def execute(args, cfg, source_cfg):
    if version('roboid') != kin.SDK_VERSION:
        raise ValueError('roboid==1.7.3 환경에서 실행하세요.')
    from roboid import Raccoon

    stamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
    folder = args.out.expanduser() / stamp
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder/'config.json', cfg)
    meta = dict(purpose='supervised_plate_clearance_check', training_episode=False,
                status='starting', cup_contact='unverified', source_setup=str(args.setup.resolve()),
                source_config_hash=hash_data(source_cfg), expected_home_qpos_deg=args.home_q,
                speed=cfg['speed'], source_speed=source_cfg['speed'],
                observation='SDK encoder cache; FK wrist height is not measured object clearance')
    atomic_json(folder/'metadata.json', meta)
    log = JsonLog(folder/'events.jsonl')
    robot = port = sampler = None
    error = None
    print('접시 검사 기록:', folder, flush=True)
    print('카메라/YOLO 없이 티칭 좌표로 접시 계획만 실행합니다.', flush=True)
    print('Ctrl+C/Q로 중단하면 자동 복귀하지 않습니다. 잡힌 접시와 집게 상태를 확인하세요.', flush=True)
    try:
        robot = Raccoon(port_name=args.port)
        wait_until(robot.is_connected, lambda: None, 8., '로봇 연결 실패')
        port = NativePort(robot, cfg['speed'], log)
        home = port.read()
        verify_pose(home, args.home_q)
        stages = plate_stages(cfg, home)
        atomic_json(folder/'plan.json', dict(home_qpos_deg=home, stages=stages,
                    scope='original plate stages; joint/native-mode checks; no collision proof'))
        meta['home_qpos_deg'] = home
        sampler = Sampler(port, log, 20)
        sampler.start()
        wait_until(lambda: bool(sampler.samples), lambda: sampler_error_guard(sampler),
                   1., '관절 표본 없음')
        print('집게가 비어 있고 네 물체가 티칭한 배치에 있는지 확인하세요.', flush=True)
        meta['status'] = 'checking'
        drive_stages(port, sampler, stages, home, log)
        meta['status'] = 'motion_complete'
    except BaseException as exc:
        error = exc
        meta.update(status='aborted', error=repr(exc))
        log.write('check_error', message=repr(exc))
        print('접시 검사 중단:', exc, flush=True)
    finally:
        for label, resource, method in [('stop', port, 'stop'), ('sampler', sampler, 'close'),
                                        ('dispose', robot, 'dispose')]:
            if resource is not None:
                try:
                    getattr(resource, method)()
                except BaseException as exc:
                    meta[label+'_error'] = repr(exc)
                    meta['status'] = 'aborted'
                    error = error or exc
        log.close()
        from core import jsonl
        meta['minimum_fk_wrist_z_cm_by_phase'] = minimum_heights(jsonl(folder/'events.jsonl'))
        atomic_json(folder/'metadata.json', meta)
    for name in ('LIFT_DOWNWARD', 'TO_DISPOSAL_ABOVE', 'DISPOSAL_POSITION'):
        z = meta['minimum_fk_wrist_z_cm_by_phase'].get(name)
        if z is not None:
            print(f'{name}: 관절 기록으로 계산한 손목 최저 높이 {z:.3f}cm', flush=True)
    if error is None:
        try:
            reply = input('전체 검사에서 접시와 컵의 접촉이 없었나요? Y: 없음 / N: 접촉 / 기타: 미확인 > ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            reply = ''
        meta['cup_contact'] = {'y':'user_reported_none', 'n':'user_reported_contact'}.get(reply, 'unverified')
        atomic_json(folder/'metadata.json', meta)
    print('학습 에피소드가 아닌 접시 검사 기록입니다:', folder, flush=True)
    return 1 if error is not None or meta['cup_contact'] != 'user_reported_none' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', type=Path, required=True)
    parser.add_argument('--home-q', type=float, nargs=4, required=True, metavar=('Q1','Q2','Q3','Q4'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--speed', type=int, help='이번 검사에만 적용할 속도; 원래 설정 파일은 유지')
    parser.add_argument('--port', default='/dev/ttyACM0')
    parser.add_argument('--out', type=Path, default=ROOT.parents[1]/'raccoon_dataset/tray4_checks')
    args = parser.parse_args()
    try:
        source_cfg = read_json(args.setup)
        cfg = copy.deepcopy(source_cfg)
        if args.speed is not None:
            cfg['speed'] = args.speed
        stages = plate_stages(cfg, args.home_q)
        print(f"접시만 검사 / 검사 속도 {cfg['speed']} / 원래 설정 속도 {source_cfg['speed']}")
        for index, stage in enumerate(stages, 1):
            print(f'{index:02d}. {describe(stage)}')
        if not args.execute:
            print('오프라인 계획 검사 완료. 로봇/카메라 연결 없음. 실제 검사는 --execute로 시작합니다.')
            return 0
        return execute(args, cfg, source_cfg)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, str(exc)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())

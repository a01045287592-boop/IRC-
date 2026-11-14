#!/usr/bin/env python3
"""No-camera, no-dataset check of one tray5 taught grasp position."""
import argparse
import copy
import tempfile
import time
from pathlib import Path
from importlib.metadata import version

from core import LABELS, ROLES, JsonLog, read_json, validate_config
from control import Controller, NativePort, Sampler
from planner import kin, object_plan


ROOT = Path(__file__).resolve().parent


def wait_connected(robot, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if robot.is_connected():
            return
        time.sleep(.1)
    raise TimeoutError('로봇 연결 실패: USB BLE 동글과 포트를 확인하세요.')


def run_controller(port, sampler, stages, role):
    controller = Controller(port, stages, NoneLog(), role)
    while not controller.finished:
        row = sampler.before(time.monotonic_ns(), 100.)
        controller.tick(row['qpos_deg'], time.monotonic())
        time.sleep(.05)


class NoneLog:
    """Controller/NativePort logging sink; this checker saves no episode."""
    def write(self, *args, **kwargs):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', required=True, choices=ROLES)
    parser.add_argument('--setup', type=Path, default=ROOT.parents[1] / 'raccoon_dataset/tray4_setups/setup_20261113_first_demo.json')
    parser.add_argument('--port', default='/dev/ttyACM0')
    parser.add_argument('--execute', action='store_true', help='로봇을 실제 이동시킵니다.')
    args = parser.parse_args()

    cfg = read_json(args.setup)
    validate_config(cfg)
    # Offline preview: compute only the approach targets, without hardware access.
    if not args.execute:
        stages = object_plan(cfg, args.object, [0., -10., -140., 60.], [0., -10., -140., 60.])
        grasp = next(i for i,s in enumerate(stages) if s['kind']=='close')
        print(LABELS[args.object], '파지 전 접근 단계:')
        for s in stages[:grasp]:
            if s['kind'] in ('move','rotate'):
                print(s['name'], 'wrist cm=', [round(x,3) for x in kin.wrist_fk(s['target'])],
                      'pitch=', round(s['pitch'],2))
        print('오프라인 계획만 확인했습니다. 실제 이동에는 --execute가 필요합니다.')
        return

    if version('roboid') != kin.SDK_VERSION:
        parser.error(f'roboid=={kin.SDK_VERSION} 환경에서 실행하세요.')
    from roboid import Raccoon

    robot = port = sampler = None
    log = None
    try:
        robot = Raccoon(port_name=args.port)
        wait_connected(robot)
        with tempfile.TemporaryDirectory(prefix='tray5_pick_check_') as temp:
            log = JsonLog(Path(temp) / 'events.jsonl')
            port = NativePort(robot, cfg['speed'], log)
            home = port.read()
            stages = object_plan(cfg, args.object, home, home)
            grasp_index = next(i for i,s in enumerate(stages) if s['kind']=='close')
            approach = stages[:grasp_index]
            print('대상:', LABELS[args.object])
            print('파지 위치:', cfg['pickups'][args.object], 'cm')
            print('집게를 닫지 않고 파지 위치까지만 이동합니다. 영상/에피소드는 저장하지 않습니다.')
            reply = input('주변 공간이 비고 로봇이 시작 자세인지 확인 후 Enter / Q 취소 > ').strip().lower()
            if reply == 'q':
                return

            sampler = Sampler(port, log, 20)
            sampler.start()
            deadline = time.monotonic() + 2
            while not sampler.samples and time.monotonic() < deadline:
                if sampler.error:
                    raise RuntimeError(sampler.error)
                time.sleep(.02)
            if not sampler.samples:
                raise RuntimeError('관절 표본을 받지 못했습니다.')
            run_controller(port, sampler, approach, args.object)
            actual = port.read()
            target = approach[-1]['target']
            print('파지 위치 도착 확인')
            print('목표 손목 XYZ(cm):', [round(x,3) for x in kin.wrist_fk(target)])
            print('현재 손목 XYZ(cm):', [round(x,3) for x in kin.wrist_fk(actual)])
            print('관절(deg):', [round(x,2) for x in actual])
            reply = input('위치 확인 후 Enter: 같은 경로로 시작 자세 복귀 / Q: 그 자리에서 종료 > ').strip().lower()
            if reply == 'q':
                return

            # Reverse the already-validated approach path. Never close the gripper.
            previous = list(home)
            motion_rows = []
            for stage in approach:
                if stage['kind'] in ('move','rotate'):
                    motion_rows.append((stage, previous))
                    previous = list(stage['target'])
            reverse = []
            for stage, prior in reversed(motion_rows):
                row = copy.deepcopy(stage)
                row['name'] = 'RETURN_' + stage['name']
                row['target'] = prior
                if row['kind'] == 'rotate':
                    row['pitch'] = kin.pitch(prior)
                reverse.append(row)
            run_controller(port, sampler, reverse, args.object)
            print('시작 자세 복귀 완료')
    finally:
        if port:
            try: port.stop()
            except Exception: pass
        if sampler:
            try: sampler.close()
            except Exception: pass
        if log:
            log.close()
        if robot:
            robot.dispose()


if __name__ == '__main__':
    main()

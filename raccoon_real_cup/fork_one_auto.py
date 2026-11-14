"""RaccoonBot fork pick and place.

손목 좌표 단위: cm
기본 실행: 경로 계산만 수행
--execute: 실제 로봇 동작
--empty-run: 그리퍼를 닫지 않고 시험
--step: 단계마다 Enter
"""

import argparse
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

from cup_pick_real import (
    Bench,
    SDK_VERSION,
    check_q,
    pitch,
    wrist_fk,
)
from plate_two_auto import ik, execute_motion


ROOT = Path(__file__).resolve().parent


def load_settings():
    plate_path = ROOT / "plate_config.json"
    cup_path = ROOT / "fixed_cup_config.json"

    plate = json.loads(plate_path.read_text())
    cup = json.loads(cup_path.read_text())

    # 접시 1 위치에서 Z만 1cm 낮춤.
    # 포크가 접시 1과 같은 XY에 놓여 있다는 가정.
    # 실제 로봇에서 측정한 포크 파지 손목 좌표 (cm)
    fork = [0.093, 15.202, 9.400]

    speed = cup["speed"]

    if (
        isinstance(speed, bool)
        or not isinstance(speed, int)
        or not 1 <= speed <= 40
    ):
        raise ValueError(f"컵 속도 설정을 확인하세요: {speed!r}")

    return plate, fork, speed


def make_plan(start_q, fork, clearance_z, left_angle):
    check_q(start_q)

    if abs(pitch(start_q)) > 3:
        raise ValueError(
            "시작 자세에서 그리퍼가 수평이 아닙니다. "
            "안전한 수평 시작 자세를 확인하세요."
        )

    if not 0 < left_angle <= 90:
        raise ValueError("왼쪽 각도는 0도 초과 90도 이하로 입력하세요.")

    radius = math.hypot(fork[0], fork[1])

    # IK에서 joint1 = atan2(-x, y)
    angle_rad = math.radians(left_angle)

    destination = [
        -radius * math.sin(angle_rad),
        radius * math.cos(angle_rad),
        clearance_z,
    ]

    q = list(start_q)
    stages = []

    def move(name, xyz, desired_pitch):
        nonlocal q

        goal = ik(xyz, desired_pitch)

        stages.append({
            "name": name,
            "kind": "move",
            "xyz": list(xyz),
            "pitch": desired_pitch,
            "target": goal,
        })

        q = goal

    def rotate(name, desired_pitch):
        nonlocal q

        goal = q.copy()
        goal[3] = desired_pitch - 90 - q[1] - q[2]
        check_q(goal)

        stages.append({
            "name": name,
            "kind": "rotate",
            "pitch": desired_pitch,
            "target": goal,
        })

        q = goal

    def event(name, kind):
        stages.append({
            "name": name,
            "kind": kind,
        })

    here = wrist_fk(q)

    # 수평 자세로 안전 높이까지 상승
    move(
        "INITIAL_RAISE",
        [here[0], here[1], max(here[2], clearance_z)],
        0,
    )

    move("ROTATION_CLEARANCE", [0.0, 10.0, clearance_z], 0)

    event("OPEN", "open")

    # 그리퍼 아래 방향
    rotate("POINT_GRIPPER_DOWN", -90)

    # 포크 위로 이동 후 아래로 접근
    move(
        "ABOVE_FORK",
        [fork[0], fork[1], clearance_z],
        -90,
    )

    move("LOWER_TO_FORK", fork, -90)

    event("GRASP", "close")

    # 포크를 든 상태로 상승
    move(
        "LIFT_FORK",
        [fork[0], fork[1], clearance_z],
        -90,
    )

    # 높은 위치에서 왼쪽 배치 위치로 이동
    move("LEFT_TRANSIT", [0.0, 10.0, clearance_z], -90)

    move(
        "ABOVE_FORK_PLACE",
        [destination[0], destination[1], clearance_z],
        -90,
    )

    # 내려가지 않고 현재 이송 높이에서 포크를 놓음
    event("RELEASE", "open")

    return stages, destination


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--empty-run", action="store_true")
    parser.add_argument("--step", action="store_true")
    parser.add_argument("--port", default="/dev/ttyACM0")

    parser.add_argument(
        "--left-angle",
        type=float,
        default=90.0,
        help="1번 관절 왼쪽 배치 각도, 기본 90도",
    )

    args = parser.parse_args()

    if args.empty_run and not args.execute:
        parser.error("--empty-run은 --execute와 함께 사용하세요.")

    try:
        plate, fork, speed = load_settings()
        clearance_z = float(plate["clearance_z_cm"])

        # 실제 연결 전 예시 시작 자세로 endpoint 검사
        stages, destination = make_plan(
            [0, -10, -140, 60],
            fork,
            clearance_z,
            args.left_angle,
        )

    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"설정/계획 오류: {exc}")

    print("포크 파지 손목(cm):", [round(v, 3) for v in fork])
    print("포크 배치 손목(cm):", [round(v, 3) for v in destination])
    print("배치 1번 관절 목표(deg):", args.left_angle)
    print("속도:", speed)
    print("기구학 endpoint 검사 통과. 실제 충돌·파지는 확인되지 않았습니다.")

    if not args.execute:
        print("\n실제 로봇은 움직이지 않습니다. 계획:")
        for stage in stages:
            print(
                stage["name"],
                stage.get("xyz", stage.get("pitch", "")),
            )
        return

    if version("roboid") != SDK_VERSION:
        raise SystemExit(f"roboid {SDK_VERSION} 환경이 필요합니다.")

    from roboid import Raccoon

    robot = None

    def record(event, **kwargs):
        pass

    try:
        robot = Raccoon(port_name=args.port)
        time.sleep(0.5)

        bench = Bench(
            robot,
            speed,
            ROOT / "cup_pose.json",
            2,
            2,
        )

        # 실제 현재 관절각을 기준으로 다시 계획
        stages, destination = make_plan(
            bench.read(),
            fork,
            clearance_z,
            args.left_angle,
        )

        print("\n실제 동작 시작. 빈 그리퍼:", args.empty_run)
        print("Ctrl+C: 중단 요청")

        for stage in stages:
            print(
                "\nPHASE:",
                stage["name"],
                stage.get("xyz", stage.get("pitch", "")),
                flush=True,
            )

            if args.step:
                input("주변 확인 후 Enter: ")

            if stage["kind"] in ("move", "rotate"):
                execute_motion(robot, bench, stage, record)

            elif stage["kind"] == "open":
                robot.open_gripper()
                time.sleep(0.5)

            elif stage["kind"] == "close":
                if args.empty_run:
                    print("EMPTY RUN: 그리퍼 닫기 생략")
                else:
                    robot.close_gripper()

                time.sleep(0.5)

        print("\n포크 동작 명령 완료. 실제 파지와 배치 상태를 확인해줘.")

    except (KeyboardInterrupt, EOFError):
        print("\n중단 요청.")

    except Exception as exc:
        print("STOP:", exc)
        raise SystemExit(1)

    finally:
        if robot is not None:
            robot.stop()
            time.sleep(0.2)
            robot.dispose()


if __name__ == "__main__":
    main()

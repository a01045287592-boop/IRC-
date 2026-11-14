import argparse
import time
from pathlib import Path

import numpy as np
import torch
import pyrealsense2 as rs

from torchvision.transforms.functional import to_tensor
from PIL import Image

from roboid import Raccoon, wait_until_ready

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


# =========================================================
# 1. 설정
# =========================================================

BASE = Path("/home/min/Raccoonbot_Cafe_Clean")

CHECKPOINT = BASE / "checkpoint/072328/pretrained_model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 공식 라쿤봇 관절 범위 (degree)
JOINT_MIN = np.array([-120, -90, -150, -105], dtype=float)
JOINT_MAX = np.array([ 120,  30,    0,  105], dtype=float)

# 현재 관절 위치에서 허용할 최대 이동량
MAX_DELTA_DEG = 5.0

# 로봇 이동 속도. 공식 API의 speed 파라미터.
ROBOT_SPEED = 20

# 명령 수행 후 위치 허용 오차
POSITION_TOL_DEG = 3.0


# =========================================================
# 2. 인자
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--execute",
    action="store_true",
    help="검증을 통과한 ACT 관절 명령을 실제 로봇에 전송"
)

parser.add_argument(
    "--steps",
    type=int,
    default=1
)

parser.add_argument(
    "--gripper",
    type=int,
    choices=[0, 1],
    required=True,
    help="현재 그리퍼의 마지막 명령 상태: open=0, closed=1"
)

parser.add_argument(
    "--allow-gripper",
    action="store_true",
    help="ACT가 예측한 그리퍼 명령 전송 허용"
)

parser.add_argument(
    "--allow-mode",
    action="store_true",
    help="ACT가 예측한 고정 모드 명령 전송 허용"
)

args = parser.parse_args()

if not 1 <= args.steps <= 10:
    parser.error("--steps는 1~10만 허용합니다.")


# =========================================================
# 3. ACT 모델
# =========================================================

print("[1] Loading ACT Checkpoint 072328...")

policy = ACTPolicy.from_pretrained(
    str(CHECKPOINT),
    local_files_only=True
).to(DEVICE)

policy.eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=CHECKPOINT,
    preprocessor_overrides={
        "device_processor": {"device": DEVICE}
    }
)

print("ACT Ready:", DEVICE)


# =========================================================
# 4. RealSense
# =========================================================

print("[2] Connecting RealSense...")

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.rgb8,
    30
)

camera_started = False
robot = None

# 이 값은 그리퍼 센서 측정값이 아니라 마지막 명령 상태
gripper_state = args.gripper


def get_rgb():
    frames = pipeline.wait_for_frames(timeout_ms=3000)
    frame = frames.get_color_frame()

    if not frame:
        raise RuntimeError("RGB Frame 수신 실패")

    rgb = np.asanyarray(frame.get_data()).copy()

    if rgb.shape != (480, 640, 3):
        raise RuntimeError(f"Unexpected RGB shape: {rgb.shape}")

    return rgb


# =========================================================
# 5. RaccoonBot 상태
# =========================================================

def read_joints():
    # 공식 Python API: degree 단위 엔코더 4개
    values = robot.encoder(-1)

    if values is None:
        raise RuntimeError("엔코더 데이터 없음")

    joints = np.asarray(values, dtype=float)

    if joints.shape != (4,):
        raise RuntimeError(
            f"Unexpected encoder shape: {joints.shape}"
        )

    if not np.isfinite(joints).all():
        raise RuntimeError("엔코더 데이터 비정상")

    if np.any(joints < JOINT_MIN) or np.any(joints > JOINT_MAX):
        raise RuntimeError(
            f"엔코더 관절 범위 비정상: {joints}"
        )

    return joints


# =========================================================
# 6. ACT 추론
# =========================================================

def predict_action(rgb, joint_deg):

    # 실제 엔코더 degree -> 학습 데이터 rad
    joint_rad = np.deg2rad(joint_deg)

    state = np.concatenate([
        joint_rad,
        np.array([gripper_state], dtype=float)
    ]).astype(np.float32)

    rgb_tensor = to_tensor(Image.fromarray(rgb))

    observation = {
        "observation.images.front":
            rgb_tensor.unsqueeze(0),

        "observation.state":
            torch.from_numpy(state).unsqueeze(0)
    }

    # 매 관측마다 독립적으로 Action Chunk 예측
    policy.reset()

    with torch.inference_mode():

        processed = preprocessor(observation)

        chunk = policy.predict_action_chunk(processed)

        # 첫 번째 Action만 실행 대상으로 사용
        action = postprocessor(chunk[:, 0, :])

    action = action.squeeze(0).cpu().numpy()

    if action.shape != (8,) or not np.isfinite(action).all():
        raise RuntimeError(f"Invalid ACT action: {action}")

    return action


# =========================================================
# 7. 안전성 검사
# =========================================================

def validate_action(action, current_deg):

    # ACT 관절 목표는 rad
    target_deg = np.rad2deg(action[:4])

    if np.any(target_deg < JOINT_MIN):
        raise RuntimeError(
            f"Joint lower limit exceeded: {target_deg}"
        )

    if np.any(target_deg > JOINT_MAX):
        raise RuntimeError(
            f"Joint upper limit exceeded: {target_deg}"
        )

    delta = np.abs(target_deg - current_deg)

    print("Current :", np.round(current_deg, 2))
    print("Target  :", np.round(target_deg, 2))
    print("Delta   :", np.round(delta, 2))

    if np.any(delta > MAX_DELTA_DEG):
        print(
            f"Large movement detected: {delta.max():.2f} degrees"
        )
        print("Movement will be divided into smaller steps.")

    return target_deg


# =========================================================
# 8. 그리퍼 / 모드
# =========================================================

def execute_gripper(action):

    global gripper_state

    target = int(action[4] >= 0.5)

    if target == gripper_state:
        return

    if target == 1:
        print("Gripper: CLOSE")
        robot.close_gripper()
    else:
        print("Gripper: OPEN")
        robot.open_gripper()

    gripper_state = target


def execute_mode(action):

    values = action[5:8]

    # 학습 데이터의 one-hot 출력이라는 가정
    mode = int(np.argmax(values))

    if mode == 0:
        print("Mode: HORIZONTAL")
        robot.lock_horz()

    elif mode == 1:
        print("Mode: VERTICAL")
        robot.lock_vert()

    else:
        print("Mode: WRIST / UNLOCK")
        robot.unlock()


# =========================================================
# 9. 실제 실행
# =========================================================

try:
    pipeline.start(config)
    camera_started = True

    print("[3] Connecting RaccoonBot...")

    robot = Raccoon()
    wait_until_ready()

    print("RaccoonBot Connected")

    current = read_joints()
    print("Initial joints (deg):", current)

    if not args.execute:
        print("\nDRY RUN: 로봇에 이동 명령을 전송하지 않습니다.")
    else:
        print("\nREAL CONTROL ENABLED")
        print("긴급 정지 수단을 확보하고 주변을 비워주세요.")

        confirm = input(
            "실제 관절 이동을 시작하려면 MOVE 입력: "
        ).strip()

        if confirm != "MOVE":
            raise RuntimeError("Operator cancelled execution")

    # 카메라 자동 노출 안정화
    for _ in range(10):
        get_rgb()

    for step in range(args.steps):

        print(f"\n========== STEP {step + 1} ==========")

        # 실제 관측
        current_deg = read_joints()
        rgb = get_rgb()

        # ACT Inference
        action = predict_action(rgb, current_deg)

        target_deg = validate_action(action, current_deg)

        print("Predicted Action:", np.round(action, 4))

        if not args.execute:
            print("DRY RUN: action not sent")
            continue

        # 위치 제어: 명령 완료까지 기다리는 API
        print("Sending joint command...")

        # 현재 버전은 Horizontal 모드 이동만 허용
        mode_scores = action[5:8]
        mode = int(np.argmax(mode_scores))

        if mode != 0 or mode_scores[0] < 0.8:
            raise RuntimeError(
                f"Horizontal 이외 모드 예측: {mode_scores}"
            )

        start_deg = read_joints()

        # Horizontal 모드의 손목 자세 검증
        start_pitch = (
            90.0 + start_deg[1]
            + start_deg[2] + start_deg[3]
        )

        target_pitch = (
            90.0 + target_deg[1]
            + target_deg[2] + target_deg[3]
        )

        if abs(start_pitch) > 3.0:
            raise RuntimeError(
                f"현재 손목이 수평이 아님: {start_pitch:.2f}도"
            )

        if abs(target_pitch) > 3.0:
            raise RuntimeError(
                f"ACT 목표 손목이 수평이 아님: {target_pitch:.2f}도"
            )

        print("Mode: HORIZONTAL")
        print("Start pitch:", round(start_pitch, 2))
        print("Target pitch:", round(target_pitch, 2))

        # q4는 로봇의 Horizontal Lock이 제어
        robot.lock_horz()

        total_delta = target_deg - start_deg

        num_waypoints = max(
            1,
            int(np.ceil(np.max(np.abs(total_delta)) / MAX_DELTA_DEG))
        )

        print("Total waypoints:", num_waypoints)

        for i in range(1, num_waypoints + 1):
            ratio = i / num_waypoints
            waypoint = start_deg + total_delta * ratio

            print(
                f"Waypoint {i}/{num_waypoints}:",
                np.round(waypoint, 2)
            )

            robot.degree_to(
                float(waypoint[0]),
                float(waypoint[1]),
                float(waypoint[2]),
                None,
                ROBOT_SPEED,
                wait=True
            )

            actual = read_joints()
            error = np.abs(actual - waypoint)

            if np.any(error > POSITION_TOL_DEG):
                raise RuntimeError(
                    f"Waypoint tracking error: {error}"
                )

        # 실제 엔코더 피드백 검사
        actual_deg = read_joints()
        error = np.abs(actual_deg - target_deg)

        print("Actual:", np.round(actual_deg, 2))
        print("Tracking error:", np.round(error, 2))

        if np.any(error > POSITION_TOL_DEG):
            raise RuntimeError(
                f"Tracking error exceeded tolerance: {error}"
            )

        if args.allow_mode:
            execute_mode(action)

        if args.allow_gripper:
            execute_gripper(action)

        print("STEP COMPLETED")

    print("\nACT REAL CONTROL FINISHED")

except KeyboardInterrupt:
    print("\nInterrupted by user")

except Exception as e:
    print("\nCONTROL STOPPED:", e)

finally:
    if robot is not None:
        try:
            robot.stop()
        except Exception as e:
            print("Robot stop failed:", e)

        try:
            robot.dispose()
        except Exception:
            pass

    if camera_started:
        pipeline.stop()

    print("Resources closed")

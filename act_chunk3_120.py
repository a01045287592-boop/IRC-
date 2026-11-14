import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import pyrealsense2 as rs

from PIL import Image
from torchvision.transforms.functional import to_tensor

from roboid import Raccoon, wait_until_ready
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


# ======================================================
# CONFIG
# ======================================================

ROOT = Path("/home/min/Raccoonbot_Cafe_Clean")

CHECKPOINT = ROOT / "checkpoint/054246/pretrained_model"

TOTAL_ACTIONS = 120
CHUNK_ACTIONS = 5  # Reobserve after each predicted action.
SPEED = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODE_NAMES = ["HORIZONTAL", "VERTICAL", "WRIST"]

JOINT_LOW = np.array([-120., -90., -150., -105.])
JOINT_HIGH = np.array([120., 30., 0., 105.])

MAX_JUMP = 65.0
WAYPOINT_DEG = 5.0
JOINT_TOL = 3.0


# ======================================================
# LOAD MODEL
# ======================================================

def load_model():

    print("Loading ACT:", CHECKPOINT)

    policy = ACTPolicy.from_pretrained(
        str(CHECKPOINT),
        local_files_only=True
    ).to(DEVICE)

    policy.eval()
    policy.reset()

    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": DEVICE}
        }
    )

    print("ACT READY:", DEVICE)

    return policy, pre, post


# ======================================================
# REAL ROBOT STATE
# ======================================================

def read_joints(robot):

    q = np.asarray(
        robot.encoder(-1),
        dtype=np.float64
    )

    if q.shape != (4,) or not np.isfinite(q).all():
        raise RuntimeError("Invalid encoder state")

    if np.any(q < JOINT_LOW) or np.any(q > JOINT_HIGH):
        raise RuntimeError(f"Encoder outside joint limits: {q}")

    return q


def pitch(q):
    return 90.0 + q[1] + q[2] + q[3]


# ======================================================
# OBSERVATION
# ======================================================

def get_observation(robot, pipeline, gripper_state):

    frames = pipeline.wait_for_frames(timeout_ms=3000)

    color = frames.get_color_frame()

    if not color:
        raise RuntimeError("No RealSense image")

    rgb = np.asanyarray(color.get_data()).copy()

    current = read_joints(robot)

    state = np.concatenate([
        np.deg2rad(current),
        [float(gripper_state)]
    ]).astype(np.float32)

    observation = {
        "observation.images.front":
            to_tensor(Image.fromarray(rgb)).unsqueeze(0),

        "observation.state":
            torch.from_numpy(state).unsqueeze(0)
    }

    return observation, rgb, current


# ======================================================
# ACT INFERENCE: FULL CHUNK
# ======================================================

@torch.inference_mode()
def predict_chunk(policy, pre, post, observation):

    processed = pre(observation)

    raw = policy.predict_action_chunk(processed)

    actions = []

    for i in range(raw.shape[1]):

        action = post(
            raw[:, i, :]
        ).squeeze(0).cpu().numpy()

        actions.append(action)

    actions = np.asarray(actions)

    if actions.ndim != 2 or actions.shape[1] != 8:
        raise RuntimeError(f"Invalid chunk shape: {actions.shape}")

    if not np.isfinite(actions).all():
        raise RuntimeError("ACT produced invalid values")

    return actions


# ======================================================
# ACTION DECODE
# ======================================================

def decode_action(action):

    target = np.rad2deg(action[:4]).astype(float)

    gripper = int(action[4] >= 0.5)

    mode_scores = action[5:8]

    mode = int(np.argmax(mode_scores))

    return target, gripper, mode, mode_scores


# ======================================================
# ROBOT EXECUTION
# ======================================================

def execute_action(robot, action, gripper_state):

    target, gripper, mode, scores = decode_action(action)

    current = read_joints(robot)

    if np.any(target < JOINT_LOW) or np.any(target > JOINT_HIGH):
        raise RuntimeError(f"Target outside joint limits: {target}")

    if np.max(np.abs(target - current)) > MAX_JUMP:
        raise RuntimeError("Excessive joint target change")

    if scores[mode] < 0.8:
        raise RuntimeError(f"Ambiguous mode prediction: {scores}")

    # ----------------------------------------------
    # HORIZONTAL / VERTICAL
    # ----------------------------------------------

    if mode in (0, 1):

        expected_pitch = 0.0 if mode == 0 else -90.0

        if abs(pitch(current) - expected_pitch) > 3.0:
            raise RuntimeError(
                f"Cannot enter {MODE_NAMES[mode]}: "
                f"current pitch={pitch(current):.2f}"
            )

        if abs(pitch(target) - expected_pitch) > 4.0:
            raise RuntimeError(
                f"Target pitch mismatch: "
                f"target pitch={pitch(target):.2f}"
            )

        if mode == 0:
            robot.lock_horz(wait=True)
        else:
            robot.lock_vert(wait=True)

        start = read_joints(robot)

        delta = target[:3] - start[:3]

        n = max(
            1,
            int(np.ceil(np.max(np.abs(delta)) / WAYPOINT_DEG))
        )

        for k in range(1, n + 1):

            q = start[:3] + delta * (k / n)

            robot.degree_to(
                float(q[0]),
                float(q[1]),
                float(q[2]),
                None,
                SPEED,
                wait=True
            )

            actual = read_joints(robot)

            if np.max(np.abs(actual[:3] - q)) > JOINT_TOL:
                raise RuntimeError(
                    f"Joint tracking error: {actual[:3] - q}"
                )

    # ----------------------------------------------
    # WRIST
    # ----------------------------------------------

    elif mode == 2:

        if np.max(np.abs(current[:3] - target[:3])) > 3.0:
            raise RuntimeError(
                "Wrist target changes q1-q3"
            )

        robot.unlock(wait=True)

        start = read_joints(robot)

        delta = target[3] - start[3]

        n = max(
            1,
            int(np.ceil(abs(delta) / WAYPOINT_DEG))
        )

        for k in range(1, n + 1):

            q4 = start[3] + delta * (k / n)

            robot.degree_to(
                4,
                float(q4),
                SPEED,
                wait=True
            )

            actual = read_joints(robot)

            if abs(actual[3] - q4) > JOINT_TOL:
                raise RuntimeError(
                    f"Wrist tracking error: {actual[3] - q4:.2f}"
                )

    # ----------------------------------------------
    # GRIPPER
    # ----------------------------------------------

    if gripper != gripper_state:

        if gripper:
            print("GRIPPER CLOSE", flush=True)
            robot.close_gripper()
        else:
            print("GRIPPER OPEN", flush=True)
            robot.open_gripper()

        time.sleep(0.3)

    return gripper


# ======================================================
# OPENCV DISPLAY
# ======================================================

def show_image(rgb, step, current, target, mode, gripper):

    display = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    lines = [
        f"ACT ACTION {step}/{TOTAL_ACTIONS}",
        f"MODE: {MODE_NAMES[mode]} | GRIPPER: {gripper}",
        f"CURRENT: {np.round(current, 1)}",
        f"TARGET: {np.round(target, 1)}",
    ]

    for i, line in enumerate(lines):

        cv2.putText(
            display,
            line,
            (12, 30 + i * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if i == 0 else (255, 255, 255),
            2 if i == 0 else 1
        )

    cv2.imshow("RACCOONBOT ACT CHUNK-3", display)

    key = cv2.waitKey(1) & 0xFF

    return key != ord("q")


# ======================================================
# MAIN
# ======================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true"
    )

    parser.add_argument(
        "--gripper",
        type=int,
        choices=[0, 1],
        default=0
    )

    parser.add_argument(
        "--chunk-actions",
        type=int,
        default=CHUNK_ACTIONS,
        help="Number of predicted actions to execute before observing again (default: 1)."
    )

    args = parser.parse_args()
    if not 1 <= args.chunk_actions <= 10:
        parser.error("--chunk-actions must be between 1 and 10")

    policy, pre, post = load_model()

    output_dir = (
        ROOT
        / "act_debug"
        / time.strftime("chunk3_%Y%m%d_%H%M%S")
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        640,
        480,
        rs.format.rgb8,
        30
    )

    robot = None
    camera_started = False
    gripper_state = args.gripper

    action_count = 0
    inference_count = 0

    try:

        pipeline.start(config)
        camera_started = True

        robot = Raccoon()
        wait_until_ready()

        print("Initial Joints:", np.round(read_joints(robot), 2))
        print("TOTAL ACTIONS:", TOTAL_ACTIONS)
        print("ACTIONS PER CHUNK:", args.chunk_actions)
        print("EXECUTE:", args.execute)

        for _ in range(10):
            pipeline.wait_for_frames(timeout_ms=3000)

        while action_count < TOTAL_ACTIONS:

            inference_count += 1

            observation, rgb, current = get_observation(
                robot,
                pipeline,
                gripper_state
            )

            actions = predict_chunk(
                policy,
                pre,
                post,
                observation
            )

            print(
                f"\n========== INFERENCE {inference_count} =========="
            )

            np.savez(
                output_dir / f"chunk_{inference_count:03d}.npz",
                current=current,
                state=observation["observation.state"].numpy(),
                actions=actions,
                timestamp=time.time()
            )

            Image.fromarray(rgb).save(
                output_dir / f"rgb_{inference_count:03d}.png"
            )

            count = min(
                args.chunk_actions,
                TOTAL_ACTIONS - action_count,
                len(actions)
            )

            # Execute the selected prefix, then capture a fresh observation.
            for i in range(count):

                action_count += 1

                action = actions[i]

                target, gripper, mode, scores = decode_action(
                    action
                )

                current = read_joints(robot)

                print(
                    f"\n[{action_count:03d}/{TOTAL_ACTIONS}] "
                    f"CHUNK={inference_count} "
                    f"ACTION={i+1}/{count}"
                )

                print(
                    "MODE:",
                    MODE_NAMES[mode],
                    "GRIPPER:",
                    gripper
                )

                print("CURRENT:", np.round(current, 2))
                print("TARGET :", np.round(target, 2))
                print("SCORES :", np.round(scores, 4))

                if not show_image(
                    rgb,
                    action_count,
                    current,
                    target,
                    mode,
                    gripper
                ):
                    print("Stopped by user")
                    return

                if args.execute:

                    gripper_state = execute_action(
                        robot,
                        action,
                        gripper_state
                    )

            print(
                f"Chunk completed. "
                f"Executed actions: {action_count}/{TOTAL_ACTIONS}"
            )

        print("\nACT CHUNK INFERENCE COMPLETED")
        print("Saved:", output_dir)

    except KeyboardInterrupt:

        print("\nInterrupted")

    except Exception as e:

        print("\nCONTROL STOPPED:", repr(e))

    finally:

        if robot is not None:

            try:
                robot.stop()
            except Exception:
                pass

            try:
                robot.dispose()
            except Exception:
                pass

        if camera_started:
            pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

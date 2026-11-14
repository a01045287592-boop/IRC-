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


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("/home/min/Raccoonbot_Cafe_Clean")

CHECKPOINT = (
    ROOT / "checkpoint/054246/pretrained_model"
)

STEPS = 120
SPEED = 20

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODE_NAMES = ["HORIZONTAL", "VERTICAL", "WRIST"]

JOINT_LOW = np.array([-120., -90., -150., -105.])
JOINT_HIGH = np.array([120., 30., 0., 105.])


# ============================================================
# LOAD ACT
# ============================================================

def load_act():

    print("\n[1] Loading ACT Checkpoint")
    print("Checkpoint:", CHECKPOINT)

    policy = ACTPolicy.from_pretrained(
        str(CHECKPOINT),
        local_files_only=True
    ).to(DEVICE)

    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {
                "device": DEVICE
            }
        }
    )

    print("ACT READY:", DEVICE)

    return policy, preprocessor, postprocessor


# ============================================================
# ROBOT STATE
# ============================================================

def get_joint_state(robot):

    joint_deg = np.asarray(
        robot.encoder(-1),
        dtype=np.float64
    )

    if joint_deg.shape != (4,):
        raise RuntimeError("Invalid joint encoder shape")

    if not np.isfinite(joint_deg).all():
        raise RuntimeError("Invalid joint encoder values")

    return joint_deg


# ============================================================
# OBSERVATION
# ============================================================

def get_observation(robot, pipeline, gripper_state):

    frames = pipeline.wait_for_frames(
        timeout_ms=3000
    )

    color_frame = frames.get_color_frame()

    if not color_frame:
        raise RuntimeError("RealSense RGB unavailable")

    # RGB IMAGE
    rgb = np.asanyarray(
        color_frame.get_data()
    ).copy()

    # ROBOT ENCODER
    joint_deg = get_joint_state(robot)

    # degrees -> radians
    joint_rad = np.deg2rad(joint_deg)

    # State [5]
    state = np.concatenate([
        joint_rad,
        [float(gripper_state)]
    ]).astype(np.float32)

    # RGB [1,3,480,640]
    image_tensor = to_tensor(
        Image.fromarray(rgb)
    ).unsqueeze(0)

    # STATE [1,5]
    state_tensor = torch.from_numpy(
        state
    ).unsqueeze(0)

    observation = {
        "observation.images.front": image_tensor,
        "observation.state": state_tensor
    }

    return observation, rgb, joint_deg


# ============================================================
# ACT INFERENCE
# ============================================================

@torch.inference_mode()
def predict_action(
    policy,
    preprocessor,
    postprocessor,
    observation
):

    processed = preprocessor(observation)

    # ACT predicts 10 actions
    chunk = policy.predict_action_chunk(processed)

    # First action
    action = postprocessor(
        chunk[:, 0, :]
    ).squeeze(0).cpu().numpy()

    if action.shape != (8,):
        raise RuntimeError("Invalid ACT action shape")

    if not np.isfinite(action).all():
        raise RuntimeError("Non-finite ACT prediction")

    # joints: radians -> degrees
    target_deg = np.rad2deg(
        action[:4]
    )

    # gripper
    gripper = int(action[4] >= 0.5)

    # HORIZONTAL / VERTICAL / WRIST
    mode = int(np.argmax(
        action[5:8]
    ))

    return target_deg, gripper, mode, action


# ============================================================
# ROBOT MOVEMENT
# ============================================================

def get_pitch(q):

    return (
        90.0
        + q[1]
        + q[2]
        + q[3]
    )


def execute_action(
    robot,
    target,
    mode,
    gripper,
    gripper_state
):

    current = get_joint_state(robot)

    if np.any(target < JOINT_LOW):
        raise RuntimeError(
            f"Joint target below limit: {target}"
        )

    if np.any(target > JOINT_HIGH):
        raise RuntimeError(
            f"Joint target above limit: {target}"
        )

    # ------------------------------------
    # HORIZONTAL
    # ------------------------------------

    if mode == 0:

        if abs(get_pitch(current)) > 3.0:
            raise RuntimeError(
                "Cannot enter horizontal lock from current pitch"
            )

        if abs(get_pitch(target)) > 4.0:
            raise RuntimeError(
                "ACT horizontal target has inconsistent pitch"
            )

        robot.lock_horz(wait=True)

        robot.degree_to(
            float(target[0]),
            float(target[1]),
            float(target[2]),
            None,
            SPEED,
            wait=True
        )

    # ------------------------------------
    # VERTICAL
    # ------------------------------------

    elif mode == 1:

        if abs(get_pitch(current) + 90.0) > 3.0:
            raise RuntimeError(
                "Cannot enter vertical lock from current pitch"
            )

        if abs(get_pitch(target) + 90.0) > 4.0:
            raise RuntimeError(
                "ACT vertical target has inconsistent pitch"
            )

        robot.lock_vert(wait=True)

        robot.degree_to(
            float(target[0]),
            float(target[1]),
            float(target[2]),
            None,
            SPEED,
            wait=True
        )

    # ------------------------------------
    # WRIST
    # ------------------------------------

    elif mode == 2:

        if np.max(
            np.abs(current[:3] - target[:3])
        ) > 3.0:
            raise RuntimeError(
                "Wrist mode target changes q1-q3"
            )

        robot.unlock(wait=True)

        robot.degree_to(
            4,
            float(target[3]),
            SPEED,
            wait=True
        )

    # ------------------------------------
    # GRIPPER
    # ------------------------------------

    if gripper != gripper_state:

        if gripper == 1:

            print("GRIPPER CLOSE")

            robot.close_gripper()

        else:

            print("GRIPPER OPEN")

            robot.open_gripper()

        time.sleep(0.3)

    return gripper


# ============================================================
# OPENCV DISPLAY
# ============================================================

def show_observation(
    rgb,
    step,
    current,
    target,
    mode,
    gripper
):

    display = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    cv2.putText(
        display,
        f"ACT INFERENCE {step}/{STEPS}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"MODE: {MODE_NAMES[mode]}",
        (15, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"GRIPPER: {gripper}",
        (15, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"CURRENT: {np.round(current, 1)}",
        (15, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    cv2.putText(
        display,
        f"TARGET: {np.round(target, 1)}",
        (15, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "RACCOONBOT ACT 120 STEP",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    return key != ord("q")


# ============================================================
# MAIN
# ============================================================

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

    args = parser.parse_args()

    policy, preprocessor, postprocessor = load_act()

    # OUTPUT DIRECTORY

    output_dir = (
        ROOT
        / "act_debug"
        / time.strftime("run120_%Y%m%d_%H%M%S")
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # REALSENSE

    pipeline = rs.pipeline()

    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        rs.format.rgb8,
        30
    )

    robot = None
    camera_started = False
    gripper_state = args.gripper

    try:

        print("\n[2] Connecting RealSense")

        pipeline.start(config)
        camera_started = True

        print("\n[3] Connecting RaccoonBot")

        robot = Raccoon()
        wait_until_ready()

        print(
            "Initial Joints:",
            np.round(get_joint_state(robot), 2)
        )

        print("\n[4] Starting ACT Inference")

        print("STEPS:", STEPS)
        print("EXECUTE:", args.execute)

        if args.execute:
            print("WARNING: Physical robot commands enabled.")
            print("Emergency stop must be accessible.")

        # CAMERA WARMUP

        for _ in range(10):

            pipeline.wait_for_frames(
                timeout_ms=3000
            )

        # ========================================
        # 120 STEP INFERENCE
        # ========================================

        for step in range(STEPS):

            # Observation
            observation, rgb, current = get_observation(
                robot,
                pipeline,
                gripper_state
            )

            # ACT
            target, gripper, mode, action = predict_action(
                policy,
                preprocessor,
                postprocessor,
                observation
            )

            # Print
            print(
                f"\n[{step + 1:03d}/{STEPS}]",
                MODE_NAMES[mode],
                "GRIPPER:",
                gripper
            )

            print(
                "CURRENT:",
                np.round(current, 2)
            )

            print(
                "TARGET:",
                np.round(target, 2)
            )

            print(
                "MODE SCORES:",
                np.round(action[5:8], 4)
            )

            # OpenCV
            keep_running = show_observation(
                rgb,
                step + 1,
                current,
                target,
                mode,
                gripper
            )

            if not keep_running:
                print("Stopped by user")
                break

            # Save
            Image.fromarray(rgb).save(
                output_dir / f"rgb_{step:03d}.png"
            )

            np.savez(
                output_dir / f"step_{step:03d}.npz",
                state=observation[
                    "observation.state"
                ].numpy(),
                current=current,
                target=target,
                action=action,
                mode=mode,
                gripper=gripper,
                timestamp=time.time()
            )

            # REAL ROBOT EXECUTION
            if args.execute:

                gripper_state = execute_action(
                    robot,
                    target,
                    mode,
                    gripper,
                    gripper_state
                )

        print("\nACT INFERENCE FINISHED")
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

            robot.dispose()

        if camera_started:
            pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

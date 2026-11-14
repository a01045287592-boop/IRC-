"""Bounded ACT closed-loop control for RaccoonBot. Physical supervision required."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import pyrealsense2 as rs
from PIL import Image
from torchvision.transforms.functional import to_tensor
from roboid import Raccoon, wait_until_ready
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ROOT = Path('/home/min/Raccoonbot_Cafe_Clean')
CKPT = ROOT / 'checkpoint/072328/pretrained_model'
LOW = np.array([-120., -90., -150., -105.])
HIGH = np.array([120., 30., 0., 105.])
MAX_TARGET_JUMP = 65.0
MAX_WAYPOINT = 5.0
TOLERANCE = 3.0
SPEED = 20


def joints(robot):
    q = np.asarray(robot.encoder(-1), dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise RuntimeError('Invalid encoder values')
    if np.any(q < LOW) or np.any(q > HIGH):
        raise RuntimeError(f'Joint position outside limits: {q}')
    return q


def pitch(q):
    return 90.0 + q[1] + q[2] + q[3]


def assert_target(target, current):
    if target.shape != (4,) or not np.isfinite(target).all():
        raise RuntimeError(f'Invalid joint target: {target}')
    if np.any(target < LOW) or np.any(target > HIGH):
        raise RuntimeError(f'Target outside joint limits: {target}')
    jump = np.max(np.abs(target - current))
    if jump > MAX_TARGET_JUMP:
        raise RuntimeError(f'Target jump {jump:.2f} deg exceeds {MAX_TARGET_JUMP:.0f} deg')


def move_joint_mode(robot, current, target, mode):
    if mode == 0:
        expected = 0.0
    elif mode == 1:
        expected = -90.0
    else:
        raise RuntimeError('Invalid locked mode')

    # Servo locking q4 from an arbitrary pitch could cause unexpected movement.
    if abs(pitch(current) - expected) > 3.0:
        raise RuntimeError(f'Unsafe mode change: current pitch={pitch(current):.2f}, expected={expected:.1f}')
    if abs(pitch(target) - expected) > 4.0:
        raise RuntimeError(f'Predicted target inconsistent with mode: pitch={pitch(target):.2f}')

    if mode == 0:
        robot.lock_horz(wait=True)
    else:
        robot.lock_vert(wait=True)

    current = joints(robot)
    # q4 under lock is controlled by the robot; interpolate q1..q3.
    count = max(1, int(np.ceil(np.max(np.abs(target[:3] - current[:3])) / MAX_WAYPOINT)))
    for k in range(1, count + 1):
        a = current[:3] + (target[:3] - current[:3]) * (k / count)
        robot.degree_to(float(a[0]), float(a[1]), float(a[2]), None, SPEED, wait=True)
        actual = joints(robot)
        expected_q4 = (-90. if mode == 0 else -180.) - actual[1] - actual[2]
        if np.max(np.abs(actual[:3] - a)) > TOLERANCE or abs(actual[3] - expected_q4) > 4.0:
            raise RuntimeError(f'Waypoint tracking error: actual={actual}, target(q1..q3)={a}')
    actual = joints(robot)
    if np.max(np.abs(actual - target)) > 4.0:
        raise RuntimeError(f'Final tracking error: {actual - target}')
    return actual


def move_wrist(robot, current, target):
    if np.max(np.abs(current[:3] - target[:3])) > 3.0:
        raise RuntimeError('Wrist mode cannot modify q1..q3')
    robot.unlock(wait=True)
    current = joints(robot)
    count = max(1, int(np.ceil(abs(target[3] - current[3]) / MAX_WAYPOINT)))
    for k in range(1, count + 1):
        q4 = current[3] + (target[3] - current[3]) * (k / count)
        robot.degree_to(4, float(q4), SPEED, wait=True)
        actual = joints(robot)
        if abs(actual[3] - q4) > TOLERANCE:
            raise RuntimeError(f'Wrist tracking error: {actual[3]:.2f} vs {q4:.2f}')
    return joints(robot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    p.add_argument('--inspect-chunk', action='store_true')
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--gripper', type=int, choices=[0, 1], required=True)
    args = p.parse_args()
    if not 1 <= args.steps <= 100:
        p.error('--steps must be between 1 and 100')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = ACTPolicy.from_pretrained(str(CKPT), local_files_only=True).to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=CKPT,
        preprocessor_overrides={'device_processor': {'device': device}},
    )
    print('ACT ready:', CKPT, device)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    robot = None
    camera_started = False
    gripper_state = args.gripper
    stable_steps = 0
    debug_dir = ROOT / 'act_debug' / 'real_execution'
    debug_dir.mkdir(parents=True, exist_ok=True)
    try:
        pipeline.start(config)
        camera_started = True
        robot = Raccoon()
        wait_until_ready()
        print('Current:', np.round(joints(robot), 2))
        print('Mode and gripper state are based on last commands, not sensors.')
        if args.execute:
            if input('Type MOVE to enable physical motion: ').strip() != 'MOVE':
                return
        for _ in range(10):
            pipeline.wait_for_frames(timeout_ms=3000)

        for step in range(args.steps):
            frame = pipeline.wait_for_frames(timeout_ms=3000).get_color_frame()
            if not frame:
                raise RuntimeError('Color frame unavailable')
            rgb = np.asarray(frame.get_data()).copy()
            Image.fromarray(rgb).save(
                debug_dir / f'step_{step:03d}.png'
            )
            current = joints(robot)
            state = np.r_[np.deg2rad(current), float(gripper_state)].astype(np.float32)
            obs = {
                'observation.images.front': to_tensor(Image.fromarray(rgb)).unsqueeze(0),
                'observation.state': torch.from_numpy(state).unsqueeze(0),
            }
            policy.reset()
            with torch.inference_mode():
                raw = policy.predict_action_chunk(pre(obs))
                actions = np.stack([
                    post(raw[:, t, :]).squeeze(0).cpu().numpy()
                    for t in range(raw.shape[1])
                ])
                act = actions[0]

            if args.inspect_chunk:
                print("\n===== ACT ACTION CHUNK =====")
                for t, a in enumerate(actions):
                    q = np.rad2deg(a[:4])
                    mode_idx = int(np.argmax(a[5:8]))
                    print(
                        f"{t+1:02d} | "
                        f"q={np.round(q, 1)} | "
                        f"gripper={int(a[4] >= 0.5)} | "
                        f"mode={['H','V','W'][mode_idx]}"
                    )
                print("============================\n")
                break
            if act.shape != (8,) or not np.isfinite(act).all():
                raise RuntimeError('Invalid ACT output')
            target = np.rad2deg(act[:4].astype(float))
            scores = act[5:8]
            mode = int(np.argmax(scores))
            if scores[mode] < 0.8:
                raise RuntimeError(f'Ambiguous mode scores: {scores}')
            assert_target(target, current)
            want_gripper = int(act[4] >= 0.5)

            np.savez(
                debug_dir / f'action_{step:03d}.npz',
                current=current,
                target=target,
                chunk=actions,
                mode_scores=scores,
                gripper=want_gripper,
                timestamp=time.time(),
            )

            print(
                f'  Mode scores: {np.round(scores, 4)}',
                flush=True
            )

            print(f'[{step + 1}/{args.steps}] mode={"horizontal vertical wrist".split()[mode]}, '
                  f'gripper={want_gripper}, current={np.round(current, 1)}, target={np.round(target, 1)}', flush=True)

            if not args.execute:
                continue

            if mode == 2:
                actual = move_wrist(robot, current, target)
            else:
                actual = move_joint_mode(robot, current, target, mode)

            # Gripper changes are commanded only after the robot reaches the target.
            if want_gripper != gripper_state:
                if want_gripper:
                    robot.close_gripper()
                else:
                    robot.open_gripper()
                gripper_state = want_gripper
                time.sleep(0.3)

            reached = np.max(np.abs(actual - current)) < 0.3 and want_gripper == gripper_state
            stable_steps = stable_steps + 1 if reached else 0
            if stable_steps >= 8:
                print('No observed motion for 8 cycles; stopping to avoid indefinite loop.')
                break
        print('CONTROL LOOP COMPLETED')
    except KeyboardInterrupt:
        print('Operator interrupted.')
    except Exception as e:
        print('CONTROL STOPPED:', repr(e))
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception as e:
                print('Stop call failed:', repr(e))
            try:
                robot.dispose()
            except Exception:
                pass
        if camera_started:
            pipeline.stop()

if __name__ == '__main__':
    main()

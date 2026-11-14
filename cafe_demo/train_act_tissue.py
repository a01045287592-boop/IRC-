"""Collect MuJoCo tissue-sorting demonstrations and train a task-specific ACT policy."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from run_sorting import SortingDemo
from two_tray_scene import make_scene


IMAGE_SIZE = 96
FPS = 12
CHUNK_SIZE = 10
TASK_NAME = "pick_tissue_1_and_place_in_general_waste"


def collect_episodes(repo, output, count, seed):
    episodes_dir = output / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    if any(episodes_dir.glob("episode_*.npz")):
        raise FileExistsError(f"Episodes already exist in {episodes_dir}; use a new --out directory.")

    rng = random.Random(seed)
    np.random.seed(seed)
    metadata = []
    with tempfile.TemporaryDirectory(prefix="raccoon_act_tissue_scene_") as temp:
        scene, manifest = make_scene(repo, Path(temp))
        model = mujoco.MjModel.from_xml_path(str(scene))
        renderer = mujoco.Renderer(model, height=IMAGE_SIZE, width=IMAGE_SIZE)
        arm_joint_ids = [model.joint(f"Joint{i}").id for i in range(1, 5)]
        arm_qpos = np.asarray([model.jnt_qposadr[joint] for joint in arm_joint_ids])
        gripper_joint = model.joint("Gripper_L_Inner").id
        gripper_qpos = model.jnt_qposadr[gripper_joint]
        tissue_body = model.body("tissue_1").id
        tissue_joint = model.body_jntadr[tissue_body]
        tissue_qpos = model.jnt_qposadr[tissue_joint]
        base_tissue_xyz = model.qpos0[tissue_qpos:tissue_qpos + 3].copy()

        for episode_index in range(count):
            args = SimpleNamespace(
                fps=FPS, fast=True, out=Path(temp), only="tissue_1",
                max_items=1, skip_trays=True,
            )
            demo = SortingDemo(model, args, manifest)
            offset = np.asarray([rng.uniform(-0.004, 0.004), rng.uniform(-0.004, 0.004)])
            demo.data.qpos[tissue_qpos:tissue_qpos + 3] = base_tissue_xyz
            demo.data.qpos[tissue_qpos:tissue_qpos + 2] += offset
            mujoco.mj_forward(model, demo.data)

            frames, states, actions = [], [], []
            last_joint_target = demo.setpoint.copy()
            original_step = demo.step

            def capture_step(target, qvel=None):
                nonlocal last_joint_target
                last_joint_target = np.asarray(target, dtype=np.float32).copy()
                original_step(target, qvel)

            def capture_frame():
                renderer.update_scene(demo.data, camera="overview")
                rgb = renderer.render().copy()
                resized = Image.fromarray(rgb).resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
                )
                frames.append(np.asarray(resized, dtype=np.uint8).copy())
                states.append(np.r_[demo.data.qpos[arm_qpos], demo.data.qpos[gripper_qpos]])
                actions.append(np.r_[last_joint_target, demo.grip])

            demo.step = capture_step
            demo.output_frame = capture_frame
            error = None
            try:
                with redirect_stdout(io.StringIO()):
                    demo.run()
            except Exception as exc:
                error = str(exc)

            if error or not demo.result.get("success") or len(frames) < CHUNK_SIZE + 1:
                print(f"episode {episode_index:03d}: FAILED ({error or 'incomplete rollout'})")
                continue

            episode_path = episodes_dir / f"episode_{episode_index:03d}.npz"
            np.savez_compressed(
                episode_path,
                images=np.stack(frames),
                states=np.asarray(states, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32),
                times=np.asarray([i / FPS for i in range(len(frames))], dtype=np.float32),
                tissue_offset_m=offset.astype(np.float32),
            )
            metadata.append({
                "episode": episode_path.name,
                "task": TASK_NAME,
                "success": True,
                "frames": len(frames),
                "tissue_offset_m": offset.tolist(),
            })
            print(f"episode {episode_index:03d}: success, {len(frames)} frames, offset={offset.round(4).tolist()}")

        renderer.close()

    (output / "dataset_metadata.json").write_text(
        json.dumps({"task": TASK_NAME, "fps": FPS, "image_size": IMAGE_SIZE,
                    "episodes": metadata}, indent=2) + "\n",
        encoding="utf-8",
    )
    if len(metadata) < 2:
        raise RuntimeError(f"Need at least two successful episodes; got {len(metadata)}.")
    return episodes_dir, metadata


def load_episodes(episodes_dir):
    episodes = []
    for path in sorted(episodes_dir.glob("episode_*.npz")):
        with np.load(path) as data:
            episodes.append({key: data[key].copy() for key in ("images", "states", "actions")})
    return episodes


def make_batch(episodes, samples, state_low, state_high, action_low, action_high, device):
    images, states, actions, padding = [], [], [], []
    for episode_index, timestep in samples:
        episode = episodes[episode_index]
        length = len(episode["states"])
        images.append(episode["images"][timestep])
        state = episode["states"][timestep]
        states.append(2 * (state - state_low) / (state_high - state_low) - 1)

        action_chunk, action_padding = [], []
        for step in range(CHUNK_SIZE):
            action_index = timestep + step
            is_padding = action_index >= length
            action_index = min(action_index, length - 1)
            action = episode["actions"][action_index]
            action_chunk.append(2 * (action - action_low) / (action_high - action_low) - 1)
            action_padding.append(is_padding)
        actions.append(action_chunk)
        padding.append(action_padding)

    return {
        "observation.state": torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device),
        "observation.images.front": torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div_(255).to(device),
        "action": torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device),
        "action_is_pad": torch.as_tensor(np.asarray(padding), dtype=torch.bool, device=device),
    }


def train(episodes_dir, output, steps, batch_size, seed):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for this ACT training run.")

    episodes = load_episodes(episodes_dir)
    if len(episodes) < 2:
        raise RuntimeError("Need at least two successful episodes to split train/validation.")

    order = list(range(len(episodes)))
    random.Random(seed).shuffle(order)
    val_count = max(1, round(len(order) * 0.2))
    val_episodes = order[:val_count]
    train_episodes = order[val_count:]
    train_samples = [(i, t) for i in train_episodes for t in range(len(episodes[i]["states"]))]
    val_samples = [(i, t) for i in val_episodes for t in range(len(episodes[i]["states"]))]

    input_low = np.asarray([-2.1, -1.571, -2.618, -1.833, -0.85], dtype=np.float32)
    input_high = np.asarray([2.1, 0.524, 0.0, 1.833, 0.15701], dtype=np.float32)
    action_low, action_high = input_low.copy(), input_high.copy()
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (5,)),
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, IMAGE_SIZE, IMAGE_SIZE)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (5,))},
        device=device,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        dim_model=128,
        n_heads=4,
        dim_feedforward=512,
        n_encoder_layers=2,
        n_decoder_layers=1,
        use_vae=True,
        pretrained_backbone_weights=None,
    )
    policy = ACTPolicy(config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = random.Random(seed + 1)
    metrics = []

    for step in range(1, steps + 1):
        selected = [rng.choice(train_samples) for _ in range(batch_size)]
        batch = make_batch(episodes, selected, input_low, input_high, action_low, action_high, device)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss, terms = policy(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            record = {"step": step, "loss": float(loss.item()), **terms}
            metrics.append(record)
            print(json.dumps(record), flush=True)

    policy.eval()
    val_selected = rng.sample(val_samples, min(32, len(val_samples)))
    val_batch = make_batch(episodes, val_selected, input_low, input_high, action_low, action_high, device)
    with torch.inference_mode():
        val_loss, val_terms = policy(val_batch)
        prediction = policy.predict_action_chunk({
            "observation.state": val_batch["observation.state"][:1],
            "observation.images.front": val_batch["observation.images.front"][:1],
        })
    assert prediction.shape == (1, CHUNK_SIZE, 5) and bool(torch.isfinite(prediction).all())

    checkpoint = output / "act_checkpoint"
    policy.save_pretrained(checkpoint)
    report = {
        "task": TASK_NAME,
        "policy": "LeRobot ACT",
        "episodes": len(episodes),
        "train_episodes": len(train_episodes),
        "validation_episodes": len(val_episodes),
        "steps": steps,
        "final_train": metrics[-1],
        "validation_loss": float(val_loss.item()),
        "validation_terms": val_terms,
        "prediction_shape": list(prediction.shape),
        "device": torch.cuda.get_device_name(0),
        "instruction_conditioning": False,
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("TRAINING_REPORT:", json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "act_tissue_experiment")
    args = parser.parse_args()
    if args.episodes < 2 or args.steps < 1 or args.batch_size < 1:
        parser.error("episodes >= 2, steps >= 1, batch-size >= 1이어야 합니다.")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes_dir, metadata = collect_episodes(Path(__file__).resolve().parents[1], output, args.episodes, args.seed)
    print(f"Collected {len(metadata)} successful episodes at {episodes_dir}", flush=True)
    train(episodes_dir, output, args.steps, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
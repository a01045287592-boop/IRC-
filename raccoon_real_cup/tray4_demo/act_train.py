"""Train/check a LeRobot ACT policy on verified real tray4 episodes.

Unlike the MuJoCo demo trainer, this module uses the native robot schema:
5-D observation state and 8-D goal action (q1-q4, gripper, mode one-hot).
Episodes, never individual frames, are split into train/validation sets.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Allows --check on the robot-only environment.
    torch = None
    class Dataset:  # pragma: no cover - training environment provides torch
        pass
try:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
except ImportError:  # --check only needs the native file schema.
    FeatureType = PolicyFeature = ACTConfig = ACTPolicy = None

from act_export import export_episode
from review import audit, chunk

IMAGE_SIZE = 96
STATE_DIM = 5
ACTION_DIM = 8

LIGHTING_CONFIG = dict(probability=0.8, brightness=[0.8, 1.2],
                       contrast=[0.85, 1.15], saturation=[0.9, 1.1],
                       red_blue_gain=[0.95, 1.05])


def augment_lighting(image, rng):
    """Photometric changes only; preserve object positions and action labels."""
    if rng.random() >= LIGHTING_CONFIG['probability']:
        return image
    for enhancer, key in ((ImageEnhance.Brightness, 'brightness'),
                          (ImageEnhance.Contrast, 'contrast'),
                          (ImageEnhance.Color, 'saturation')):
        image = enhancer(image).enhance(rng.uniform(*LIGHTING_CONFIG[key]))
    gain = rng.uniform(*LIGHTING_CONFIG['red_blue_gain'])
    pixels = np.asarray(image, dtype=np.float32) * np.array([gain, 1., 2.-gain])
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def decode_native_action(action):
    """Convert one predicted native action to the robot controller vocabulary."""
    a = np.asarray(action, dtype=np.float32)
    if a.shape != (ACTION_DIM,) or not np.isfinite(a).all():
        raise ValueError('expected one finite 8-D native action')
    modes = ('horizontal', 'vertical', 'wrist')
    mode = modes[int(np.argmax(a[5:8]))]
    return {
        'goal_q_rad': a[:4].tolist(),
        'gripper_closed': bool(a[4] >= 0.5),
        'mode': mode,
    }


class Tray4ACTDataset(Dataset):
    def __init__(self, episode_dirs, horizon=10, lighting_aug=False, seed=0, phase_spec=None):
        self.phase_spec = phase_spec
        self.horizon = int(horizon)
        self.lighting_aug = lighting_aug
        self.augmentation_rng = random.Random(seed)
        self.episodes = []
        self.samples = []
        for raw in episode_dirs:
            folder = Path(raw).expanduser().resolve()
            schema = json.loads((folder / 'training' / 'schema.json').read_text())
            partial = schema.get('format') == 'raccoon_partial_act_v1'
            report = audit(folder, partial=partial)
            if not report['data_checks_passed']:
                raise ValueError(f'{folder.name}: audit failed: {report["errors"][:3]}')
            path = folder / 'training' / 'episode.npz'
            if not path.exists():
                raise FileNotFoundError(path)
            data = np.load(path, allow_pickle=False)
            n = len(data['state'])
            if not n or data['state'].shape != (n, STATE_DIM) or data['action'].shape != (n, ACTION_DIM):
                raise ValueError(f'{folder.name}: invalid state/action shapes')
            if not np.isfinite(data['state']).all() or not np.isfinite(data['action']).all():
                raise ValueError(f'{folder.name}: non-finite state/action')
            for key in ('image', 'segment_id', 'object_id', 'frame_index'):
                if data[key].shape != (n,):
                    raise ValueError(f'{folder.name}: invalid {key} shape')
            if partial and (set(data['object_id']) != {schema['object_id']} or len(set(data['segment_id'])) != 1):
                raise ValueError(f'{folder.name}: partial object/segment mismatch')
            ep = {'folder': folder, 'data': data}
            if phase_spec:
                steps = {r['frame_index']: r for r in map(json.loads, (folder/'steps.jsonl').read_text().splitlines())}
                ep['phases'] = [steps[int(f)]['phase'] for f in data['frame_index']]
                from act_phase import feature
                for name in set(ep['phases']): feature(name, phase_spec)
            index = len(self.episodes)
            self.episodes.append(ep)
            self.samples.extend((index, i) for i in range(len(data['state'])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ep_index, t = self.samples[index]
        ep = self.episodes[ep_index]
        folder, data = ep['folder'], ep['data']
        with Image.open(folder / str(data['image'][t])) as source:
            image = source.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        if self.lighting_aug:
            image = augment_lighting(image, self.augmentation_rng)
        image = np.asarray(image, dtype=np.float32) / 255.0
        actions, padding = chunk(data, t, self.horizon)
        state = np.asarray(data['state'][t], np.float32).copy()
        if self.phase_spec:
            from act_phase import feature
            name = ep['phases'][t]
            state = np.concatenate([state, feature(name, self.phase_spec)])
            for k in range(1, self.horizon):
                if t+k >= len(ep['phases']) or ep['phases'][t+k] != name:
                    padding[k:] = True
                    actions[k:] = 0
                    break
        return {
            'observation.images.front': torch.from_numpy(image.transpose(2, 0, 1).copy()),
            'observation.state': torch.from_numpy(state),
            'action': torch.from_numpy(np.asarray(actions, np.float32)),
            'action_is_pad': torch.from_numpy(np.asarray(padding, bool)),
        }


def _batch(dataset, indices, device):
    rows = [dataset[i] for i in indices]
    return {k: torch.stack([r[k] for r in rows]).to(device) for k in rows[0]}


def _policy(device, horizon, state_dim=STATE_DIM):
    config = ACTConfig(
        input_features={
            'observation.state': PolicyFeature(FeatureType.STATE, (state_dim,)),
            'observation.images.front': PolicyFeature(FeatureType.VISUAL, (3, IMAGE_SIZE, IMAGE_SIZE)),
        },
        output_features={'action': PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))},
        device=device, chunk_size=horizon, n_action_steps=horizon,
        dim_model=128, n_heads=4, dim_feedforward=512,
        n_encoder_layers=2, n_decoder_layers=1, use_vae=True,
        pretrained_backbone_weights=None,
    )
    return ACTPolicy(config).to(device)


def check_dataset(episode_dirs, horizon=10):
    ds = Tray4ACTDataset(episode_dirs, horizon)
    if not len(ds):
        raise ValueError('no valid samples')
    ep, t = ds.samples[0]
    data = ds.episodes[ep]['data']
    state = np.asarray(data['state'][t], np.float32)
    actions, padding = chunk(data, t, horizon)
    if state.shape != (STATE_DIM,):
        raise ValueError('invalid state shape')
    if np.asarray(actions).shape != (horizon, ACTION_DIM):
        raise ValueError('invalid action chunk shape')
    if not bool(np.isfinite(state).all() and np.isfinite(actions).all()):
        raise ValueError('non-finite training value')
    return {'episodes': len(ds.episodes), 'samples': len(ds), 'state_shape': [STATE_DIM],
            'action_shape': [horizon, ACTION_DIM], 'image_shape': [3, IMAGE_SIZE, IMAGE_SIZE]}


def train(episode_dirs, output, steps=1000, batch_size=16, horizon=10, seed=0, lighting_aug=False, phase_spec=None):
    if torch is None or ACTPolicy is None:
        raise RuntimeError('PyTorch and LeRobot are required; use the project .venv/bin/python.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for ACT training; use --check for CPU validation.')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dirs = [Path(x).expanduser().resolve() for x in episode_dirs]
    if len(dirs) < 2:
        raise ValueError('need at least two successful episodes for episode-level validation split')
    rng = random.Random(seed)
    order = list(range(len(dirs))); rng.shuffle(order)
    val_count = max(1, round(len(order) * .2))
    val_dirs = [dirs[i] for i in order[:val_count]]
    train_dirs = [dirs[i] for i in order[val_count:]]
    train_ds = Tray4ACTDataset(train_dirs, horizon, lighting_aug=lighting_aug, seed=seed, phase_spec=phase_spec)
    val_ds = Tray4ACTDataset(val_dirs, horizon, phase_spec=phase_spec)
    device = 'cuda'
    policy = _policy(device, horizon, STATE_DIM + len(phase_spec or []))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-4)
    metrics = []
    for step in range(1, steps + 1):
        selected = [rng.randrange(len(train_ds)) for _ in range(batch_size)]
        batch = _batch(train_ds, selected, device)
        policy.train(); optimizer.zero_grad(set_to_none=True)
        loss, terms = policy(batch); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            rec = {'step': step, 'loss': float(loss.item()), **terms}; metrics.append(rec); print(json.dumps(rec), flush=True)
    policy.eval(); selected = [rng.randrange(len(val_ds)) for _ in range(min(32, len(val_ds)))]
    with torch.inference_mode():
        val_batch = _batch(val_ds, selected, device)
        val_loss, val_terms = policy(val_batch)
        pred = policy.predict_action_chunk({k: val_batch[k] for k in ('observation.state', 'observation.images.front')})
    if tuple(pred.shape) != (len(selected), horizon, ACTION_DIM):
        raise RuntimeError(f'unexpected policy output shape: {tuple(pred.shape)}')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'act_tray4_policy'; policy.save_pretrained(checkpoint)
    (output / 'training_report.json').write_text(json.dumps({
        'dataset': 'tray4_native_goal', 'episodes': len(dirs), 'train_episodes': len(train_dirs),
        'validation_episodes': len(val_dirs), 'steps': steps, 'horizon': horizon,
        'phase_spec': phase_spec, 'state_dim': STATE_DIM + len(phase_spec or []), 'action_dim': ACTION_DIM, 'validation_loss': float(val_loss.item()),
        'validation_terms': val_terms, 'prediction_shape': list(pred.shape),
        'seed': seed, 'lighting_augmentation': LIGHTING_CONFIG if lighting_aug else None,
        'train_episode_paths': [str(p) for p in train_dirs],
        'validation_episode_paths': [str(p) for p in val_dirs],
    }, indent=2) + '\n', encoding='utf-8')
    (output / 'training_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('episodes', nargs='+', type=Path)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--lighting-aug', action='store_true', help='Apply mild lighting augmentation to training images only')
    ap.add_argument('--exclude', type=Path, action='append', default=[],
                    help='Exact episode folder to exclude (repeatable); files are preserved')
    ap.add_argument('--out', type=Path, default=Path('raccoon_dataset/tray4_act_training'))
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--horizon', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    excluded = {p.expanduser().resolve() for p in args.exclude}
    supplied = {p.expanduser().resolve() for p in args.episodes}
    if excluded - supplied:
        ap.error('--exclude must match a supplied episode folder')
    args.episodes = [p for p in args.episodes if p.expanduser().resolve() not in excluded]
    if not args.episodes:
        ap.error('no episodes remain after exclusions')
    if args.check:
        print(json.dumps(check_dataset(args.episodes, args.horizon), indent=2))
    else:
        train(args.episodes, args.out, args.steps, args.batch_size, args.horizon, args.seed, args.lighting_aug)

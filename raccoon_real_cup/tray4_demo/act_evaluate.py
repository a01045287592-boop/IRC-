"""Offline held-out evaluation; never connects to the robot."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from act_train import ACTPolicy, Tray4ACTDataset, _batch


def evaluate(run, per_episode=32, device='cpu'):
    if per_episode < 1:
        raise ValueError('per_episode must be positive')
    report = json.loads((run / 'training_report.json').read_text())
    policy = ACTPolicy.from_pretrained(str(run / 'act_tray4_policy')).to(device).eval()
    results = []
    mode_errors = {name: [] for name in ('horizontal', 'vertical', 'wrist')}
    first_errors = []
    for folder in report['validation_episode_paths']:
        ds = Tray4ACTDataset([folder], report['horizon'], phase_spec=report.get('phase_spec'))
        indices = np.linspace(0, len(ds)-1, min(per_episode, len(ds)), dtype=int).tolist()
        errors, grip, mode = [], [], []
        with torch.inference_mode():
            for start in range(0, len(indices), 16):
                batch = _batch(ds, indices[start:start+16], device)
                pred = policy.predict_action_chunk({k: batch[k] for k in
                    ('observation.state', 'observation.images.front')})
                if not torch.isfinite(pred).all():
                    raise ValueError(f'non-finite prediction: {folder}')
                valid = ~batch['action_is_pad']
                p, target = pred[valid], batch['action'][valid]
                first_errors.extend(torch.rad2deg((pred[:, 0, :4]-batch['action'][:, 0, :4]).abs()).cpu().tolist())
                for m, name in enumerate(mode_errors):
                    selected = target[:, 5:8].argmax(-1) == m
                    mode_errors[name].extend(torch.rad2deg((p[selected, :4]-target[selected, :4]).abs()).cpu().tolist())
                errors.extend(torch.rad2deg((p[:, :4]-target[:, :4]).abs()).cpu().numpy().tolist())
                grip.extend(((p[:, 4] >= .5) == (target[:, 4] >= .5)).cpu().tolist())
                mode.extend((p[:, 5:8].argmax(-1) == target[:, 5:8].argmax(-1)).cpu().tolist())
        row = dict(episode=folder, sampled_frames=len(indices),
                   joint_mae_deg=np.mean(errors, axis=0).tolist(),
                   joint_p95_deg=np.percentile(errors, 95, axis=0).tolist(),
                   gripper_accuracy=float(np.mean(grip)), mode_accuracy=float(np.mean(mode)))
        results.append(row)
        print(json.dumps(row), flush=True)
    output = dict(description='Offline inference on evenly spaced held-out frames; not a rollout success rate.',
                  episodes=len(results), sampled_frames=sum(r['sampled_frames'] for r in results),
                  episode_mean_joint_mae_deg=np.mean([r['joint_mae_deg'] for r in results], axis=0).tolist(),
                  episode_mean_gripper_accuracy=float(np.mean([r['gripper_accuracy'] for r in results])),
                  episode_mean_mode_accuracy=float(np.mean([r['mode_accuracy'] for r in results])),
                  first_action_joint_mae_deg=np.mean(first_errors, axis=0).tolist(),
                  by_reference_mode={name: dict(count=len(rows),
                      joint_mae_deg=np.mean(rows, axis=0).tolist() if rows else None,
                      commanded_joints=[4] if name == 'wrist' else [1, 2, 3])
                      for name, rows in mode_errors.items()},
                  results=results)
    (run / 'offline_evaluation.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--per-episode', type=int, default=32)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.run, args.per_episode, args.device)

"""Export verified tray episodes as ACT action-chunk samples.

Images remain referenced by path so export does not duplicate thousands of PNGs.
The native robot action keeps all eight values, including the required mode.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from core import atomic_json
from review import audit, chunk


def export_episode(folder, horizon=10, output=None):
    folder = Path(folder).expanduser().resolve()
    report = audit(folder)
    if not report['data_checks_passed']:
        raise ValueError('audit 실패: ' + ', '.join(report['errors'][:5]))
    source = np.load(folder / 'training' / 'episode.npz', allow_pickle=False)
    n = len(source['state'])
    actions = np.empty((n, horizon, source['action'].shape[1]), dtype=np.float32)
    padding = np.ones((n, horizon), dtype=bool)
    for i in range(n):
        actions[i], padding[i] = chunk(source, i, horizon)
    out = Path(output) if output else folder / 'training' / f'act_h{horizon}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + '.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(
        handle,
        image=np.asarray(source['image']),
        observation_state=np.asarray(source['state'], np.float32),
        action=actions,
        action_is_pad=padding,
        segment_id=np.asarray(source['segment_id']),
        object_id=np.asarray(source['object_id']),
        frame_index=np.asarray(source['frame_index']),
        )
    temp.replace(out)
    atomic_json(out.with_suffix('.schema.json'), {
        'format': 'raccoon_act_action_chunks_v1',
        'episode': folder.name,
        'image_root': str(folder),
        'observation_state_dim': 5,
        'action_dim': 8,
        'action_horizon': horizon,
        'action_semantics': 'native robot goal q1-q4, gripper command, horizontal/vertical/wrist mode one-hot',
        'split_rule': 'split complete episodes, never individual frames',
    })
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=Path)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(export_episode(args.episode, args.horizon, args.output))

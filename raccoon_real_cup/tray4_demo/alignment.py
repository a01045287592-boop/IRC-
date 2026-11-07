"""Create a transparent per-frame index without inventing sensor readings."""
import bisect
import json
import math
import os
from pathlib import Path
from core import jsonl, read_json


def write_frame_alignment(folder):
    folder = Path(folder)
    frames = list(jsonl(folder / 'frames.jsonl'))
    steps = {r['frame_index']: r for r in jsonl(folder / 'steps.jsonl')}
    events = list(jsonl(folder / 'events.jsonl'))
    joints = sorted((r for r in events if r['event'] == 'joint_observation'), key=lambda r: r['read_end_ns'])
    commands = sorted((r for r in events if r['event'] == 'command'), key=lambda r: r['call_end_ns'])
    jt = [r['read_end_ns'] for r in joints]
    ct = [r['call_end_ns'] for r in commands]
    annotations_path = folder / 'annotations.jsonl'
    annotations = {r['frame_index']: r for r in jsonl(annotations_path)} if annotations_path.exists() else {}
    max_age = read_json(folder / 'config.json')['max_joint_age_ms']
    counts = {'frames': 0, 'complete_state': 0, 'missing_joint': 0, 'unknown_gripper': 0,
              'active_steps': 0, 'missing_annotation': 0}
    tmp = folder / 'frame_alignment.jsonl.tmp'
    try:
        with tmp.open('w', encoding='utf-8') as out:
            for frame in frames:
                idx = frame['frame_index']; t = frame['host_arrival_ns']; step = steps.get(idx)
                ji = bisect.bisect_right(jt, t) - 1
                ci = bisect.bisect_right(ct, t) - 1
                joint = joints[ji] if ji >= 0 else None
                age = (t - joint['read_end_ns']) / 1e6 if joint else None
                if joint and age <= max_age:
                    state = [math.radians(x) for x in joint['qpos_deg']]
                else:
                    state = [None] * 4
                    counts['missing_joint'] += 1
                grip = commands[ci]['gripper_command'] if ci >= 0 else None
                if step:
                    state = [v if ok else None for v, ok in zip(step['observation_state'][:4], step['observation_state_mask'][:4])]
                    grip = step['observation_state'][4] if step['observation_state_mask'][4] else None
                    age = (t - step['joint_read_ns']) / 1e6
                if grip is None: counts['unknown_gripper'] += 1
                state.append(grip)
                if all(x is not None for x in state): counts['complete_state'] += 1
                if step: counts['active_steps'] += 1
                ann = annotations.get(idx)
                if ann is None: counts['missing_annotation'] += 1
                row = dict(frame_index=idx, image=frame['file'], frame_host_ns=t,
                           observation_state=state, joint_read_ns=step['joint_read_ns'] if step else (joint['read_end_ns'] if joint else None),
                           host_joint_read_age_ms=age, gripper_source='last_command_not_measured',
                           joint_source='SDK_cache_sensor_timestamp_unavailable',
                           active_step=bool(step), action=step['action'] if step else None,
                           segment_id=step['segment_id'] if step else None,
                           object_id=step['object_id'] if step else None,
                           phase=step['phase'] if step else None,
                           objects=ann['objects'] if ann else None)
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                counts['frames'] += 1
        os.replace(tmp, folder / 'frame_alignment.jsonl')
    finally:
        if tmp.exists(): tmp.unlink()
    return counts

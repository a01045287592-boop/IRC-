"""Join recorded observations to causal packet states. No invented ACT actions."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path

from tray_core import atomic_json


def lines(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def build(out):
    meta=json.loads((out/'metadata.json').read_text())
    events=lines(out/'events.jsonl')
    observations=lines(out/'observations.jsonl')
    packets=sorted((e for e in events if e['kind']=='joint_packet'),key=lambda e:e['read_end_ns'])
    times=[e['read_end_ns'] for e in packets]
    rows=[]; reasons={}; segment=-1; last_valid=None
    t0=observations[0]['host_arrival_ns'] if observations else 0
    for obs in observations:
        t=obs['host_arrival_ns']; i=bisect_right(times,t)-1
        why=[]; q=None; age=None
        if i<0:
            why.append('joint_not_available')
        else:
            age=(t-times[i])/1e9; q=packets[i]['joint_deg']
            if age>.15:why.append('joint_stale')
        grip=obs['state']['gripper_command']
        if grip not in ('open','close'):why.append('gripper_command_unknown')
        if not obs['input_valid']:why.append('target_not_visible_before_grasp')
        if not (out/obs['image']).is_file():why.append('observation_image_missing')
        if why:
            last_valid=None
        else:
            if last_valid is None or t-last_valid>150_000_000:segment+=1
            last_valid=t
        for r in why:reasons[r]=reasons.get(r,0)+1
        rows.append(dict(frame_index=obs['frame_index'],timestamp=(t-t0)/1e9,
            host_arrival_ns=t, image=obs['image'],raw_image=obs['raw_image'],
            object_id=obs['state']['object_id'],rank=obs['state']['rank'],phase=obs['state']['phase'],
            joint_deg=q,joint_age_s=age,gripper_command=grip,
            gripper_command_value={'open':0,'close':1}.get(grip),
            observation_valid=not why,exclusion_reasons=why,segment_id=segment if not why else None,
            action=None))
    path=out/'samples.jsonl'
    with path.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary=dict(frames=len(rows),valid_observations=sum(r['observation_valid'] for r in rows),
        exclusions=reasons,status=meta['status'],success=meta['success'],
        actual_gripper_state_available=False,actions_available=False,
        note='동기화 관찰 검토용. gripper는 명령값이며 측정 폭이 아님. ACT action 라벨과 성공 검수는 별도 필요.')
    atomic_json(out/'manifest_summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path)
    a=p.parse_args();print(json.dumps(build(a.episode),ensure_ascii=False,indent=2))

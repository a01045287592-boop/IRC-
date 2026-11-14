#!/usr/bin/env python3
"""Create a review-only JSONL preview from the existing Raccoon raw recorder.

Does not move a robot, alter source data, detect targets or create LeRobot data.
The action field is an explicitly marked future measured-position proxy, not a
recorded action. See the printed summary and README in the output directory.
"""
import argparse
import bisect
import json
import math
import os
from collections import Counter
from pathlib import Path


def read_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(path):
    events = []
    with path.open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f'{path}:{n}: {e}') from e
    return events


def ns(event, *keys):
    for key in keys:
        value = event.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def valid_q(q):
    return isinstance(q, (list, tuple)) and len(q) == 4 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in q
    )


def before(times, values, t):
    i = bisect.bisect_right(times, t) - 1
    return None if i < 0 else (times[i], values[i])


def nearest_future(times, values, base, target, tolerance_ns):
    i = bisect.bisect_left(times, target)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(times) and times[j] > base]
    if not candidates:
        return None
    j = min(candidates, key=lambda x: abs(times[x] - target))
    return (times[j], values[j]) if abs(times[j] - target) <= tolerance_ns else None


def command_time(e):
    if e.get('command') in ('open_gripper', 'close_gripper'):
        return ns(e, 'call_acquired_ns', 'call_start_ns', 'host_monotonic_ns')
    return ns(e, 'call_end_ns', 'host_monotonic_ns')


def main():
    p = argparse.ArgumentParser(description='기존 raw episode를 시간 정렬한 검토용 JSONL로 변환 (로봇 움직이지 않음)')
    p.add_argument('episode', type=Path, help='events.jsonl이 있는 raw episode 폴더')
    p.add_argument('--target-class', required=True, choices=['cup', 'plate', 'tissue', 'ring', 'fork'])
    p.add_argument('--target-id', required=True, help='시연에서 실제로 잡은 대상의 이름. 예: cup_1')
    p.add_argument('--horizon-ms', type=float, default=100.0, help='미래 관측 관절각까지의 시간. 기본 100ms')
    p.add_argument('--max-age-ms', type=float, default=200.0, help='현재 관절값 허용 최대 경과 시간')
    p.add_argument('--future-tolerance-ms', type=float, default=60.0, help='미래 관절 읽기 시간 오차 허용치')
    args = p.parse_args()
    for value in (args.horizon_ms, args.max_age_ms, args.future_tolerance_ms):
        if not math.isfinite(value) or value <= 0:
            p.error('시간 설정은 양수여야 합니다.')
    episode = args.episode.expanduser().resolve()
    source = episode / 'events.jsonl'
    if not source.is_file():
        p.error(f'events.jsonl을 찾을 수 없습니다: {source}')
    meta_path = episode / 'metadata.json'
    meta = read_json(meta_path) if meta_path.exists() else {}
    events = read_jsonl(source)
    frames, joints, commands = [], [], []
    for e in events:
        kind = e.get('kind')
        if kind == 'camera_frame':
            t = ns(e, 'host_arrival_ns', 'host_monotonic_ns')
            if t is not None and isinstance(e.get('file'), str):
                frames.append((t, e))
        elif kind == 'joint_packet':
            t = ns(e, 'read_end_ns', 'host_monotonic_ns')
            if t is not None and valid_q(e.get('joint_deg')):
                joints.append((t, [float(v) for v in e['joint_deg']]))
        elif kind == 'robot_command':
            t = command_time(e)
            if t is not None:
                commands.append((t, e))
    frames.sort(key=lambda x:x[0]); joints.sort(key=lambda x:x[0]); commands.sort(key=lambda x:x[0])
    if not frames or not joints:
        raise SystemExit(f'변환 불가: 카메라 이벤트 {len(frames)}개, 관절 읽기 {len(joints)}개')
    jt, jv = [x[0] for x in joints], [x[1] for x in joints]
    grips = [(t, 0.0 if e['command'] == 'open_gripper' else 1.0)
             for t,e in commands if e.get('command') in ('open_gripper','close_gripper')]
    gt, gv = [x[0] for x in grips], [x[1] for x in grips]
    ct = [t for t,_ in commands]
    safety_stop = [t for t,e in commands if e.get('command') in ('stop','dispose')]
    output = episode / 'act_preview'
    try:
        output.mkdir()
    except FileExistsError:
        raise SystemExit(f'출력 폴더가 이미 있습니다. 기존 파일은 덮어쓰지 않습니다: {output}')
    horizon = round(args.horizon_ms * 1e6)
    max_age = round(args.max_age_ms * 1e6)
    tolerance = round(args.future_tolerance_ms * 1e6)
    start = frames[0][0]
    counts = Counter()
    with (output / 'samples.jsonl').open('x', encoding='utf-8') as f:
        for t, frame in frames:
            image = (episode / frame['file']).resolve()
            if not image.is_relative_to(episode):
                raise ValueError('episode 외부 이미지 경로가 있어 중단합니다.')
            now = before(jt, jv, t)
            future = nearest_future(jt, jv, t, t+horizon, tolerance)
            grip = before(gt, gv, t)
            future_grip = before(gt, gv, t+horizon)
            reasons=[]
            if not image.is_file(): reasons.append('image_missing')
            if not now: reasons.append('joint_not_yet_available')
            elif t-now[0] > max_age: reasons.append('joint_stale')
            if not future: reasons.append('future_joint_missing')
            if not grip: reasons.append('gripper_command_unknown')
            if not future_grip: reasons.append('future_gripper_command_unknown')
            k = bisect.bisect_right(safety_stop, t)
            if k < len(safety_stop) and safety_stop[k] <= t+horizon:
                reasons.append('stop_or_dispose_in_horizon')
            before_commands = commands[max(0,bisect.bisect_left(ct,t)-1):bisect.bisect_right(ct,t)]
            interval_commands = commands[bisect.bisect_right(ct,t):bisect.bisect_right(ct,t+horizon)]
            def short_command(item):
                tm,e=item
                return {'relative_s': round((tm-start)/1e9,6), 'command':e.get('command'), 'args':e.get('args'), 'kwargs':e.get('kwargs')}
            row = {
                'timestamp':round((t-start)/1e9,6),
                'frame_index':frame.get('frame_index'),
                'image':os.path.relpath(image,output),
                'observation_state': (now[1] + [grip[1] if grip else None]) if now else None,
                'action': (future[1] + [future_grip[1] if future_grip else None]) if future else None,
                'action_source':'future_measured_joint_proxy_not_sdk_command',
                'target_class':args.target_class, 'target_id':args.target_id,
                'target_source':'user_supplied_episode_label_not_yolo',
                'target_bbox':None, 'target_overlay':False,
                'gripper_source':'last_issued_command_not_measured',
                'observation_age_ms':round((t-now[0])/1e6,3) if now else None,
                'future_joint_offset_ms':round((future[0]-t)/1e6,3) if future else None,
                'last_command':short_command(before_commands[-1]) if before_commands else None,
                'commands_in_horizon':[short_command(c) for c in interval_commands],
                'sync_ok':not reasons, 'reasons':reasons,
            }
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            counts['frames']+=1
            counts['sync_ok']+=not reasons
            counts.update(reasons)
    summary={'episode':str(episode),'source_status':meta.get('status','unknown'),
             'source_success':meta.get('success','unknown'), 'target_class':args.target_class,
             'target_id':args.target_id, 'horizon_ms':args.horizon_ms, 'counts':dict(counts),
             'camera_events':len(frames),'joint_events':len(joints),'command_events':len(commands),
             'warning':'PREVIEW ONLY: target labels are manual, images have no overlays; future measured joint is a proxy, not actual command; success unverified; no LeRobot conversion.'}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'README.txt').write_text('검토용 결과입니다. 학습 가능한 LeRobot 데이터가 아닙니다.\n'
        '원본은 변경하지 않습니다. image는 이 폴더 기준 상대 경로입니다.\n'
        'action 앞 네 값은 실제 보낸 목표 명령이 아니라 horizon 후 측정 관절각입니다.\n'
        '그리퍼는 측정된 벌림이 아니라 마지막 명령입니다. target은 수동 기입입니다.\n'
        'YOLO 선택, 타깃 추적 및 초록 표시 이미지 생성은 포함되지 않습니다.\n'
        'sync_ok는 동기화 검사만 의미하며 성공/파지/에피소드 품질을 보증하지 않습니다.\n'
        '기록이 중단됐거나 empty-run이면 성공 데모로 쓰지 마세요.\n',encoding='utf-8')
    print('결과:',output)
    print('전체 프레임:',counts['frames'],'동기화 통과:',counts['sync_ok'])
    for name,count in counts.items():
        if name not in ('frames','sync_ok'): print(f'  {name}: {count}')
    print('에피소드 상태:',summary['source_status'],'성공:',summary['source_success'])
    print('주의: 검토용이며 ACT 학습 데이터가 아닙니다. YOLO 표시 없음, 행동은 미래 측정 관절값입니다.')
    print('확인: cat', output/'summary.json')
    print('확인: head -n 3', output/'samples.jsonl')

if __name__ == '__main__':
    main()

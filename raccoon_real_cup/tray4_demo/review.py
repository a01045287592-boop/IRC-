"""Audit alignment/labels; export custom ACT native-goal chunks without pause leakage."""
import argparse
import collections
import math
import statistics
from pathlib import Path
from core import ROLES,SCHEMA,read_json,jsonl,atomic_json,decode_action,encode_action

def rate(times):
    if len(times)<2:return dict(count=len(times),hz=None,max_gap_ms=None)
    gaps=[(b-a)/1e6 for a,b in zip(times,times[1:])]
    if any(x<=0 for x in gaps):return dict(count=len(times),hz=None,max_gap_ms=max(gaps),nonmonotonic=True)
    return dict(count=len(times),hz=(len(times)-1)*1e9/(times[-1]-times[0]),
                median_gap_ms=statistics.median(gaps),max_gap_ms=max(gaps))

def audit(folder):
    folder=Path(folder);m=read_json(folder/'metadata.json');errors=[];notes=[]
    cfg=read_json(folder/'config.json')
    def rows(name):return list(jsonl(folder/name)) if (folder/name).is_file() else []
    frames=rows('frames.jsonl');steps=rows('steps.jsonl');events=rows('events.jsonl')
    frame_by={r['frame_index']:r for r in frames};command_by={r['command_index']:r for r in events if r['event']=='command'}
    if len(frame_by)!=len(frames):errors.append('duplicate frame IDs')
    if not frames or not steps:errors.append('empty RGB or steps')
    if m.get('status')!='complete' or m.get('success')!='user_confirmed':errors.append('full episode not confirmed successful')
    if m.get('simulated'):errors.append('simulation is not a real demonstration')
    if m.get('empty_run'):errors.append('empty gripper run is not a successful task demonstration')
    if {r.get('id') for r in m.get('objects',[]) if r.get('success')=='user_confirmed'}!=set(ROLES):
        errors.append('four object success confirmations required')
    if any(r['event'] in ('error','joint_error','command_error') for r in events):errors.append('capture/controller error in raw events')
    if any(k in m for k in ('stop_error','sampler_close_error','dispose_error','camera_forced_stop')):errors.append('unclean hardware/capture shutdown')
    if (folder/'camera_error.json').exists():errors.append('camera worker error')
    annotations=rows('annotations.jsonl')
    if [r['frame_index'] for r in annotations]!=[r['frame_index'] for r in frames]:
        errors.append('offline annotation rows missing or inconsistent')
    for ann in annotations:
        if set(ann.get('objects',{}))!=set(ROLES):
            errors.append('annotation must include all four IDs, with nulls for missing detections');break
        bad=False
        for obj in ann['objects'].values():
            if obj.get('status')=='detected_semantic_id_unverified':
                box=obj.get('bbox_xyxy');center=obj.get('center_pixel');corners=obj.get('bbox_corners')
                if (not isinstance(box,list) or len(box)!=4 or not isinstance(center,list) or len(center)!=2
                    or not isinstance(corners,list) or len(corners)!=4):bad=True;break
                if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in box+center):bad=True;break
            elif obj.get('status') not in ('not_detected','ambiguous_multiple_detections'):bad=True;break
        if bad:errors.append('invalid bbox/center annotation');break
    grouped=collections.defaultdict(list);seen=set();max_age=0.;max_latency=0.
    for i,s in enumerate(steps):
        tag=f'step {i}'
        if s['frame_index'] in seen:errors.append(tag+': duplicate observation frame')
        seen.add(s['frame_index']);f=frame_by.get(s['frame_index'])
        if not f or f['file']!=s['image'] or f['host_arrival_ns']!=s['frame_host_ns']:errors.append(tag+': frame mismatch');continue
        if not (folder/s['image']).is_file():errors.append(tag+': RGB missing')
        age=(s['frame_host_ns']-s['joint_read_ns'])/1e6
        latency=(s['decision_ns']-s['frame_host_ns'])/1e6
        max_age=max(max_age,age);max_latency=max(max_latency,latency)
        if not 0<=age<=cfg['max_joint_age_ms']:errors.append(tag+': noncausal or stale state')
        if not 0<=latency<=cfg['max_step_gap_ms']:errors.append(tag+': decision lag')
        if len(s['observation_state'])!=5 or not all(math.isfinite(x) for x in s['observation_state']):errors.append(tag+': state shape/finite')
        try:decode_action(s['action'])
        except ValueError as exc:errors.append(tag+': '+str(exc))
        c=command_by.get(s['command_index'])
        if c is None:errors.append(tag+': command not recorded')
        else:
            expected=encode_action(c['goal_deg'],c['gripper_command'],c['mode'])
            if len(s['action'])!=8 or max(abs(a-b) for a,b in zip(expected,s['action']))>1e-6:errors.append(tag+': command/action mismatch')
            if c['call_end_ns']>s['decision_ns']:errors.append(tag+': command timestamp mismatch')
        if s['object_id'] not in ROLES:errors.append(tag+': invalid object')
        if s['train_valid'] and s['observation_state_mask']!=[1]*5:errors.append(tag+': masked state exported as valid')
        grouped[s['segment_id']].append(s)
    segment_stats={}
    for key,group in grouped.items():
        stats=rate([s['frame_host_ns'] for s in group]);segment_stats[str(key)]=stats
        if stats.get('nonmonotonic') or stats.get('max_gap_ms',0)>cfg['max_step_gap_ms']:errors.append(f'segment {key}: timing gap')
        if stats['hz'] is None or not 8.5<=stats['hz']<=11.5:errors.append(f'segment {key}: 10Hz target missed')
        if len({s['object_id'] for s in group})!=1:errors.append(f'segment {key}: mixed objects')
        if not any(s['phase']=='RETURN_EXACT_HOME' for s in group):errors.append(f'segment {key}: home return missing')
        role=group[0]['object_id'];phases={s['phase'] for s in group}
        if not {'GRASP','RELEASE','RETURN_START'}<=phases:errors.append(f'segment {key}: grasp/release/return phase missing')
        if role in ('plastic_1','glass_1') and not {'TILT_DOWN_1','TILT_UP_1','TILT_DOWN_2','TILT_UP_2'}<=phases:
            errors.append(f'segment {key}: cup pour cycle missing')
        if role=='plate_1' and not {'TIP_UP_1','TIP_UP_2','POINT_DOWN_1','POINT_DOWN_2'}<=phases:
            errors.append(f'segment {key}: plate pour cycle missing')
    if len(grouped)!=4:errors.append('four segments required')
    joints=[r['read_end_ns'] for r in events if r['event']=='joint_observation']
    joint_stats=rate(joints)
    if not joint_stats['hz'] or not 17<=joint_stats['hz']<=23:errors.append('20Hz encoder read target missed')
    if joint_stats.get('max_gap_ms') and joint_stats['max_gap_ms']>150:errors.append('encoder read gap over 150ms')
    notes += ['Encoder reads may return cached samples. No sensor timestamp or measured gripper aperture.',
              'Native-goal actions require 8D action/5D state and the same mode-aware controller; not stock 5D ACT.',
              'Terminal waits are raw frames only; exported chunks never cross segment boundaries.',
              'IK/range checks are not collision or grasp-success measurements.']
    return dict(data_checks_passed=not errors,errors=errors,notes=notes,frames=len(frames),steps=len(steps),
                camera=rate([f['host_arrival_ns'] for f in frames]),joints=joint_stats,
                active_segments=segment_stats,max_joint_age_ms=max_age,max_decision_latency_ms=max_latency,
                scope='host-time collection integrity; physical task success is user confirmed')

def export_episode(folder):
    import numpy as np
    folder=Path(folder);report=audit(folder);atomic_json(folder/'audit.json',report)
    if not report['data_checks_passed']:raise ValueError('검증 실패. audit.json을 확인하세요: '+', '.join(report['errors'][:5]))
    from alignment import write_frame_alignment
    write_frame_alignment(folder)
    all_steps=list(jsonl(folder/'steps.jsonl'))
    valid_indices=[i for i,s in enumerate(all_steps) if s['train_valid']]
    rows=[all_steps[i] for i in valid_indices]
    annotations={a['frame_index']:a for a in jsonl(folder/'annotations.jsonl')}
    corners=np.full((len(rows),len(ROLES),4,2),np.nan,np.float32)
    centers=np.full((len(rows),len(ROLES),2),np.nan,np.float32)
    bbox_valid=np.zeros((len(rows),len(ROLES)),dtype=bool)
    for out_index,source_index in enumerate(valid_indices):
        for j,role in enumerate(ROLES):
            obj=annotations[rows[out_index]['frame_index']]['objects'][role]
            if obj['status']=='detected_semantic_id_unverified':
                corners[out_index,j]=obj['bbox_corners'];centers[out_index,j]=obj['center_pixel'];bbox_valid[out_index,j]=True
    out=folder/'training';out.mkdir(exist_ok=True)
    temp=out/'episode.tmp.npz'
    np.savez_compressed(temp,state=np.asarray([r['observation_state'] for r in rows],np.float32),
        action=np.asarray([r['action'] for r in rows],np.float32),image=np.asarray([r['image'] for r in rows]),
        segment_id=np.asarray([r['segment_id'] for r in rows],np.int32),
        object_id=np.asarray([r['object_id'] for r in rows]),
        host_ns=np.asarray([r['frame_host_ns'] for r in rows],np.int64),
        frame_index=np.asarray([r['frame_index'] for r in rows],np.int64),
        bbox_corners=corners,bbox_center=centers,bbox_valid=bbox_valid)
    temp.replace(out/'episode.npz')
    atomic_json(out/'schema.json',dict(**SCHEMA,episode_id=folder.name,hz=10,state_dim=5,action_dim=8,
                dataset_kind='native_goal_behavior_cloning',parent_episode=str(folder.resolve()),
                bbox_role_order=list(ROLES),bbox_note='YOLO detections are unverified; invalid boxes use NaN and bbox_valid=false',
                split_rule='split whole parent episodes/layouts, never frames or four segments independently'))
    return out/'episode.npz'

def chunk(npz,index,horizon):
    """A sampled action chunk cannot include user pauses or the next object's plan."""
    import numpy as np
    sid=npz['segment_id'][index];end=index
    while end<len(npz['action']) and end-index<horizon and npz['segment_id'][end]==sid:
        if end>index and 'frame_index' in npz and npz['frame_index'][end]!=npz['frame_index'][end-1]+1:break
        end+=1
    result=np.zeros((horizon,8),np.float32);mask=np.ones(horizon,dtype=bool)
    result[:end-index]=npz['action'][index:end];mask[:end-index]=False
    return result,mask

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--export',action='store_true');a=p.parse_args()
    report=audit(a.episode);atomic_json(a.episode/'audit.json',report)
    print('검증 통과:',report['data_checks_passed']);print('RGB:',report['camera']);print('관절:',report['joints'])
    for error in report['errors']:print('실패:',error)
    if a.export:print('학습 표본:',export_episode(a.episode))

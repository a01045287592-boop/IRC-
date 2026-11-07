"""Reuse supplied full-motion planners; normalize their native endpoint goals."""
import copy
import math
import sys
from pathlib import Path
from core import ROLES, validate_config

sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import cup_pick_real as kin
import fixed_cup_demo as cup
import plate_one_demo as plate
import object_pick_demo as tissue

def last_goal(stages):
    for s in reversed(stages):
        if 'target' in s:return list(s['target'])
        if s.get('targets'):return list(s['targets'][-1])
    raise ValueError('목표 없는 계획')

def normalize(stages,cfg):
    result=[]
    for source in stages:
        s=copy.deepcopy(source)
        if s['kind']=='marker':continue
        if s['kind'] in ('move','rotate'):
            s['target']=s.get('target',s.get('targets',[None])[-1]);s.pop('targets',None)
            s['pitch']=s.get('pitch',s.get('pitch_deg',kin.pitch(s['target'])))
            s['mode']='wrist' if s['kind']=='rotate' else ('horizontal' if abs(s['pitch'])<.01 else 'vertical')
            kin.check_q(s['target'])
            if s['kind']=='move' and min(abs(s['pitch']),abs(s['pitch']+90))>.01:
                raise ValueError('이송 자세는 0 / -90도여야 합니다.')
            fine=s['name'] in ('LOWER_TO_PLATE','LOWER_TO_OBJECT','FRONT_APPROACH','LOWER_BEFORE_CUP','LOWER_TO_PLACE')
            s['position_tolerance_cm']=.3 if fine else .5
            if s['name']=='LOWER_TO_STACK':s['position_tolerance_cm']=.4
            if s['name']=='LIFT_OPEN_GRIPPER':s['position_tolerance_cm']=.65
            s['pitch_tolerance_deg']=3. if s['kind']=='rotate' else 2.5
        else:s['seconds']=cfg['hold_seconds'] if s['kind']=='hold' else cfg['gripper_settle_seconds']
        result.append(s)
    return result

def object_plan(cfg,role,start,home):
    dest=cfg['destinations'];g=cfg['pickups'][role];h=cfg['clearance_z_cm']
    if role=='plate_1':
        # PLATE_CLEARANCE_17CM_V1
        # 접시 작업에만 최소 손목 높이 17cm를 적용한다.
        h = max(h, 17.0)
        stages=plate.plan(dict(plate_grasp_wrist_cm=[g,g],plate_stack_wrist_cm=dest['plate_place'],
                    stack_rise_mm=5.,pour_wrist_cm=dest['plate_pour'],clearance_z_cm=h,
                    pour_pitch_deg=cfg['plate_pour_pitch_deg'],tilt_cycles=2,hold_seconds=cfg['hold_seconds']),start)
    elif role=='tissue_1':
        stages=tissue.make_plan(start,dict(grasp_wrist_cm=g,drop_wrist_cm=dest['tissue_bin'],clearance_z_cm=h),'tissue')
    elif role in ('plastic_1','glass_1'):
        place=dest['plastic_place' if role=='plastic_1' else 'glass_place']
        # GLASS_DIRECT_PLACE_V1
        # 높은 유리컵 배치점에서도 기존 컵 계획의 높이 조건을 만족.
        if role == 'glass_1':
            h = max(h, place[2] + 2.0)

        c=dict(grasp_wrist_cm=g,pour_wrist_cm=dest['cup_pour'],place_wrist_cm=place,
               clearance_z_cm=h,approach_cm=cfg['approach_cm'],pour_pitch_deg=cfg['cup_pour_pitch_deg'],
               shake_cycles=2,hold_seconds=cfg['hold_seconds'],speed=cfg['speed'])
        stages=cup.plan(c,start)
        if role=='glass_1':
            # Match the taught wrist angle at the stationary grasp point.
            # Native travel supports only horizontal/vertical pitch, so return
            # to horizontal after closing and before the existing upright lift.
            grasp_pitch=cfg.get('glass_grasp_pitch_deg',0.)
            if type(grasp_pitch) not in (int,float) or not math.isfinite(grasp_pitch) or abs(grasp_pitch)>10:
                raise ValueError('유리컵 파지 pitch는 -10~10도의 유한한 숫자여야 합니다.')
            if abs(grasp_pitch)>1e-9:
                matches=[i for i,s in enumerate(stages) if s['name']=='GRASP' and s['kind']=='close']
                if len(matches)!=1:
                    raise ValueError('유리컵 파지 단계를 정확히 찾지 못했습니다.')
                i=matches[0]
                if i==0 or i+1>=len(stages) or stages[i-1]['name']!='FRONT_APPROACH' or stages[i+1]['name']!='LIFT_UPRIGHT':
                    raise ValueError('유리컵 파지 전후 단계가 예상과 다릅니다.')
                horizontal=last_goal(stages[:i])
                if abs(kin.pitch(horizontal))>.01:
                    raise ValueError('유리컵 파지 전 자세가 수평이 아닙니다.')
                tilted=list(horizontal)
                tilted[3]+=grasp_pitch-kin.pitch(tilted)
                kin.check_q(tilted)
                stages[i:i]=[dict(name='GLASS_MATCH_GRASP_PITCH',kind='rotate',pitch=grasp_pitch,target=tilted)]
                stages[i+2:i+2]=[dict(name='GLASS_STRAIGHTEN_AFTER_GRASP',kind='rotate',pitch=0.,target=horizontal)]

            # 붓기 후 중앙과 배치점 위 경유를 빼고 배치점으로 이동.
            stages = [
                s for s in stages
                if s['name'] not in ('RIGHT_TRANSIT', 'ABOVE_PLACE')
            ]
            stages=stages[:next(i for i,s in enumerate(stages) if s['name']=='RELEASE')]
            q=last_goal(stages);q[3]=-180-q[1]-q[2];kin.check_q(q)
            stages += [dict(name='GLASS_POINT_DOWN_AT_PLACE',kind='rotate',pitch=-90,target=q),
                       dict(name='RELEASE',kind='open'),
                       dict(name='GLASS_LIFT_OPEN',kind='move',pitch=-90,target=plate.ik([place[0],place[1],h],-90)),
                       dict(name='GLASS_RETURN_HORIZONTAL',kind='rotate',pitch=0,target=plate.ik([place[0],place[1],h],0))]
        stages += [dict(name='RETURN_CENTER',kind='move',pitch=0,target=plate.ik([0.,10.,h],0)),
                   dict(name='RETURN_START',kind='move',pitch=0,target=plate.ik(kin.wrist_fk(home),0))]
    else:raise ValueError(role)
    # Always return to the episode's exact initial joint posture, including its
    # measured small pitch offset. Only q4 changes at this final settle phase.
    normalized=normalize(stages,cfg)

    # TISSUE_RETURN_TOLERANCE_6_5MM
    # 휴지를 놓은 뒤 위로 이탈하는 단계에만 6.5mm 적용.
    if role == 'tissue_1':
        matches = [s for s in normalized if s['name'] == 'RETURN_ABOVE_BIN']
        if len(matches) != 1:
            raise ValueError('휴지 복귀 단계를 정확히 찾지 못했습니다.')
        matches[0]['position_tolerance_cm'] = .65

    final=last_goal(normalized)
    if max(abs(a-b) for a,b in zip(final[:3],home[:3]))>1e-4:raise ValueError('복귀 좌표 오류')
    normalized.append(dict(name='RETURN_EXACT_HOME',kind='rotate',mode='wrist',pitch=kin.pitch(home),
                           target=list(home),position_tolerance_cm=.5,pitch_tolerance_deg=3.))
    if sum(s['kind']=='close' for s in normalized)!=1 or sum(s['name']=='RELEASE' for s in normalized)!=1:
        raise ValueError('파지/놓기 횟수 오류')
    return normalized

def validate_transitions(stages,start):
    """Joint range and mode checks only; NOT collision/continuous Cartesian proof."""
    q=list(start)
    for s in stages:
        if s['kind'] not in ('move','rotate'):continue
        goal=kin.check_q(s['target'])
        if s['kind']=='rotate' and max(abs(a-b) for a,b in zip(q[:3],goal[:3]))>1e-5:
            raise ValueError(s['name']+': q4 회전 중 q1..3 변경')
        if s['kind']=='move' and abs(kin.pitch(q)-s['pitch'])>4:
            raise ValueError(s['name']+': 자세 전환 누락')
        # Native mode computes q4 from q2/q3. Check the commanded joint-space
        # endpoints/interpolation; physical firmware timing is hardware-tested separately.
        for i in range(21):
            p=[a+(b-a)*i/20 for a,b in zip(q,goal)]
            if s['kind']=='move':p[3]=s['pitch']-90-p[1]-p[2]
            kin.check_q(p)
        q=goal
    return q

def plan_episode(cfg,order,home):
    validate_config(cfg);kin.check_q(home)
    if len(order)!=4 or set(order)!=set(ROLES):raise ValueError('서로 다른 네 물체 순서 필요')
    if abs(kin.pitch(home))>3:raise ValueError('시작 자세 pitch가 0±3도가 아닙니다.')
    plans=[];q=list(home)
    for role in order:
        stages=object_plan(cfg,role,q,home);q=validate_transitions(stages,q)
        plans.append(dict(id=role,stages=stages))
    return plans

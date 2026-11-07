"""Four-object demonstration schema; no SDK/camera imports."""
import copy
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path

VERSION = 'TRAY4_DEMO_1.0'
ROLES = ('plastic_1', 'glass_1', 'plate_1', 'tissue_1')
LABELS = dict(zip(ROLES, ('플라스틱 컵 1', '유리컵 1', '접시 1', '휴지 1')))
KINDS = dict(zip(ROLES, ('plastic', 'glass', 'plate', 'tissue')))
MODES = ('horizontal', 'vertical', 'wrist')
SCHEMA = {
    'observation_state': ['q1_rad', 'q2_rad', 'q3_rad', 'q4_rad', 'previous_gripper_command_closed'],
    'action': ['goal_q1_rad', 'goal_q2_rad', 'goal_q3_rad', 'goal_q4_rad',
               'gripper_command_closed', 'mode_horizontal', 'mode_vertical', 'mode_wrist'],
    'joint_command_mask': {'horizontal': [1,1,1,0], 'vertical': [1,1,1,0], 'wrist': [0,0,0,1]},
    'action_semantics': 'Persistent native SDK goal, not measured future joints. Mode is required.',
    'gripper_semantics': '0=open command, 1=close command; actual aperture/success unmeasured',
    'time': 'host monotonic_ns; camera host arrival is not exposure synchronization',
}

def default_config():
    return {
        'schema': 1, 'version': VERSION,
        'destinations': {
            'tissue_bin': [12.610,11.823,9.725], 'plate_pour': [16.074,6.658,14.824],
            'cup_pour': [13.428,6.225,13.165], 'plate_place': [12.625,-1.165,9.676],
            'glass_place': [16.492,-1.089,10.792], 'plastic_place': [11.313,-.486,5.716]},
        'pickups': {r: None for r in ROLES},
        'speed': 20, 'clearance_z_cm': 15., 'approach_cm': 2.,
        'cup_pour_pitch_deg': -90., 'plate_pour_pitch_deg': 10.,
        'pour_cycles': 2, 'hold_seconds': 1., 'gripper_settle_seconds': 1.,
        'model': '~/yolo_ws/runs/segment/angle3/weights/best.pt',
        'vision_python': '~/yolo_ws/.venv/bin/python',
        'confidence': .4, 'glass_max': .20, 'plastic_min': .25,
        'stable_frames': 5, 'stable_pixels': 8.,
        'camera_hz': 10, 'joint_read_hz': 20, 'camera_serial': None,
        'max_joint_age_ms': 100., 'max_step_gap_ms': 200.,
        'instruction': '트레이의 아래쪽 변 중앙에서 가까운 물체부터 네 물체를 정리한다.',
    }

def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def hash_data(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()

def finite_point(point):
    return (isinstance(point, list) and len(point)==3 and
            all(type(v) in (int,float) and math.isfinite(v) for v in point))

def validate_config(cfg, require_pickups=True):
    if cfg.get('schema') != 1: raise ValueError('설정 schema 오류')
    if set(cfg.get('pickups', {})) != set(ROLES): raise ValueError('파지 위치는 네 ID만 사용합니다.')
    for role in ROLES:
        if cfg['pickups'][role] is None and not require_pickups: continue
        if not finite_point(cfg['pickups'][role]): raise ValueError(f'{LABELS[role]} 손목 XYZ(cm)가 필요합니다.')
    for k in default_config()['destinations']:
        if not finite_point(cfg.get('destinations', {}).get(k)): raise ValueError(f'목적지 {k} 오류')
    if type(cfg['speed']) is not int or not 1<=cfg['speed']<=40: raise ValueError('speed 정수 1..40')
    for k in ('clearance_z_cm','approach_cm','cup_pour_pitch_deg','plate_pour_pitch_deg',
              'hold_seconds','gripper_settle_seconds','confidence','glass_max','plastic_min',
              'stable_pixels','max_joint_age_ms','max_step_gap_ms'):
        if type(cfg[k]) not in (float,int) or not math.isfinite(cfg[k]): raise ValueError(k+' 값 오류')
    if not 0<=cfg['glass_max']<cfg['plastic_min']<=1: raise ValueError('재질 임계값 오류')
    if not 0<cfg['confidence']<1 or not 0<cfg['stable_pixels']<=20: raise ValueError('검출 설정 오류')
    if type(cfg['stable_frames']) is not int or cfg['stable_frames']<3: raise ValueError('stable_frames >=3')
    if cfg['camera_hz']!=10 or cfg['joint_read_hz']!=20: raise ValueError('이 버전은 RGB 10Hz / 관절 읽기 20Hz입니다.')
    if not .5<=cfg['hold_seconds']<=3 or not .5<=cfg['gripper_settle_seconds']<=3:
        raise ValueError('유지 시간은 .5..3초')
    if cfg['pour_cycles']!=2: raise ValueError('붓기는 두 번입니다.')
    if not 0<cfg['max_joint_age_ms']<=100 or not 100<=cfg['max_step_gap_ms']<=200:
        raise ValueError('동기화 허용 범위를 늘려 결측을 숨길 수 없습니다.')

def selection(objects, ref):
    """Exactly one per semantic role; distance ties resolved by ID, never random."""
    expected = set(KINDS.values())
    if len(objects)!=4 or {o['kind'] for o in objects}!=expected:
        counts={k:sum(o['kind']==k for o in objects) for k in expected}
        raise ValueError('접시1 휴지1 플라스틱1 유리1 필요: '+str(counts))
    rows=[]
    for obj in objects:
        role=next(r for r,k in KINDS.items() if k==obj['kind'])
        box=list(map(float,obj['box'])); cx=list(map(float,obj['center']))
        x1,y1,x2,y2=box
        rows.append(dict(id=role, kind=obj['kind'], center_pixel=cx, bbox_xyxy=box,
                         bbox_corners=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                         distance_pixel=math.dist(cx,ref), white_ratio=obj.get('white_ratio')))
    rows.sort(key=lambda o:(o['distance_pixel'],o['id']))
    return dict(order=[o['id'] for o in rows],objects=rows,reference_pixel=list(ref),
                distance_unit='pixel', rule='front tray bottom edge midpoint to mask center')

def compatible_selections(a,b,tolerance=8.):
    if not a or not b or a['order']!=b['order']: return False
    if math.dist(a['reference_pixel'],b['reference_pixel'])>tolerance: return False
    return all(x['id']==y['id'] and math.dist(x['center_pixel'],y['center_pixel'])<=tolerance
               for x,y in zip(a['objects'],b['objects']))

def encode_action(goal_deg, gripper, mode):
    if len(goal_deg)!=4 or mode not in MODES or gripper not in (0,1): raise ValueError('action 오류')
    if not all(math.isfinite(v) for v in goal_deg): raise ValueError('action finite 오류')
    return [math.radians(v) for v in goal_deg]+[float(gripper)]+[float(m==mode) for m in MODES]

def decode_action(values):
    if len(values)!=8 or not all(math.isfinite(float(v)) for v in values): raise ValueError('8차원 action 필요')
    if values[4] not in (0,1) or list(values[5:]).count(1)!=1 or any(v not in (0,1) for v in values[5:]):
        raise ValueError('기록 action에는 이산 grip/mode가 필요합니다.')
    return [math.degrees(v) for v in values[:4]],int(values[4]),MODES[list(values[5:]).index(1)]

class JsonLog:
    def __init__(self,path):
        self.path=Path(path);self.file=self.path.open('x',encoding='utf-8');self.lock=threading.RLock()
    def write(self, event, **kwargs):
        with self.lock:
            row=dict(event=event,host_ns=time.monotonic_ns(),**kwargs)
            self.file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.file.flush()
        return row
    def close(self):
        with self.lock:self.file.close()

def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            try: yield json.loads(line)
            except ValueError as e: raise ValueError(f'{path}:{i} 손상된 JSON') from e

"""Shared episode state. No camera, model, SDK imports or motor writes."""
import copy
import hashlib
import json
import math
import threading
import time
from pathlib import Path

VERSION = 'TRAY_EPISODE_V1'
ROLES = {'plate': 'plate', 'tissue': 'tissue', 'plastic_1': 'plastic',
         'plastic_2': 'plastic', 'glass': 'glass'}
COUNTS = {'plate': 1, 'tissue': 1, 'plastic': 2, 'glass': 1}


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def point(value):
    return isinstance(value, list) and len(value) == 3 and all(number(v) for v in value)


def config_errors(cfg):
    errors = []
    if cfg.get('schema') != 1:
        errors.append('schema must be 1')
    for role in ROLES:
        if not point(cfg.get('pickups', {}).get(role)):
            errors.append('pickups.' + role + ' : 손목 [x,y,z] cm 미입력')
    for name in ('plate_stack', 'plate_pour', 'tissue_bin', 'cup_pour', 'plastic_stack', 'glass_place'):
        if not point(cfg.get('destinations', {}).get(name)):
            errors.append('destinations.' + name + ' : 손목 [x,y,z] cm 미입력')
    v = cfg.get('plastic_stack_step_mm')
    if not number(v) or not 0 < v <= 30:
        errors.append('plastic_stack_step_mm : 두 컵 적층 높이 증가량(0 초과 30mm 이하) 미입력')
    if not number(cfg.get('clearance_z_cm')) or not 0 < cfg['clearance_z_cm'] <= 20:
        errors.append('clearance_z_cm must be 0..20 cm')
    if type(cfg.get('speed')) is not int or not 1 <= cfg['speed'] <= 40:
        errors.append('speed must be an integer 1..40')
    names = cfg.get('model', {}).get('class_names', {})
    if set(names) != {'plate', 'tissue', 'plastic', 'glass', 'tray'}:
        errors.append('model.class_names : plate/tissue/plastic/glass/tray 필요')
    elif any(not isinstance(v, str) or not v for v in names.values()) or len(set(names.values())) != 5:
        errors.append('유리컵과 플라스틱컵을 포함해 5개 클래스 이름은 서로 달라야 합니다.')
    if not cfg.get('model', {}).get('weights'):
        errors.append('model.weights : 컵 재질을 분리 학습한 segmentation 모델 경로 미입력')
    return errors


def validate_model_names(model_names, class_names):
    names = set(model_names.values() if isinstance(model_names, dict) else model_names)
    missing = set(class_names.values()) - names
    if missing:
        raise ValueError('모델 클래스 부족: ' + ', '.join(sorted(missing)) +
                         '. 기존 cup 클래스만으로 재질을 구분하지 않습니다.')


def make_selection(objects, reference, frame_index, frame_ns, config_hash):
    """objects have canonical kind/center/box and are already on the chosen tray."""
    counts = {k: sum(o['kind'] == k for o in objects) for k in COUNTS}
    if counts != COUNTS or len(objects) != 5:
        raise ValueError('정확히 접시1/휴지1/플라스틱컵2/유리컵1 필요: ' + str(counts))
    plastic = sorted((o for o in objects if o['kind'] == 'plastic'),
                     key=lambda o: (o['center'][0], o['center'][1]))
    selected = []
    for o in objects:
        role = ('plastic_' + str(next(i for i, p in enumerate(plastic, 1) if p is o))
                if o['kind'] == 'plastic' else o['kind'])
        selected.append({'id': role, 'kind': o['kind'], 'center': list(o['center']),
                         'box': list(o['box']), 'distance_px': math.dist(reference, o['center'])})
    selected.sort(key=lambda o: (o['distance_px'], o['id']))
    for i, o in enumerate(selected, 1):
        o['rank'] = i
    return {'schema': 1, 'reference_px': list(reference), 'objects': selected,
            'frame_index': frame_index, 'frame_ns': frame_ns, 'config_hash': config_hash}


def validate_selection(selection, cfg):
    objects = selection['objects']
    if len(objects) != 5 or {o['id'] for o in objects} != set(ROLES):
        raise ValueError('선택 대상은 서로 다른 5개여야 합니다.')
    if selection['config_hash'] != digest(cfg):
        raise ValueError('화면 확인 이후 설정이 달라졌습니다.')
    prev = -1
    for i, o in enumerate(objects, 1):
        if o['rank'] != i or o['kind'] != ROLES[o['id']]:
            raise ValueError('대상 순번/종류가 잘못되었습니다.')
        d = math.dist(selection['reference_px'], o['center'])
        if not math.isfinite(d) or abs(d - o['distance_px']) > 1e-6 or d < prev:
            raise ValueError('거리 정렬 검증 실패')
        prev = d


def wants_overlay(frame):
    return frame['state']['phase'] == 'before_grasp'


class EpisodeState:
    """Camera stamps state atomically at frame receipt. No delayed signal polling."""
    def __init__(self):
        self.lock = threading.RLock()
        self.data = dict(phase='waiting', recording=False, object_id=None, rank=None,
                         gripper_command=None, revision=0, changed_ns=time.monotonic_ns())
        self.order = []
        self.finished = []

    def _update(self, **kw):
        self.data.update(kw)
        self.data['revision'] += 1
        self.data['changed_ns'] = time.monotonic_ns()
        return copy.deepcopy(self.data)

    def snapshot(self):
        with self.lock:
            return time.monotonic_ns(), copy.deepcopy(self.data)

    def start(self, order):
        with self.lock:
            if self.data['phase'] != 'waiting' or len(order) != 5 or set(order) != set(ROLES):
                raise ValueError('에피소드 시작 상태/대상이 잘못되었습니다.')
            self.order = list(order)
            return self._update(phase='between', recording=True)

    def begin_object(self, role):
        with self.lock:
            if self.data['phase'] != 'between' or role != self.order[len(self.finished)]:
                raise ValueError('다음 대상 순서가 잘못되었습니다.')
            return self._update(phase='before_grasp', object_id=role, rank=len(self.finished)+1)

    def close_command(self):
        with self.lock:
            if self.data['phase'] != 'before_grasp':
                raise ValueError('파지 전 상태에서만 그리퍼를 닫을 수 있습니다.')
            return self._update(phase='after_grasp', gripper_command='close')

    def open_command(self):
        with self.lock:
            return self._update(gripper_command='open')  # does NOT re-enable YOLO

    def complete_object(self):
        with self.lock:
            if self.data['phase'] != 'after_grasp' or self.data['gripper_command'] != 'open':
                raise ValueError('파지/배출 단계가 끝나지 않았습니다.')
            self.finished.append(self.data['object_id'])
            return self._update(phase='between', object_id=None, rank=None)

    def finish(self, failed=False):
        with self.lock:
            if not failed and len(self.finished) != 5:
                raise ValueError('5개 명령이 모두 끝나지 않았습니다.')
            return self._update(phase='failed' if failed else 'finished', recording=False)

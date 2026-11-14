"""Adapters around the supplied motion code, without calling its separate mains."""
import copy
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '_motion'))
import cup_pick_real as bench_module
import plate_one_demo as plate
import object_pick_demo as tissue
import plate_two_auto as tissue_executor
import fixed_cup_demo as cup
import smooth_cup_motion as cup_executor


def settings(cfg, role, plastic_placed):
    h = cfg['clearance_z_cm']; dest = cfg['destinations']; grasp = cfg['pickups'][role]
    if role == 'plate':
        c = copy.deepcopy(cfg['plate_motion'])
        c.update(plate_grasp_wrist_cm=[grasp, grasp], plate_stack_wrist_cm=dest['plate_stack'],
                 pour_wrist_cm=dest['plate_pour'], clearance_z_cm=h)
    elif role == 'tissue':
        c = dict(grasp_wrist_cm=grasp, drop_wrist_cm=dest['tissue_bin'], clearance_z_cm=h,
                 hold_seconds=.5)
    else:
        c = copy.deepcopy(cfg['cup_motion'])
        target = list(dest['glass_place'] if role == 'glass' else dest['plastic_stack'])
        if role.startswith('plastic_'):
            target[2] += plastic_placed * cfg['plastic_stack_step_mm'] / 10
        c.update(grasp_wrist_cm=grasp, pour_wrist_cm=dest['cup_pour'],
                 place_wrist_cm=target, clearance_z_cm=h, speed=cfg['speed'])
    return c


def final_target(stages):
    for s in reversed(stages):
        if 'target' in s:
            return list(s['target'])
        if s.get('targets'):
            return list(s['targets'][-1])
    raise ValueError('동작 경로가 비었습니다.')


def home_stages(home, height):
    return [dict(name='RETURN_CENTER', kind='move', pitch=0, target=plate.ik([0.,10.,height],0)),
            dict(name='RETURN_START', kind='move', pitch=0,
                 target=plate.ik(bench_module.wrist_fk(home),0))]



# GLASS_DOWN_RELEASE_V1
def glass_down_plan(cfg, start):
    """Keep the existing horizontal pick/pour; rotate down at destination."""
    original = cup.plan(cfg, start)
    release = next(i for i, s in enumerate(original) if s['name'] == 'RELEASE')
    stages = original[:release]
    current = final_target(stages)
    destination = cfg['place_wrist_cm']

    def rotate(name, desired):
        nonlocal current
        goal = list(current)
        goal[3] += desired - bench_module.pitch(current)
        # Check the full wrist rotation, without loosening joint limits.
        n = max(1, math.ceil(abs(goal[3] - current[3]) / 5))
        for i in range(1, n + 1):
            bench_module.check_q([a + (b-a)*i/n for a,b in zip(current,goal)])
        stages.append(dict(name=name, kind='rotate', pitch=desired,
                           target=goal, engine='plate'))
        current = goal

    rotate('GLASS_POINT_DOWN_AT_PLACE', -90)
    stages.append(dict(name='RELEASE', kind='open', engine='plate'))
    raised = [destination[0], destination[1], cfg['clearance_z_cm']]
    initial = bench_module.wrist_fk(current)
    n = max(1, math.ceil(math.dist(initial, raised) / .1))
    previous = current
    for i in range(1, n + 1):
        q = plate.ik([a + (b-a)*i/n for a,b in zip(initial,raised)], -90)
        if max(abs(a-b) for a,b in zip(previous,q)) > 5:
            raise ValueError('GLASS_LIFT_OPEN: IK jump or singularity')
        previous = q
    current = previous
    stages.append(dict(name='GLASS_LIFT_OPEN', kind='move', pitch=-90,
                       target=current, xyz=raised, engine='plate'))
    rotate('GLASS_RETURN_HORIZONTAL', 0)
    return stages


def plan_object(cfg, role, start, home, plastic_placed):
    c = settings(cfg, role, plastic_placed)
    if role == 'plate':
        stages = plate.plan(c, start)
        engine = 'plate'
    elif role == 'tissue':
        stages = tissue.make_plan(start, c, 'tissue')
        engine = 'tissue'
    else:
        stages = glass_down_plan(c, start) if role == 'glass' else cup.plan(c, start)
        engine = 'cup'
    for s in stages:
        s.setdefault('engine', engine)
    if engine == 'cup':
        # Existing cup routine ends raised; add a horizontal return via the center.
        for s in home_stages(home, cfg['clearance_z_cm']):
            s['engine'] = 'plate'
            stages.append(s)
    if sum(s['kind'] == 'close' for s in stages) != 1:
        raise ValueError('대상당 파지 명령은 정확히 한 번이어야 합니다.')
    return c, stages


def preflight(cfg, order, start):
    current = list(start); home = list(start); plastic_placed = 0; result = []
    for role in order:
        c, stages = plan_object(cfg, role, current, home, plastic_placed)
        current = final_target(stages)
        result.append(dict(object_id=role, settings=c, stages=stages))
        if role.startswith('plastic_'):
            plastic_placed += 1
    return result


def execute_stage(robot, bench, stage, record):
    engine = stage['engine']
    if getattr(bench, '_tray_engine', None) != engine:
        # The original executors use separate mode caches. Do not reuse another
        # executor's cached lock state when switching cup/plate/tissue routines.
        bench._native_horizontal_active = False
        bench._plate_mode = None
        bench._tray_engine = engine
    if engine == 'cup':
        cup_executor.execute_phase(robot, bench, stage['targets'], stage, record)
    elif engine == 'tissue':
        tissue_executor.execute_motion(robot, bench, stage, record)
    else:
        plate.execute_motion(robot, bench, stage, record)

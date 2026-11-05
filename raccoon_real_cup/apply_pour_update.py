"""Run from any directory; place beside fixed_cup_demo.py before running."""
import ast
import json
import math
import shutil
from datetime import datetime
from pathlib import Path


def replace_once(text, old, new):
    if new in text:
        return text
    if text.count(old)!=1:
        raise RuntimeError('예상한 코드와 다릅니다. 파일을 수정하지 않았습니다: '+old[:70])
    return text.replace(old,new,1)


def apply(root):
    control=root/'cup_pick_real.py'
    demo=root/'fixed_cup_demo.py'
    config=root/'fixed_cup_config.json'
    a=control.read_text();b=demo.read_text();cfg=json.loads(config.read_text())
    a=replace_once(a,'def move_q(self, target, cartesian=False):',
                   'def move_q(self, target, cartesian=False, position_tolerance_cm=.3):')
    a=replace_once(a,'position_ok = math.dist(wrist_fk(q),wrist_fk(target)) <= .3',
                   'position_ok = math.dist(wrist_fk(q),wrist_fk(target)) <= position_tolerance_cm')
    b=replace_once(b,"b.move_q(q,cartesian=True)",
        """# Clear transit/pouring area: 5mm; grasp/place: 3mm.
                    transit = s['kind'] == 'rotate' or s['name'] in ('INITIAL_RAISE', 'ABOVE_APPROACH',
                        'LIFT_UPRIGHT', 'ABOVE_POUR', 'POUR_POSITION',
                        'LEAVE_POUR', 'ABOVE_PLACE', 'FINISH_RAISED')
                    b.move_q(q, cartesian=True, position_tolerance_cm=.5 if transit else .3)""")
    # Remember the previous pour point so rerunning the installer does not rotate again.
    marker='_pour_left20_patch'
    if marker not in cfg:
        old=cfg['pour_wrist_cm']
        if len(old)!=3 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in old):
            raise ValueError('Invalid pour_wrist_cm')
        angle=math.radians(20)
        x,y,z=old
        cfg['pour_wrist_cm']=[x*math.cos(angle)-y*math.sin(angle),
                              x*math.sin(angle)+y*math.cos(angle),z]
        cfg[marker]={'original_wrist_cm':old,'extra_joint1_deg':20}
    cfg['speed']=20
    ast.parse(a);ast.parse(b)
    # Check the updated full path offline against the existing compatible planner.
    import importlib.util
    import sys
    sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location('fixed_patch_check',demo)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.plan(cfg,[0,-10,-140,60])
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    updates=((control,a),(demo,b),(config,json.dumps(cfg,indent=2)+'\n'))
    for p,content in updates:
        if p.read_text()!=content:
            backup=p.with_name(p.name+'.before_pour_update_'+stamp+'.bak')
            shutil.copy2(p,backup)
            p.write_text(content)
    x,y,z=cfg['pour_wrist_cm']
    print('적용 완료. speed:',cfg['speed'])
    print('파지 좌표 유지:',cfg['grasp_wrist_cm'])
    print('배출 손목 좌표(cm):',[round(v,4) for v in (x,y,z)])
    print('배출 최종 joint1(deg):',round(math.degrees(math.atan2(-x,y)),3))
    print('이송·배출 허용 오차 5mm / 접근·배치 3mm. 기구학 검사 통과; 실물 시험 필요.')


if __name__=='__main__':
    apply(Path(__file__).resolve().parent)

"""Install after apply_pour_update.py; keep config and prior files backed up."""
import ast
import shutil
from datetime import datetime
from pathlib import Path

root=Path(__file__).resolve().parent
p=root/'fixed_cup_demo.py'
module=root/'smooth_cup_motion.py'
if not module.is_file():raise SystemExit('smooth_cup_motion.py가 같은 폴더에 없습니다.')
s=p.read_text()
if 'from smooth_cup_motion import execute_phase' in s:
    raise SystemExit('이미 적용됐습니다. 추가 변경 없음.')
tree=ast.parse(s)
if not any(isinstance(n,ast.FunctionDef) and n.name=='plan' for n in tree.body):
    raise SystemExit('fixed_cup_demo.py가 원본 동작 파일이 아닙니다. 변경하지 않았습니다.')
old='''                for q in s['targets']:
                    observed=b.read()
                    if max(abs(a-v) for a,v in zip(observed,q))>8:
                        raise RuntimeError('계획 경로에서 크게 벗어났습니다. 자동 복귀하지 않습니다.')
                    record('command',observation_q=observed,target_q=q)
                    # Clear transit/pouring area: 5mm; grasp/place: 3mm.
                    transit = s['kind'] == 'rotate' or s['name'] in ('INITIAL_RAISE', 'ABOVE_APPROACH',
                        'LIFT_UPRIGHT', 'ABOVE_POUR', 'POUR_POSITION',
                        'LEAVE_POUR', 'ABOVE_PLACE', 'FINISH_RAISED')
                    b.move_q(q, cartesian=True, position_tolerance_cm=.5 if transit else .3)
'''
new="                execute_phase(robot, b, s['targets'], s, record)\n"
if s.count(old)!=1:
    raise SystemExit('이전 업데이트가 없거나 파일 구조가 다릅니다. 변경하지 않았습니다.')
s=s.replace(old,new,1)
marker='from cup_pick_real import (Bench, SDK_VERSION, check_q, wrist_fk, horizontal_ik,\n                           pitch, execution_plan)'
if s.count(marker)!=1:raise SystemExit('import 위치를 찾을 수 없습니다. 변경하지 않았습니다.')
s=s.replace(marker,marker+'\nfrom smooth_cup_motion import execute_phase',1)
ast.parse(s);ast.parse(module.read_text())
backup=p.with_name(p.name+'.before_smooth_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.bak')
shutil.copy2(p,backup)
p.write_text(s)
print('적용 완료:',p)
print('기존 코드 백업:',backup)
print('설정 파일과 좌표는 수정하지 않았습니다.')

"""Train four explicitly routed ACT policies from existing partial episodes."""
import argparse
import json
from pathlib import Path
from core import ROLES, KINDS, atomic_json
from act_train import train
from review import audit

ROOT=Path(__file__).absolute().parents[2]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--phase-conditioned',action='store_true')
    p.add_argument('--role',choices=ROLES,action='append')
    p.add_argument('--check',action='store_true')
    p.add_argument('--out',type=Path,default=ROOT/'raccoon_dataset/tray4_object_act')
    a=p.parse_args()
    datasets={};excluded=[]
    for role in (a.role or ROLES):
        folders=[]
        for npz in sorted((ROOT/'raccoon_dataset/tray4_collected'/KINDS[role]).glob('*/ */training/episode.npz'.replace(' ',''))):
            folder=npz.parent.parent
            schema=json.loads((folder/'training/schema.json').read_text())
            report=audit(folder,partial=True)
            if schema.get('object_id')==role and report['data_checks_passed']:folders.append(folder)
            else:excluded.append(str(folder))
        if len(folders)<2:raise ValueError(f'{role}: 학습 에피소드 부족')
        datasets[role]=folders
    print(json.dumps({'episodes':{r:len(v) for r,v in datasets.items()},'excluded':excluded},indent=2),flush=True)
    if not a.check:
        if a.out.exists():raise FileExistsError(f'기존 결과 보호: 다른 --out 경로를 지정하세요: {a.out}')
        a.out.mkdir(parents=True)
        manifest={r:str((a.out/r).absolute()) for r in datasets}
        for role,folders in datasets.items():
            print('TRAIN',role,flush=True)
            from act_phase import build_spec
            import random
            order=list(range(len(folders)));random.Random(0).shuffle(order)
            train_folders=[folders[i] for i in order[max(1,round(len(order)*.2)):]]
            spec=build_spec(train_folders) if a.phase_conditioned else None
            train(folders,a.out/role,phase_spec=spec,steps=a.steps,batch_size=16,horizon=10,seed=0,lighting_aug=True)
        atomic_json(a.out/'policies.json',manifest)
        print('물체별 정책 저장 완료:',a.out/'policies.json')

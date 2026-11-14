"""Phase features and feedback progression; never supplies taught joint targets."""
import json
import time
from pathlib import Path
import numpy as np


def build_spec(folders):
    plans=[json.loads((Path(p)/'plan.json').read_text())['stages'] for p in folders]
    plans=[[s for s in p if s['kind']!='marker'] for p in plans]
    names=[s['name'] for s in plans[0]]
    if any([s['name'] for s in p]!=names for p in plans):
        raise ValueError('단계 순서가 다른 데모가 섞여 있습니다.')
    spec=[]
    for i,name in enumerate(names):
        stages=[p[i] for p in plans];s=stages[0]
        if len({x['kind'] for x in stages})!=1:raise ValueError('단계 종류 불일치')
        row={'name':name,'kind':s['kind'],'seconds':max(x.get('seconds',0) for x in stages)}
        if s['kind'] in ('move','rotate'):
            modes={x['mode'] for x in stages}
            if len(modes)!=1:raise ValueError('단계 모드 불일치')
            row['mode']=s['mode']; row['pitch']=float(s['pitch'])
            targets=np.array([x['target'] for x in stages])
            row['lower_deg']=(targets.min(0)-5).tolist();row['upper_deg']=(targets.max(0)+5).tolist()
        spec.append(row)
    return spec


def feature(name,spec):
    names=[s['name'] for s in spec]
    if name not in names:raise ValueError(f'알 수 없는 단계: {name}')
    result=np.zeros(len(names),np.float32);result[names.index(name)]=1
    return result


class PhaseProgress:
    def __init__(self,spec):
        self.spec=spec;self.index=0;self.entered=time.monotonic();self.history=[];self.goal=None
    @property
    def finished(self):return self.index>=len(self.spec)
    @property
    def stage(self):return self.spec[self.index]
    def accept(self,goal,mode,grip):
        s=self.stage
        if s['kind'] in ('move','rotate'):
            if mode!=s['mode']:raise ValueError(f"{s['name']}: 예측 모드 {mode} != {s['mode']}")
            active=[3] if mode=='wrist' else [0,1,2]
            if any(not s['lower_deg'][j]<=goal[j]<=s['upper_deg'][j] for j in active):
                raise ValueError(f"{s['name']}: 학습 목표 범위를 벗어난 예측")
        if s['kind'] in ('open','close') and grip!=int(s['kind']=='close'):
            raise ValueError(f"{s['name']}: 집게 예측 불일치")
        self.goal=np.asarray(goal);self.mode=mode
    def observe(self,q,now=None):
        from planner import kin
        now=time.monotonic() if now is None else now
        if now-self.entered>30:raise RuntimeError(f"{self.stage['name']}: 단계 시간 초과")
        if self.goal is None:return False
        s=self.stage;q=np.asarray(q);self.history.append(q);self.history=self.history[-5:]
        if s['kind'] in ('move','rotate'):
            active=[3] if self.mode=='wrist' else [0,1,2]
            reached=(len(self.history)==5 and np.max(np.ptp(self.history,axis=0))<=.5
                     and max(abs(q[j]-self.goal[j]) for j in active)<=2.5
                     and abs(kin.pitch(q)-s['pitch'])<=4)
        else:reached=now-self.entered>=max(.5,s['seconds'])
        if reached:
            self.index+=1;self.entered=now;self.history=[];self.goal=None
        return reached

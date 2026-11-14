import sys,json,collections,argparse
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).absolute().parent))
from act_train import ACTPolicy,Tray4ACTDataset,_batch
torch.set_num_threads(4)
parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
run=args.run; report=json.loads((run/'training_report.json').read_text())
model=ACTPolicy.from_pretrained(str(run/'act_tray4_policy')).to('cpu').eval()
summary={};records=[]
for split,folders in [('validation',report['validation_episode_paths']),('train_sample',report['train_episode_paths'][:2])]:
 stats=collections.defaultdict(list)
 for folder in folders:
  p=Path(folder); ds=Tray4ACTDataset([p],10,phase_spec=report.get('phase_spec')); data=ds.episodes[0]['data']
  steps={r['frame_index']:r for r in map(json.loads,(p/'steps.jsonl').read_text().splitlines())}
  indices=[i for i,f in enumerate(data['frame_index']) if steps[int(f)]['phase'] in ('INITIAL_RAISE','ROTATION_CLEARANCE','OPEN','POINT_GRIPPER_DOWN') or i%5==0]
  for offset in range(0,len(indices),32):
   ids=indices[offset:offset+32];b=_batch(ds,ids,'cpu')
   with torch.inference_mode():out=model.predict_action_chunk({k:b[k] for k in ('observation.state','observation.images.front')}).numpy()
   for i,pred in zip(ids,out):
    truth=data['action'][i]; phase=steps[int(data['frame_index'][i])]['phase'];m=int(truth[5:].argmax());pm=int(pred[0,5:].argmax())
    err=np.rad2deg(np.abs(pred[0,:4]-truth[:4])); stats[phase].append([int(m==pm),*err,float((pred[0,4]>=.5)==(truth[4]>=.5))])
    if phase in ('INITIAL_RAISE','ROTATION_CLEARANCE','OPEN','POINT_GRIPPER_DOWN'):
     records.append(dict(split=split,episode=p.name,index=i,phase=phase,state_deg=np.rad2deg(data['state'][i,:4]).tolist(),goal_deg=np.rad2deg(truth[:4]).tolist(),pred_deg=np.rad2deg(pred[0,:4]).tolist(),mode=m,pred_mode=pm,pred_future_mode=int(pred[-1,5:].argmax())))
  print(split,p.name,len(indices),flush=True)
 summary[split]={k:dict(n=len(v),mode_accuracy=float(np.mean(v,axis=0)[0]),joint_mae_deg=np.mean(v,axis=0)[1:5].tolist(),gripper_accuracy=float(np.mean(v,axis=0)[5])) for k,v in stats.items()}
(run/'phase_diagnostic.json').write_text(json.dumps(dict(summary=summary,early_records=records),indent=2))
print(json.dumps(summary,indent=2))

"""Synthetic tests only; no camera/robot connection and no real success claim."""
import contextlib
import copy
import io
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *
from planner import plan_episode,kin
from control import NativePort,Controller,Sampler,build_step
from review import audit,chunk,export_episode

def config():
    c=default_config()
    # Historical points used ONLY as regression fixtures; installed setup has null pickups.
    c['pickups'].update(plate_1=[8.665,13.097,10.004],tissue_1=[1.517,12.786,9.359],
                        plastic_1=[-.252,9.655,4.302],glass_1=[2.71,10.962,4.815])
    return c

class MemoryLog:
    def __init__(self):self.rows=[]
    def write(self,event,**kw):
        self.rows.append(dict(event=event,**kw));return self.rows[-1]

class FakeRobot:
    JOINT_MODE,JOINT_SPEED,JOINT_ANGLE,WRITE_EXT,JOINT_VELOCITY=1,2,3,4,5
    JOINT_MODE_ANGLE_HORZ,JOINT_MODE_ANGLE_VERT,JOINT_MODE_ANGLE,JOINT_MODE_VELOCITY=4,5,1,0
    def __init__(self,q=None):
        self.q=list(q or [0.,-10.,-140.,60.]);self.goal=self.q.copy();self.connected=True
        self.mode=0;self.writes=[];self.grip=None;self.stuck=False
    def is_connected(self):return self.connected
    def encoder(self,j):return list(self.q)
    def write(self,*args):
        if not self.connected:raise RuntimeError('fake disconnected')
        self.writes.append(args)
        if args[0]==self.JOINT_MODE:self.mode=args[1]
        elif args[0]==self.JOINT_ANGLE:self.goal[args[1]]=args[2]
        elif args[0]==self.WRITE_EXT:self.grip=args[1][2]
        return 8 if args[0]==self.WRITE_EXT else True
    def advance(self,dt):
        if self.stuck or self.mode==0:return
        joints=range(3) if self.mode in (4,5) else range(4)
        for j in joints:
            d=self.goal[j]-self.q[j];self.q[j]+=max(-30*dt,min(30*dt,d))
        if self.mode in (4,5):self.q[3]=(0 if self.mode==4 else -90)-90-self.q[1]-self.q[2]

class SchemaTests(unittest.TestCase):
    def test_template_needs_only_four_pickups(self):
        c=default_config();self.assertEqual(set(c['pickups']),set(ROLES));self.assertEqual(len(c['destinations']),6)
        self.assertNotIn('plastic_2',c['pickups'])
        validate_config(c,False)
        with self.assertRaises(ValueError):validate_config(c)
    def test_latest_destinations_and_cycles(self):
        c=config();self.assertEqual(c['destinations']['glass_place'],[16.492,-1.089,10.792])
        self.assertEqual(c['destinations']['cup_pour'],[13.428,6.225,13.165])
        self.assertEqual(c['pour_cycles'],2)
    def test_selection_four_instances_distance_and_tie(self):
        ds=[dict(kind=k,center=[i,0],box=[i,0,i+1,1]) for i,k in enumerate(KINDS.values())]
        s=selection(ds,[0,0]);self.assertEqual(s['order'],list(ROLES))
        ds[0]['center']=ds[1]['center']=[0,0]
        self.assertEqual(selection(ds,[0,0])['order'][:2],['glass_1','plastic_1'])
        with self.assertRaises(ValueError):selection(ds[:-1],[0,0])
        with self.assertRaises(ValueError):selection(ds+[ds[0]],[0,0])
    def test_stability_requires_reference_and_order(self):
        ds=[dict(kind=k,center=[i*30,0],box=[i,0,i+1,1]) for i,k in enumerate(KINDS.values())]
        a=selection(ds,[0,100]);b=copy.deepcopy(a);self.assertTrue(compatible_selections(a,b))
        b['reference_pixel'][0]+=30;self.assertFalse(compatible_selections(a,b))
    def test_action_separates_mode_and_gripper(self):
        for mode in MODES:
            q,g,m=decode_action(encode_action([0,-10,-140,60],1,mode))
            self.assertEqual(g,1);self.assertEqual(m,mode);self.assertAlmostEqual(q[2],-140)
        with self.assertRaises(ValueError):decode_action([0]*8)
    def test_no_nan_or_missing_point(self):
        c=config();c['pickups']['glass_1']=[float('nan'),1,2]
        with self.assertRaises(ValueError):validate_config(c)

class MotionTests(unittest.TestCase):
    def test_adapter_final_device_values_match_supplied_sdk_methods(self):
        reference=read_json(Path(__file__).with_name('sdk_reference.json'))
        r=FakeRobot()
        for key,value in reference['constants'].items():setattr(r,key,value)
        p=NativePort(r,20,MemoryLog());values={}
        for case in reference['fixtures']:
            r.q=case['observed'];start=len(r.writes)
            p.apply(case['goal'],case['grip'],case['mode'],r.q)
            for w in r.writes[start:]:
                if len(w)==3:values[str(w[0])+':'+str(w[1])]=w[2]
                else:values[str(w[0])]=w[1]
            self.assertEqual(values,case['final_device_values'])
    def test_all_24_orders_complete_and_exact_home(self):
        home=[.26,-9.93,-139.92,59.77]
        for order in itertools.permutations(ROLES):
            plans=plan_episode(config(),list(order),home)
            self.assertEqual([p['id'] for p in plans],list(order))
            for obj in plans:
                ss=obj['stages'];names=[s['name'] for s in ss]
                self.assertEqual(ss[-1]['target'],home)
                self.assertEqual(sum(s['kind']=='close' for s in ss),1)
                self.assertEqual(names.count('RELEASE'),1)
                if obj['id'] in ('plastic_1','glass_1'):
                    for n in ('FRONT_APPROACH','LIFT_UPRIGHT','TILT_DOWN_1','TILT_UP_1','TILT_DOWN_2','TILT_UP_2','RETURN_START'):
                        self.assertIn(n,names)
                if obj['id']=='glass_1':
                    i=names.index('GLASS_POINT_DOWN_AT_PLACE');self.assertEqual(names[i+1],'RELEASE')
                    self.assertEqual(ss[i]['mode'],'wrist');self.assertAlmostEqual(kin.pitch(ss[i]['target']),-90)
                if obj['id']=='plate_1':
                    self.assertIn('TIP_UP_2',names);self.assertIn('POINT_DOWN_2',names)
                if obj['id']=='tissue_1':self.assertIn('FACE_FORWARD',names)
    def test_reject_unreachable_or_nonhorizontal_start(self):
        c=config();c['pickups']['plastic_1']=[100,100,100]
        with self.assertRaises(ValueError):plan_episode(c,list(ROLES),[0,-10,-140,60])
        with self.assertRaises(ValueError):plan_episode(config(),list(ROLES),[0,-10,-140,0])
    def test_wrist_only_command_and_idempotent_hold(self):
        r=FakeRobot();log=MemoryLog();p=NativePort(r,20,log)
        p.apply(r.q,0,'horizontal',r.q);r.writes.clear()
        goal=r.q.copy();goal[3]-=90
        p.apply(goal,1,'wrist',r.q)
        joint=[w for w in r.writes if w[0]==r.JOINT_ANGLE]
        self.assertEqual(joint,[(r.JOINT_ANGLE,3,goal[3])]);self.assertEqual(r.grip,1)
        old=list(r.writes);p.apply(goal,1,'wrist',r.q);self.assertEqual(old,r.writes)
        self.assertEqual(log.rows[-1]['joint_command_mask'],[0,0,0,1])
    def test_nonblocking_gripper_same_sdk_payload(self):
        r=FakeRobot();p=NativePort(r,20,MemoryLog())
        with patch('time.sleep',side_effect=AssertionError('No blocking sleep in adapter')):
            p.apply(r.q,0,'horizontal',r.q);p.apply(r.q,1,'horizontal',r.q)
        ext=[w[1] for w in r.writes if w[0]==r.WRITE_EXT]
        self.assertEqual(ext,[[0x21,0,0,0,0,0,0,0],[0x21,0,1,0,0,0,0,0]])
    def test_mode_change_requires_correct_orientation(self):
        r=FakeRobot();p=NativePort(r,20,MemoryLog())
        with self.assertRaises(RuntimeError):p.apply([0,-10,-140,-30],0,'vertical',r.q)
        self.assertEqual(r.writes,[])
    def test_disconnect_not_retried(self):
        r=FakeRobot();p=NativePort(r,20,MemoryLog());r.connected=False
        with self.assertRaises(RuntimeError):p.read()
        with self.assertRaises(RuntimeError):p.apply(r.q,0,'horizontal',r.q)
        r.connected=True
        with self.assertRaises(RuntimeError):p.apply(r.q,0,'horizontal',r.q)
    def test_timeout_keeps_failure_instead_of_marking_complete(self):
        r=FakeRobot();p=NativePort(r,20,MemoryLog());st=plan_episode(config(),list(ROLES),r.q)[0]['stages']
        with contextlib.redirect_stdout(io.StringIO()):
            c=Controller(p,st,MemoryLog(),'plastic_1');c.tick(r.q,0)
            with self.assertRaises(RuntimeError):c.tick(r.q,61)
        self.assertFalse(c.finished)
    def test_sampler_causal_age_and_failure(self):
        r=FakeRobot();s=Sampler(NativePort(r,20,MemoryLog()),MemoryLog())
        s.samples.extend([dict(read_end_ns=10,qpos_deg=r.q),dict(read_end_ns=30,qpos_deg=r.q)])
        self.assertEqual(s.before(20,100)['read_end_ns'],10)
        with self.assertRaises(RuntimeError):s.before(200_000_000,100)
        s.error='lost'
        with self.assertRaises(RuntimeError):s.before(30,100)

class DataTests(unittest.TestCase):
    def test_pause_chunk_boundary(self):
        import numpy as np
        data={'segment_id':np.array([0,0,1,1]),'action':np.arange(32,dtype=np.float32).reshape(4,8)}
        a,pad=chunk(data,1,4);self.assertEqual(pad.tolist(),[False,True,True,True])
        np.testing.assert_array_equal(a[0],data['action'][1])
        data['segment_id']=np.array([0,0,0,0]);data['frame_index']=np.array([1,2,4,5])
        self.assertEqual(chunk(data,1,4)[1].tolist(),[False,True,True,True])
    def test_step_before_state_and_no_measured_grip(self):
        f=dict(frame_index=1,file='rgb_frames/1.png',host_arrival_ns=100)
        j=dict(read_end_ns=90,qpos_deg=[0,-10,-140,60]);r=dict(action=encode_action(j['qpos_deg'],1,'horizontal'),phase='GRASP',mode='horizontal',command_index=3)
        s=build_step(f,j,0,r,'plastic_1',0,110)
        self.assertEqual(s['observation_state'][4],0);self.assertEqual(s['action'][4],1);self.assertIsNone(s['gripper_measured'])
        self.assertFalse(build_step(f,j,None,r,'plastic_1',0,110)['train_valid'])
    def test_full_four_object_pipeline_mock_motor_feedback_and_audit(self):
        # Full generated trajectories + feedback state machine + 10/20 Hz logs.
        # Synthetic data is ALWAYS tagged simulated and blocked from real export.
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);(folder/'rgb_frames').mkdir();c=config()
            atomic_json(folder/'config.json',c)
            atomic_json(folder/'metadata.json',dict(status='complete',success='user_confirmed',simulated=True,
                        objects=[dict(id=r,success='user_confirmed') for r in ROLES]))
            clock=[1_000_000_000];r=FakeRobot();ev=JsonLog(folder/'events.jsonl');sl=JsonLog(folder/'steps.jsonl');fl=JsonLog(folder/'frames.jsonl');al=JsonLog(folder/'annotations.jsonl')
            missing={r:dict(status='not_detected',bbox_xyxy=None,bbox_corners=None,center_pixel=None) for r in ROLES}
            port=NativePort(r,20,ev);plans=plan_episode(c,list(ROLES),r.q);idx=0
            with patch('time.monotonic_ns',side_effect=lambda:clock[0]),contextlib.redirect_stdout(io.StringIO()):
                for seg,obj in enumerate(plans):
                    ctl=Controller(port,obj['stages'],ev,obj['id']);count=0
                    while not ctl.finished:
                        for _ in range(2):
                            clock[0]+=50_000_000;r.advance(.05)
                            joint=dict(read_start_ns=clock[0],read_end_ns=clock[0],qpos_deg=list(r.q))
                            ev.write('joint_observation',**joint)
                        fn=f'rgb_frames/{idx:06d}.png';(folder/fn).write_bytes(b'synthetic image stub')
                        frame=dict(frame_index=idx,file=fn,host_arrival_ns=clock[0]);fl.write('camera_frame',**frame)
                        prev=port.gripper;res=ctl.tick(list(r.q),clock[0]/1e9)
                        sl.write('step',**build_step(frame,joint,prev,res,obj['id'],seg,clock[0]))
                        al.write('annotation',frame_index=idx,image=fn,objects=missing);idx+=1;count+=1
                        if count>4000:raise AssertionError('controller failed to finish')
                    self.assertLess(max(abs(a-b) for a,b in zip(r.q,[0,-10,-140,60])),2.01)
                    # 1 sec terminal pause: raw frames/states continue, no steps.
                    for pause in range(10):
                        for _ in range(2):
                            clock[0]+=50_000_000
                            ev.write('joint_observation',read_end_ns=clock[0],qpos_deg=list(r.q))
                        fn=f'rgb_frames/{idx:06d}.png';(folder/fn).write_bytes(b'synthetic image stub')
                        fl.write('camera_frame',frame_index=idx,file=fn,host_arrival_ns=clock[0]);al.write('annotation',frame_index=idx,image=fn,objects=missing);idx+=1
            for logger in (ev,sl,fl,al):logger.close()
            report=audit(folder)
            self.assertEqual(report['errors'],['simulation is not a real demonstration'],report)
            self.assertEqual(len(report['active_segments']),4)
            self.assertAlmostEqual(report['joints']['hz'],20)
            with self.assertRaises(ValueError):export_episode(folder)
            # Exercise conversion with an explicitly stubbed PASS report. The
            # public export above correctly rejected the synthetic episode.
            with patch('review.audit',return_value=dict(data_checks_passed=True,errors=[],test_fixture=True)):
                exported=export_episode(folder)
            import numpy as np
            data=np.load(exported,allow_pickle=False)
            self.assertEqual(data['state'].shape[1],5);self.assertEqual(data['action'].shape[1],8)
            self.assertEqual(data['bbox_corners'].shape,(len(data['state']),4,4,2))
            self.assertFalse(data['bbox_valid'].any())
            for boundary in np.flatnonzero(data['segment_id'][1:] != data['segment_id'][:-1]):
                self.assertTrue(chunk(data,int(boundary),4)[1][1:].all())
            aligned=list(jsonl(folder/'frame_alignment.jsonl'))
            self.assertEqual(len(aligned),idx)
            self.assertTrue(any(not x['active_step'] and x['observation_state'][0] is not None for x in aligned))
            self.assertEqual(set(data['segment_id']),{0,1,2,3})
            self.assertGreater(len(data['action']),100)
            rows=list(jsonl(folder/'steps.jsonl'));rows[5]['joint_read_ns']=rows[5]['frame_host_ns']+1
            (folder/'steps.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            self.assertTrue(any('noncausal' in e for e in audit(folder)['errors']))

if __name__=='__main__':unittest.main()

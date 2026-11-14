"""Offline integration tests: no robot, no camera, no YOLO weights."""
import copy
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tray_core import (EpisodeState, ROLES, config_errors, digest, make_selection,
                       validate_selection, validate_model_names, wants_overlay)
from tray_episode import RecordedRobot
import tray_episode
import tray_motion as motion

ROOT=Path(__file__).resolve().parent


def complete_config():
    return json.loads((ROOT/'tray_config.json').read_text())


class CoreTests(unittest.TestCase):
    def test_config_and_generic_model(self):
        c=complete_config()
        self.assertEqual(config_errors(c),[])
        validate_model_names({0:'bowl',1:'cup',2:'tray',3:'ring',4:'tissue'},c['model']['class_names'])
        c['pickups']['glass']=None
        self.assertTrue(config_errors(c))

    def test_initial_distance_order_and_plastic_ids(self):
        c=complete_config()
        objs=[{'kind':kind,'center':[x,y],'box':[x-5,y-5,x+5,y+5]}
              for kind,x,y in [('plastic',80,40),('plate',50,85),('glass',50,50),
                               ('tissue',55,75),('plastic',20,30)]]
        s=make_selection(objs,[50,100],0,1,digest(c))
        validate_selection(s,c)
        self.assertEqual(s['objects'][0]['id'],'plate')
        self.assertEqual(next(o for o in s['objects'] if o['id']=='plastic_1')['center'],[20,30])
        with self.assertRaises(ValueError):make_selection(objs[:-1],[50,100],0,1,digest(c))
        c['speed']=39
        with self.assertRaises(ValueError):validate_selection(s,c)

    def test_cutoff_rearms_five_times_without_stopping_recording(self):
        s=EpisodeState();order=list(ROLES);s.start(order)
        for role in order:
            s.begin_object(role)
            _, before=s.snapshot()
            self.assertTrue(wants_overlay({'state':before}))
            s.open_command()
            self.assertEqual(s.snapshot()[1]['phase'],'before_grasp')
            s.close_command();s.open_command()
            _, after=s.snapshot()
            self.assertEqual(after['phase'],'after_grasp')
            self.assertFalse(wants_overlay({'state':after}))
            self.assertTrue(after['recording'])
            s.complete_object()
            self.assertTrue(s.snapshot()[1]['recording'])
            # Frame state is immutable even if the worker processes it later.
            self.assertTrue(wants_overlay({'state':before}))
        s.finish()
        self.assertEqual(s.finished,order)
        self.assertFalse(s.snapshot()[1]['recording'])

    def test_signal_is_before_sdk_close_and_open_does_not_rearm(self):
        state=EpisodeState();state.start(list(ROLES));state.begin_object('plate')
        calls=[]
        class Raw:
            def close_gripper(self):calls.append(state.snapshot()[1]['phase'])
            def open_gripper(self):calls.append(state.snapshot()[1]['phase'])
        session=SimpleNamespace(state=state,transition=lambda s:None,health=lambda:None,log=lambda *a,**k:None)
        proxy=RecordedRobot(Raw(),session);proxy.close_gripper();proxy.open_gripper()
        self.assertEqual(calls,['after_grasp','after_grasp'])

    def test_failure_does_not_start_next_object(self):
        s=EpisodeState();s.start(list(ROLES));s.begin_object('plate');s.close_command();s.finish(failed=True)
        with self.assertRaises(ValueError):s.begin_object('tissue')
        self.assertEqual(s.finished,[])

    def test_all_120_orders_have_correct_cup_stack_heights(self):
        c=complete_config()
        for order in itertools.permutations(ROLES):
            plans=motion.preflight(c,order,[0,-10,-140,60])
            self.assertEqual(len(plans),5)
            cups=[p for p in plans if p['object_id'].startswith('plastic_')]
            self.assertAlmostEqual(cups[0]['settings']['place_wrist_cm'][2],6.003)
            self.assertAlmostEqual(cups[1]['settings']['place_wrist_cm'][2],8.503)
            for p in plans:
                self.assertEqual(sum(s['kind']=='close' for s in p['stages']),1)
                self.assertAlmostEqual(motion.bench_module.pitch(motion.final_target(p['stages'])),0)

    def test_switching_executor_invalidates_both_mode_caches(self):
        bench=SimpleNamespace(_tray_engine='cup',_native_horizontal_active=True,_plate_mode='vertical')
        saved=motion.plate.execute_motion
        try:
            def fake(robot,b,stage,record):
                self.assertFalse(b._native_horizontal_active);self.assertIsNone(b._plate_mode)
            motion.plate.execute_motion=fake
            motion.execute_stage(None,bench,{'engine':'plate'},None)
        finally:
            motion.plate.execute_motion=saved

    def test_full_runner_one_connection_five_objects_and_five_cutoffs(self):
        c=complete_config();calls=[];s=None
        objs=[{'kind':kind,'center':[x,y],'box':[x-5,y-5,x+5,y+5]}
              for kind,x,y in [('plastic',80,40),('plate',50,85),('glass',50,50),
                               ('tissue',55,75),('plastic',20,30)]]
        class Raw:
            def __init__(self, **kw):
                calls.append('connect');self.q=[0.,-10.,-140.,60.];self.mode=None;self._roboid=self
            def is_connected(self):return True
            def encoder(self, index):
                s.log('joint_packet',read_end_ns=tray_episode.time.monotonic_ns(),joint_deg=list(self.q))
                return list(self.q)
            def unlock(self):self.mode=None
            def lock_horz(self):self.mode=0;self.q[3]=-90-self.q[1]-self.q[2]
            def lock_vert(self):self.mode=-90;self.q[3]=-180-self.q[1]-self.q[2]
            def set_degree(self,*args):
                if len(args)==3:self.q[int(args[0])-1]=args[1]
                else:
                    self.q[:3]=args[:3]
                    self.q[3]=(self.mode-90-self.q[1]-self.q[2]) if self.mode is not None else args[3]
            def close_gripper(self):
                calls.append(('close',s.state.snapshot()[1]['phase']))
            def open_gripper(self):calls.append('open')
            def stop(self):calls.append('stop')
            def dispose(self):calls.append('dispose')
        class Tap:
            def __init__(self,raw,log):
                self.errors=0;self.dropped=0;self.count=1
                log('joint_packet',read_end_ns=tray_episode.time.monotonic_ns(),joint_deg=list(raw.q))
            def close(self):pass
        class Clock:
            def __init__(self):self.t=0
            def monotonic(self):self.t+=.01;return self.t
            def sleep(self,t):self.t+=t
        clock=Clock()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);s=tray_episode.Session(out,c)
            selection=make_selection(objs,[50,100],0,tray_episode.time.monotonic_ns(),digest(c))
            (out/'latest_inventory.json').write_text(json.dumps(dict(selection,stable=True)))
            s.wait=lambda sec:None
            captured=[]
            def transition(snapshot):
                captured.append(snapshot)
                if snapshot['phase']=='before_grasp':
                    (out/'target_status.json').write_text(json.dumps(dict(
                        object_id=snapshot['object_id'],revision=snapshot['revision'],
                        visible=True,frame_ns=tray_episode.time.monotonic_ns())))
            s.transition=transition
            import contextlib,io
            try:
                with patch.dict(sys.modules,{'roboid':SimpleNamespace(Raccoon=Raw),
                                            'packet_tap':SimpleNamespace(PacketTap=Tap)}), \
                     patch('importlib.metadata.version',return_value='1.7.3'), \
                     patch.object(motion.plate,'time',clock), \
                     patch.object(motion.tissue_executor,'time',clock), \
                     patch.object(motion.cup_executor,'time',clock), \
                     contextlib.redirect_stdout(io.StringIO()):
                    tray_episode.execute(s,c,selection,'fake')
            finally:
                s.events.close()
            self.assertEqual(calls.count('connect'),1);self.assertEqual(calls.count('dispose'),1)
            self.assertEqual([x for x in calls if isinstance(x,tuple)],[('close','after_grasp')]*5)
            self.assertEqual(len([x for x in captured if x['phase']=='before_grasp' and x['gripper_command']!='open']),1)
            self.assertEqual(s.state.finished,[o['id'] for o in selection['objects']])
            self.assertEqual(s.state.snapshot()[1]['phase'],'finished')


try:
    import cv2
    import numpy as np
    import tray_vision as vision
    HAS_CV=True
except ImportError:
    HAS_CV=False


@unittest.skipUnless(HAS_CV,'OpenCV 없는 환경: YOLO 가상환경에서 실행하면 이미지 검사 포함')
class VisionTests(unittest.TestCase):
    def test_actual_worker_saves_exactly_what_it_displays_across_five_objects(self):
        import tray_worker
        import contextlib,io
        c=complete_config();image=np.full((160,240,3),80,np.uint8)
        dets=[]
        for i,(role,kind) in enumerate(ROLES.items()):
            x=20+i*40;y=80
            mask=np.zeros((160,240),bool);mask[y-8:y+9,x-8:x+9]=True
            dets.append(dict(kind=kind,center=[x,y],box=[x-8,y-8,x+9,y+9],mask=mask))
        selection=make_selection(dets,[120,150],0,0,digest(c))
        order=[o['id'] for o in selection['objects']]
        state=EpisodeState();state.start(order);frames=[]
        for role in order:
            state.begin_object(role);frames.append(state.snapshot()[1])
            state.close_command();frames.append(state.snapshot()[1])
            state.open_command();frames.append(state.snapshot()[1])
            state.complete_object();frames.append(state.snapshot()[1])
        shown=[];calls=[]
        def detector(*a):calls.append(1);return dets
        fake_model=SimpleNamespace(names=dict(enumerate(c['model']['class_names'].values())),task='segment')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'observation_frames').mkdir();(out/'rgb_frames').mkdir()
            (out/'config.json').write_text(json.dumps(c))
            (out/'selection.json').write_text(json.dumps(selection))
            (out/'capture_done.json').write_text('{}')
            with (out/'camera.jsonl').open('w') as f:
                for i,s in enumerate(frames):
                    name=f'rgb_frames/{i:06d}.png';cv2.imwrite(str(out/name),image)
                    f.write(json.dumps(dict(frame_index=i,file=name,host_arrival_ns=i*100_000_000,state=s))+'\n')
            with patch.dict(sys.modules,{'ultralytics':SimpleNamespace(YOLO=lambda p:fake_model)}), \
                 patch.object(sys,'argv',['tray_worker.py',str(out)]), \
                 patch.object(vision,'detect',detector), \
                 patch.object(cv2,'namedWindow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=-1), \
                 patch.object(cv2,'imshow',side_effect=lambda name,img:shown.append(img.copy())), \
                 contextlib.redirect_stdout(io.StringIO()):
                tray_worker.main()
            rows=[json.loads(x) for x in (out/'observations.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),20);self.assertEqual(len(calls),5)
            for i,row in enumerate(rows):
                stored=cv2.imread(str(out/row['image']))
                self.assertTrue(np.array_equal(stored,shown[i]))
                if row['state']['phase']=='before_grasp':
                    self.assertTrue(row['target_overlay']);self.assertTrue(row['yolo_ran'])
                else:
                    self.assertTrue(np.array_equal(image,stored));self.assertFalse(row['yolo_ran'])

    def test_five_raw_cutoffs_no_detector_calls_and_pixel_equality(self):
        image=np.full((100,100,3),90,np.uint8)
        mask=np.zeros((100,100),bool);mask[40:60,40:60]=True
        selection={'objects':[dict(id=k,kind=v,center=[49.5,49.5],box=[40,40,60,60]) for k,v in ROLES.items()]}
        state=EpisodeState();state.start(list(ROLES));calls=[]
        for role,kind in ROLES.items():
            state.begin_object(role)
            def detect(img):
                calls.append(role);return [dict(kind=kind,center=[49.5,49.5],mask=mask)]
            before={'state':state.snapshot()[1]}
            marked,ran,visible=vision.render_recorded(image,before,selection,detect,25)
            self.assertTrue(ran and visible);self.assertFalse(np.array_equal(image,marked))
            self.assertTrue(np.array_equal(image[49,49],marked[49,49])) # no fill
            state.close_command()
            for _ in range(3):
                raw,ran,visible=vision.render_recorded(image,{'state':state.snapshot()[1]},selection,detect,25)
                self.assertTrue(np.array_equal(raw,image));self.assertFalse(ran or visible)
            state.open_command();state.complete_object()
        self.assertEqual(len(calls),5)

    def test_occluded_tray_hull_accepts_bowl_and_excludes_bin(self):
        tray=np.zeros((100,100),bool);tray[20:90,10:80]=True;tray[50:80,30:60]=False
        area=vision.tray_footprint(tray)
        bowl=np.zeros_like(tray);bowl[50:80,30:60]=True
        tissue=np.zeros_like(tray);tissue[25:35,85:95]=True
        self.assertTrue(vision.on_tray(bowl,area));self.assertFalse(vision.on_tray(tissue,area))
        self.assertEqual(vision.bottom_center(tray),[44,89])

    def test_rgb_material_thresholds(self):
        c=complete_config();mask=np.ones((40,40),bool)
        white=np.full((40,40,3),240,np.uint8)
        dark=np.full((40,40,3),50,np.uint8)
        self.assertEqual(vision.cup_material(white,mask,c)[0],'plastic')
        self.assertEqual(vision.cup_material(dark,mask,c)[0],'glass')
        tiny=np.zeros((40,40),bool);tiny[20,20]=True
        self.assertEqual(vision.cup_material(white,tiny,c)[0],'uncertain_cup')

    def test_no_switch_to_another_same_kind_cup(self):
        t=dict(kind='plastic',center=[10,10])
        self.assertIsNone(vision.match_target([dict(kind='plastic',center=[90,10])],t,25))


if __name__=='__main__':unittest.main(verbosity=2)

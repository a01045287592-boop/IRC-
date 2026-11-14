import math
import io
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cup_pick_real as c

class Clock:
    def __init__(self): self.t=0
    def now(self): return self.t
    def sleep(self,dt): self.t+=dt

class FakeRobot:
    def __init__(self,q): self.q=q; self.calls=[]; self.connected=True
    def is_connected(self): return self.connected
    def encoder(self,index): return tuple(self.q)
    def unlock(self): self.calls.append('unlock')
    def set_degree(self,*a): self.q=list(a[:4]); self.calls.append('move')
    def open_gripper(self): self.calls.append('open')
    def close_gripper(self): self.calls.append('close')
    def stop(self): self.calls.append('stop')

class Tests(unittest.TestCase):
    def test_horizontal_pose(self):
        for p in ([0,12,12],[3,12,12],[-3,12,12],[0,10,14]):
            q=c.horizontal_ik(p)
            self.assertLess(math.dist(c.wrist_fk(q),p),1e-8)
            self.assertLess(abs(c.pitch(q)),1e-8)

    def test_bad_target(self):
        for p in ([0,40,10],[0,0,8],[0,10,float('nan')],[0,10,-1]):
            with self.assertRaises(ValueError): c.horizontal_ik(p)
        with self.assertRaises(ValueError): c.check_q([0,-10,-140,106])

    def test_pick_and_lower(self):
        clock=Clock()
        with tempfile.TemporaryDirectory() as d, patch('cup_pick_real.time.sleep',clock.sleep), patch('cup_pick_real.time.monotonic',clock.now):
            r=FakeRobot(c.horizontal_ik([0,12,12]))
            b=c.Bench(r,10,Path(d)/'pose.json',2,2)
            b.command('mark'); b.command('retreat'); b.command('pick')
            self.assertTrue(b.held)
            self.assertLess(math.dist(c.wrist_fk(r.q),[0,12,14]),1e-8)
            self.assertEqual([x for x in r.calls if x in ('open','close')],['open','close'])
            b.command('lower')
            self.assertFalse(b.held)
            self.assertLess(math.dist(c.wrist_fk(r.q),[0,10,12]),1e-8)

    def test_bad_start_sends_no_motion(self):
        with tempfile.TemporaryDirectory() as d:
            r=FakeRobot(c.horizontal_ik([0,12,12]))
            b=c.Bench(r,10,Path(d)/'pose.json',2,2);b.grasp=[0,12,12]
            with self.assertRaises(ValueError): b.command('pick')
            self.assertEqual(r.calls,[])

    def test_disconnection(self):
        r=FakeRobot([0,-10,-140,60]);r.connected=False
        b=c.Bench(r,10,Path('unused'),2,2)
        with self.assertRaises(RuntimeError):b.read()

    def test_joint_limit_rejected_before_send(self):
        r=FakeRobot([0,-10,-140,60]);b=c.Bench(r,10,Path('unused'),2,2)
        with self.assertRaises(ValueError):b.move_q([0,-10,-140,200])
        self.assertEqual(r.calls,[])

    def test_small_move_does_not_finish_before_settling_delay(self):
        clock=Clock()
        r=FakeRobot([0,-10,-140,60])
        b=c.Bench(r,10,Path('unused'),2,2)
        with patch('cup_pick_real.time.sleep',clock.sleep), patch('cup_pick_real.time.monotonic',clock.now):
            b.move_q([0,-10.3,-139.7,60])
        self.assertGreaterEqual(clock.t,.5)

    def test_stall_stops_and_prints_joint_errors(self):
        class Stalled(FakeRobot):
            def set_degree(self,*a): self.calls.append('move')
        clock=Clock(); r=Stalled([0,-10,-140,60])
        b=c.Bench(r,10,Path('unused'),2,2);out=io.StringIO()
        with patch('cup_pick_real.time.sleep',clock.sleep), patch('cup_pick_real.time.monotonic',clock.now), redirect_stdout(out):
            with self.assertRaises(RuntimeError): b.move_q([0,-13,-137,60])
        self.assertEqual(r.calls.count('move'),1)
        self.assertIn('stop',r.calls)
        for word in ('MOTION_TIMEOUT','TARGET','ACTUAL','ERROR'):
            self.assertIn(word,out.getvalue())

    def test_user_reported_next_step_is_kinematically_valid(self):
        q=[-.18,-10.72,-135.35,56.95]
        p=c.wrist_fk(q); p[1]+=.5
        path=c.line_plan(q,p)
        self.assertLess(math.dist(c.wrist_fk(path[-1]),p),1e-8)


class RegressionTests(unittest.TestCase):
    def test_reported_offset_is_accepted_when_stable(self):
        class Offset(FakeRobot):
            def set_degree(self,*a):
                self.q=[-.088,-13.184,-132.803,56.777];self.calls.append('move')
        clock=Clock();r=Offset([-.088,-12.129,-133.945,56.777])
        b=c.Bench(r,10,Path('unused'),2,2)
        with patch('cup_pick_real.time.sleep',clock.sleep),patch('cup_pick_real.time.monotonic',clock.now):
            b.move_q([-.084,-13.428,-133.868,57.296],cartesian=True)
        self.assertNotIn('stop',r.calls)

    def test_low_joint_error_but_bad_position_rejected(self):
        class Offset(FakeRobot):
            def set_degree(self,*a): self.q=[a[0],a[1]+1.9,a[2]+1.9,a[3]]
        clock=Clock();r=Offset([0,-10,-140,60]);b=c.Bench(r,10,Path('unused'),2,2)
        with patch('cup_pick_real.time.sleep',clock.sleep),patch('cup_pick_real.time.monotonic',clock.now),redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):b.move_q([0,-10,-140,60],cartesian=True)
        self.assertIn('stop',r.calls)

    def test_five_mm_jog_is_one_command(self):
        q=[-.09,-10.02,-139.66,59.41];p=c.wrist_fk(q);p[1]+=.5
        dense=c.line_plan(q,p)
        self.assertEqual(len(c.execution_plan(q,dense)),1)

    def test_full_fixed_motion_with_fake_robot(self):
        import json
        import fixed_cup_demo as f
        cfg=json.loads(Path(f.__file__).with_name('fixed_cup_config.json').read_text())
        start=[-.09,-10.02,-139.66,59.41]
        steps=f.plan(cfg,start)
        clock=Clock();r=FakeRobot(start);b=c.Bench(r,10,Path('unused'),2,2)
        with patch('cup_pick_real.time.sleep',clock.sleep),patch('cup_pick_real.time.monotonic',clock.now):
            for s in steps:
                for q in s.get('targets',[]):
                    c.check_q(q)
                    self.assertLessEqual(max(abs(a-v) for a,v in zip(r.q,q)),8)
                    b.move_q(q,cartesian=True)
        self.assertEqual(sum(s['kind']=='close' for s in steps),1)
        self.assertEqual(sum(s['kind']=='rotate' for s in steps),4)
        self.assertLess(abs(c.pitch(r.q)),1e-8)

if __name__=='__main__': unittest.main()

"""Tests for the optional plate movement completion rule."""
import contextlib
import io
import unittest
from unittest.mock import patch
from test_pipeline import FakeRobot, MemoryLog, kin
from control1 import Controller, NativePort

class Control1Tests(unittest.TestCase):
    def test_plate_move_requires_four_continuous_seconds_in_xyz_box(self):
        r=FakeRobot();p=NativePort(r,20,MemoryLog())
        stage=dict(name='TEST_MOVE',kind='move',target=list(r.q),mode='horizontal',
                   pitch=kin.pitch(r.q),position_tolerance_cm=.3,pitch_tolerance_deg=2.5)
        c=Controller(p,[stage],MemoryLog(),'plate_1')
        with contextlib.redirect_stdout(io.StringIO()), patch.object(kin,'wrist_fk',side_effect=lambda q:q[:3]):
            c.tick(r.q,0)
            inside=[r.q[0]+2,r.q[1],r.q[2],r.q[3]]
            outside=[r.q[0]+2.01,r.q[1],r.q[2],r.q[3]]
            self.assertFalse(c._arrived(inside,1))
            self.assertFalse(c._arrived(inside,4.9))
            self.assertFalse(c._arrived(outside,5))
            self.assertFalse(c._arrived(inside,6))
            self.assertFalse(c._arrived(inside,9.9))
            self.assertTrue(c._arrived(inside,10))
            self.assertFalse(c.finished)
            c.tick(inside,10)
            self.assertTrue(c.finished)

    def test_plate_rotation_keeps_angle_gate(self):
        r=FakeRobot();p=NativePort(r,20,MemoryLog());goal=list(r.q);goal[3]-=10
        stage=dict(name='TEST_ROTATE',kind='rotate',target=goal,mode='wrist',
                   pitch=kin.pitch(goal),position_tolerance_cm=.5,pitch_tolerance_deg=3.)
        c=Controller(p,[stage],MemoryLog(),'plate_1')
        with contextlib.redirect_stdout(io.StringIO()):
            c.tick(r.q,0)
            for t in range(1,7):self.assertFalse(c._arrived(r.q,t))
            for t in range(7,12):arrived=c._arrived(goal,t)
            self.assertTrue(arrived)

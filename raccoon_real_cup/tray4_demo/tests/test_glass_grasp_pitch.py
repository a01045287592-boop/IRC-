"""Offline checks for the optional taught glass-grasp wrist angle."""
import copy
import itertools
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from control import NativePort
from core import ROLES
from planner import kin, plan_episode
from test_pipeline import FakeRobot, MemoryLog, config


HOME=[.44,-9.84,-139.92,59.77]
GLASS_XYZ=[5.331,10.839,4.105]


def latest_config():
    c=config()
    c['pickups'].update(glass_1=GLASS_XYZ,plastic_1=[.728,10.528,4.030],
                        tissue_1=[6.446,13.522,9.841],plate_1=[.431,12.765,10.295])
    c['destinations'].update(plate_pour=[15.102,7.086,15.599],
                             cup_pour=[14.722,6.689,13.661],
                             glass_place=[15.265,.304,13.529])
    return c


class GlassGraspPitchTests(unittest.TestCase):
    def test_newly_taught_glass_pose_is_used(self):
        c=latest_config()
        c['pickups']['glass_1']=[5.621,11.081,4.739]
        c['glass_grasp_pitch_deg']=2.99
        glass=next(obj['stages'] for obj in plan_episode(c,list(ROLES),HOME) if obj['id']=='glass_1')
        match=next(s for s in glass if s['name']=='GLASS_MATCH_GRASP_PITCH')
        self.assertAlmostEqual(kin.pitch(match['target']),2.99)
        self.assertLess(math.dist(kin.wrist_fk(match['target']),c['pickups']['glass_1']),1e-6)
        self.assertAlmostEqual(match['target'][3],68.55,delta=.02)

    def test_taught_pitch_at_grasp_and_horizontal_before_lift(self):
        c=latest_config();c['glass_grasp_pitch_deg']=5.36
        plan=plan_episode(c,list(ROLES),HOME)
        glass=next(obj['stages'] for obj in plan if obj['id']=='glass_1')
        names=[s['name'] for s in glass]
        i=names.index('FRONT_APPROACH')
        self.assertEqual(names[i:i+5],['FRONT_APPROACH','GLASS_MATCH_GRASP_PITCH','GRASP',
                                       'GLASS_STRAIGHTEN_AFTER_GRASP','LIFT_UPRIGHT'])
        approach,tilt,_,straighten,lift=glass[i:i+5]
        self.assertEqual(tilt['mode'],'wrist')
        self.assertEqual(straighten['mode'],'wrist')
        self.assertEqual(tilt['target'][:3],approach['target'][:3])
        self.assertEqual(straighten['target'],approach['target'])
        self.assertAlmostEqual(kin.pitch(tilt['target']),5.36)
        self.assertAlmostEqual(kin.pitch(straighten['target']),0.)
        self.assertAlmostEqual(kin.pitch(lift['target']),0.)
        self.assertLess(math.dist(kin.wrist_fk(tilt['target']),GLASS_XYZ),1e-6)
        self.assertAlmostEqual(tilt['target'][3],74.62,delta=.02)

    def test_all_orders_and_other_objects_unchanged(self):
        original=latest_config()
        adjusted=copy.deepcopy(original);adjusted['glass_grasp_pitch_deg']=5.36
        new_names={'GLASS_MATCH_GRASP_PITCH','GLASS_STRAIGHTEN_AFTER_GRASP'}
        for order in itertools.permutations(ROLES):
            old=plan_episode(original,list(order),HOME)
            new=plan_episode(adjusted,list(order),HOME)
            for old_obj,new_obj in zip(old,new):
                filtered=[s for s in new_obj['stages'] if s['name'] not in new_names]
                self.assertEqual(filtered,old_obj['stages'])

    def test_wrist_rotations_and_close_use_existing_adapter(self):
        c=latest_config();c['glass_grasp_pitch_deg']=5.36
        glass=next(o['stages'] for o in plan_episode(c,list(ROLES),HOME) if o['id']=='glass_1')
        stages={s['name']:s for s in glass}
        q=stages['FRONT_APPROACH']['target']
        robot=FakeRobot(q);port=NativePort(robot,25,MemoryLog())
        port.apply(q,0,'horizontal',q)
        robot.writes.clear()
        goal=stages['GLASS_MATCH_GRASP_PITCH']['target']
        port.apply(goal,0,'wrist',q)
        joint_writes=[w for w in robot.writes if w[0]==robot.JOINT_ANGLE]
        self.assertEqual(joint_writes,[(robot.JOINT_ANGLE,3,goal[3])])
        robot.advance(1.)
        port.apply(goal,1,'wrist',robot.q)
        robot.writes.clear()
        straight=stages['GLASS_STRAIGHTEN_AFTER_GRASP']['target']
        port.apply(straight,1,'wrist',robot.q)
        joint_writes=[w for w in robot.writes if w[0]==robot.JOINT_ANGLE]
        self.assertEqual(joint_writes,[(robot.JOINT_ANGLE,3,straight[3])])
        self.assertEqual(robot.grip,1)

    def test_invalid_pickup_pitch_rejected(self):
        for value in (float('nan'),float('inf'),-10.01,10.01,True,'5.36'):
            c=latest_config();c['glass_grasp_pitch_deg']=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                plan_episode(c,list(ROLES),HOME)


if __name__=='__main__':unittest.main()

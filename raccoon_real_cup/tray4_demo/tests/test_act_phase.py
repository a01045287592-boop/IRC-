import sys,time,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from act_phase import PhaseProgress,feature
class PhaseTests(unittest.TestCase):
 def spec(self):
  return [{'name':'OPEN','kind':'open','seconds':1},
          {'name':'DOWN','kind':'rotate','seconds':0,'pitch':-90,'mode':'wrist','lower_deg':[-5,-45,-95,-45],'upper_deg':[5,-35,-85,-35]}]
 def test_features(self):
  self.assertEqual(feature('DOWN',self.spec()).tolist(),[0,1])
  with self.assertRaises(ValueError):feature('BAD',self.spec())
 def test_hold_not_skipped(self):
  p=PhaseProgress(self.spec());p.accept([0,-40,-90,40],'horizontal',0)
  self.assertFalse(p.observe([0,-40,-90,40],p.entered+.2))
  self.assertTrue(p.observe([0,-40,-90,40],p.entered+1.1))
  self.assertEqual(p.stage['name'],'DOWN')
 def test_wrong_mode_or_goal_rejected(self):
  p=PhaseProgress(self.spec()[1:])
  with self.assertRaises(ValueError):p.accept([0,-40,-90,-40],'horizontal',0)
  with self.assertRaises(ValueError):p.accept([0,-40,-90,40],'wrist',0)
 def test_arrival_requires_pitch(self):
  p=PhaseProgress(self.spec()[1:]);p.accept([0,-40,-90,-40],'wrist',0)
  for _ in range(8):self.assertFalse(p.observe([0,-40,-90,-40]))
 def test_rotation_advances_only_after_stable_arrival(self):
  s=[dict(name='DOWN',kind='rotate',seconds=0,pitch=-90,mode='wrist',lower_deg=[-5,-5,-105,-85],upper_deg=[5,5,-95,-75])]
  p=PhaseProgress(s);p.accept([0,0,-100,-80],'wrist',0)
  for _ in range(4):self.assertFalse(p.observe([0,0,-100,-80]))
  self.assertTrue(p.observe([0,0,-100,-80]));self.assertTrue(p.finished)
if __name__=='__main__':unittest.main()

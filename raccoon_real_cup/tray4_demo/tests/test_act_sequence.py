import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from act_sequence import Sequence

class SequenceTests(unittest.TestCase):
    def test_order(self):
        visible={r:{'distance':d} for r,d in [('plate_1',4),('tissue_1',1),('glass_1',2),('plastic_1',3)]}
        s=Sequence([0,-40,-90,40])
        self.assertEqual(s.choose(visible),'plate_1')
        s.done=['plate_1'];visible.pop('plate_1')
        self.assertEqual(s.choose(visible),'tissue_1')
        s.done.append('tissue_1');visible.pop('tissue_1')
        self.assertEqual(s.choose(visible),'glass_1')
    def test_completion_requires_more_than_absence(self):
        s=Sequence([0,-40,-90,40]);s.target='plate_1'
        for _ in range(20):self.assertFalse(s.observe(s.home,0,{},True))
        s.observe([10,-40,-90,40],1,{},True)
        for _ in range(20):self.assertFalse(s.observe(s.home,0,{},False))
        for _ in range(9):self.assertFalse(s.observe(s.home,0,{},True))
        self.assertTrue(s.observe(s.home,0,{},True))
    def test_missing_remaining_object_stops(self):
        with self.assertRaises(RuntimeError):Sequence([0]*4).choose({})

if __name__=='__main__':unittest.main()

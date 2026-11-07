import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import cv2
    import numpy as np
except ImportError as exc:raise unittest.SkipTest('vision environment requires numpy + OpenCV: '+str(exc))
from core import default_config,ROLES
from vision import inventory,material,annotations,center

def scene():
    im=np.zeros((480,640,3),np.uint8);im[:]=[80,40,80]
    tray=np.zeros((480,640),bool);tray[100:400,100:500]=True
    dets=[dict(kind='tray',mask=tray,box=[100,100,500,400],center=[300,250],confidence=.9,white_ratio=None)]
    for kind,xy in [('plate',(260,320)),('tissue',(370,300)),('plastic',(200,180)),('glass',(400,180))]:
        x,y=xy;m=np.zeros_like(tray);m[y-20:y+20,x-20:x+20]=True
        if kind in ('plastic','plate','tissue'):im[m]=255
        dets.append(dict(kind=kind,mask=m,box=[x-20,y-20,x+20,y+20],center=center(m),confidence=.9,white_ratio=None))
    return im,dets

class VisionTests(unittest.TestCase):
    def test_rgb_white_material_and_unknown(self):
        im,ds=scene();c=default_config()
        self.assertEqual(material(im,ds[3]['mask'],c)[0],'plastic')
        self.assertEqual(material(im,ds[4]['mask'],c)[0],'glass')
        self.assertEqual(material(im,np.zeros((480,640),bool),c)[0],'uncertain_cup')
    def test_four_objects_and_front_bottom_reference(self):
        im,ds=scene();s=inventory(ds)
        self.assertEqual(set(s['order']),set(ROLES));self.assertEqual(s['reference_pixel'],[299,399])
        self.assertEqual(s['order'][0],'plate_1')
    def test_tray_holes_do_not_exclude_bowl(self):
        im,ds=scene();ds[0]['mask'][ds[1]['mask']]=False
        self.assertIn('plate_1',inventory(ds)['order'])
    def test_partial_occlusion_inventory_blocks_not_ignores(self):
        im,ds=scene()
        with self.assertRaises(ValueError):inventory(ds[:-1])
        ds[3]['kind']='uncertain_cup'
        with self.assertRaises(ValueError):inventory(ds)
    def test_missing_boxes_stay_null_and_duplicates_ambiguous(self):
        im,ds=scene();ann=annotations(ds[:-1])
        self.assertIsNone(ann['objects']['glass_1']['bbox_corners'])
        self.assertEqual(ann['objects']['plastic_1']['status'],'detected_semantic_id_unverified')
        ds.append(copy.deepcopy(ds[3]));ann=annotations(ds)
        self.assertEqual(ann['objects']['plastic_1']['status'],'ambiguous_multiple_detections')

if __name__=='__main__':unittest.main()

"""Exercise the actual camera worker with injected synthetic frames/model."""
import contextlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import cv2
    import numpy as np
except ImportError as exc:raise unittest.SkipTest(str(exc))
import camera_worker
import vision
from core import atomic_json,read_json,default_config,jsonl
from test_vision import scene

class CameraTests(unittest.TestCase):
    def test_freeze_turns_off_detector_raw_frames_keep_original_pixels(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);atomic_json(root/'config.json',default_config());atomic_json(root/'camera_control.json',dict(mode='select'))
            img,dets=scene();clock=[1_000_000_000];count=[0];calls=[];images={};stopped=[]
            class Frame:
                def __init__(self,i):self.i=i;self.image=img.copy();self.image[0,0]=i
                def get_data(self):return self.image
                def get_frame_number(self):return self.i
                def get_timestamp(self):return clock[0]/1e6
                def get_frame_timestamp_domain(self):return 'synthetic'
            class Pipeline:
                def start(self,c):return SimpleNamespace(get_device=lambda:SimpleNamespace(get_info=lambda x:'MOCK'))
                def wait_for_frames(self,timeout):
                    count[0]+=1;clock[0]+=100_000_000
                    if count[0]==6:
                        ready=read_json(root/'ready.json');assert ready['ready']
                        atomic_json(root/'camera_control.json',dict(mode='freeze',selection=ready['selection']))
                    if count[0]==9:
                        assert (root/'selection.json').exists()
                        assert not list(jsonl(root/'frames.jsonl'))
                        atomic_json(root/'camera_control.json',dict(mode='record'))
                    if count[0]==17:atomic_json(root/'camera_control.json',dict(mode='stop'))
                    f=Frame(count[0]);images[count[0]]=f.image
                    return SimpleNamespace(get_color_frame=lambda:f)
                def stop(self):stopped.append(True)
            rs=SimpleNamespace(pipeline=Pipeline,config=lambda:SimpleNamespace(enable_stream=lambda *x:None),
                               stream=SimpleNamespace(color=1),format=SimpleNamespace(bgr8=1),camera_info=SimpleNamespace(serial_number=1))
            def fake_detect(*args):calls.append(count[0]);return dets
            with patch.dict(sys.modules,{'pyrealsense2':rs}),patch.object(vision,'load_model',return_value=object()),\
                 patch.object(vision,'detect',side_effect=fake_detect),patch.object(cv2,'imshow'),\
                 patch.object(cv2,'waitKey',return_value=-1),patch.object(cv2,'destroyAllWindows'),\
                 patch('time.monotonic_ns',side_effect=lambda:clock[0]),\
                 patch.object(sys,'argv',['camera_worker.py','--episode',str(root)]):
                camera_worker.main()
            frames=list(jsonl(root/'frames.jsonl'))
            self.assertGreater(len(frames),4);self.assertEqual(stopped,[True]);self.assertFalse((root/'camera_error.json').exists())
            self.assertLess(max(calls),frames[0]['camera_frame_number'])
            self.assertEqual(len(calls),7)
            for f in frames:
                saved=cv2.imread(str(root/f['file']))
                np.testing.assert_array_equal(saved,images[f['camera_frame_number']])

if __name__=='__main__':unittest.main()

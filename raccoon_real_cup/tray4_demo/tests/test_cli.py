import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core import ROLES
from test_pipeline import config
import tray4

class CLITests(unittest.TestCase):
    def call(self,*args,input=None):
        return subprocess.run([sys.executable,str(ROOT/'tray4.py'),*map(str,args)],input=input,text=True,capture_output=True)
    def test_new_four_points_and_offline_check(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'setup.json'
            self.assertEqual(self.call('new','--out',path).returncode,0)
            self.assertNotEqual(self.call('check','--setup',path).returncode,0)
            for r in ROLES:
                result=self.call('set-point','--setup',path,r,*config()['pickups'][r]);self.assertEqual(result.returncode,0,result.stderr)
            result=self.call('check','--setup',path);self.assertEqual(result.returncode,0,result.stderr);self.assertIn('24',result.stdout)
            result=self.call('run','--setup',path);self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--execute',result.stdout)
            before=path.read_bytes();self.assertNotEqual(self.call('new','--out',path).returncode,0);self.assertEqual(before,path.read_bytes())
    def test_collect_without_execute_creates_only_setup(self):
        with tempfile.TemporaryDirectory() as td:
            coords='\n'.join(' '.join(map(str,config()['pickups'][r])) for r in ROLES)+'\n'
            result=self.call('collect','--out',Path(td)/'raw',input=coords)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse((Path(td)/'raw').exists());self.assertEqual(len(list((Path(td)/'tray4_setups').glob('*.json'))),1)
    def test_execute_skips_example_plan_before_sdk_check(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'setup.json';path.write_text(json.dumps(config()))
            args=Namespace(setup=path,execute=True)
            with patch.object(tray4,'plan_episode') as plan, patch.object(tray4,'version',return_value='wrong'):
                with self.assertRaisesRegex(ValueError,'roboid==1.7.3'):
                    tray4.run(args)
                plan.assert_not_called()
    def test_collect_execute_skips_24_order_precheck(self):
        with tempfile.TemporaryDirectory() as td:
            points=[' '.join(map(str,config()['pickups'][role])) for role in ROLES]
            argv=['tray4.py','collect','--execute','--out',str(Path(td)/'raw')]
            with patch.object(sys,'argv',argv), patch('builtins.input',side_effect=points), \
                    patch.object(tray4,'plan_episode') as plan, patch.object(tray4,'run') as run:
                tray4.main()
                plan.assert_not_called()
                run.assert_called_once()
    def test_install_keeps_existing_project_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            project=Path(td);robot=project/'raccoon_real_cup';robot.mkdir()
            original=robot/'cup_pick_real.py';original.write_text('user original\n')
            command=[sys.executable,str(ROOT/'install.py'),'--project',str(project)]
            r=subprocess.run(command,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(original.read_text(),'user original\n')
            self.assertTrue((robot/'tray4_demo/tray4.py').is_file())
            r=subprocess.run(command,capture_output=True,text=True);self.assertNotEqual(r.returncode,0)

if __name__=='__main__':unittest.main()

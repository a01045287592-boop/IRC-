"""Synthetic checks only: no real robot, camera, or collision verification."""
import argparse
import builtins
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plate_clearance_check as check
from core import read_json
from control import NativePort
from planner import kin
from test_pipeline import FakeRobot, MemoryLog, config


HOME = [0., -10., -140., 60.]


class FakeSampler:
    def __init__(self, port, *args):
        self.port = port
        self.error = None
        self.samples = [True]
        self.start = Mock()
        self.close = Mock()

    def before(self, *args):
        return dict(qpos_deg=self.port.read())


class ClearanceTests(unittest.TestCase):
    def test_offline_cli_never_imports_sdk_or_changes_setup(self):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == 'roboid' or name.startswith('roboid.'):
                raise AssertionError('Offline checks must not import the SDK')
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as td:
            setup = Path(td)/'setup.json'
            setup.write_text(json.dumps(config()))
            original = setup.read_bytes()
            output = Path(td)/'checks'
            argv = ['check', '--setup', str(setup), '--home-q', *map(str, HOME),
                    '--speed', '10', '--out', str(output)]
            with patch.object(sys, 'argv', argv), patch('builtins.__import__', guarded_import), \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(check.main(), 0)
            self.assertIn('로봇/카메라 연결 없음', stdout.getvalue())
            self.assertEqual(setup.read_bytes(), original)
            self.assertFalse(output.exists())

    def test_single_stage_keeps_grasp_and_cancels_before_next_command(self):
        robot = FakeRobot(HOME)
        log = MemoryLog()
        port = NativePort(robot, 10, log)
        sampler = FakeSampler(port)
        seconds = [1.]

        def sleep(dt):
            robot.advance(dt)
            seconds[0] += dt

        clock = types.SimpleNamespace(monotonic=lambda: seconds[0],
                    monotonic_ns=lambda: int(seconds[0]*1e9), sleep=sleep)
        stages = [dict(name='MOVE', kind='move', mode='horizontal', pitch=0,
                       target=kin.horizontal_ik([0., 8., 12.]),
                       position_tolerance_cm=.5, pitch_tolerance_deg=2.5),
                  dict(name='GRASP', kind='close', seconds=1.),
                  dict(name='RELEASE', kind='open', seconds=1.)]
        with patch.object(check, 'time', clock), patch.object(check, 'ask', side_effect=['', '', 'q']), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                check.drive_stages(port, sampler, stages, HOME, log)
        self.assertEqual(port.gripper, 1)
        self.assertEqual(robot.grip, 1)
        self.assertEqual([r['phase'] for r in log.rows if r['event']=='phase'], ['MOVE','GRASP'])
        grips = [w[1][2] for w in robot.writes if w[0] == robot.WRITE_EXT]
        self.assertEqual(grips, [0, 1])

    def test_refuses_pose_changes_and_sampler_errors(self):
        with self.assertRaises(RuntimeError):
            check.verify_pose([10., *HOME[1:]], HOME)
        sampler = types.SimpleNamespace(error='disconnected')
        with self.assertRaisesRegex(RuntimeError, 'disconnected'):
            check.sampler_guard(sampler)

    def test_all_plate_stages_execute_in_order_and_return_home_with_mock_feedback(self):
        robot = FakeRobot(HOME)
        log = MemoryLog()
        port = NativePort(robot, 10, log)
        sampler = FakeSampler(port)
        seconds = [1.]

        def sleep(dt):
            robot.advance(dt)
            seconds[0] += dt

        clock = types.SimpleNamespace(monotonic=lambda: seconds[0],
                    monotonic_ns=lambda: int(seconds[0]*1e9), sleep=sleep)
        stages = check.plate_stages(config(), HOME)
        with patch.object(check, 'time', clock), patch.object(check, 'ask', return_value='') as prompt, \
                contextlib.redirect_stdout(io.StringIO()):
            check.drive_stages(port, sampler, stages, HOME, log)
        self.assertEqual(prompt.call_count, len(stages))
        self.assertEqual([r['phase'] for r in log.rows if r['event']=='phase'],
                         [s['name'] for s in stages])
        for actual, expected in zip(robot.q, HOME):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual([w[1][2] for w in robot.writes if w[0]==robot.WRITE_EXT], [0,1,0])

    def test_height_summary_excludes_pauses_between_stages(self):
        low = kin.horizontal_ik([0., 8., 10.])
        high = kin.horizontal_ik([0., 8., 17.])
        rows = [dict(event='phase', phase='LIFT'),
                dict(event='joint_observation', qpos_deg=low),
                dict(event='joint_observation', qpos_deg=high),
                dict(event='segment_motion_complete'),
                dict(event='joint_observation', qpos_deg=HOME),
                dict(event='phase', phase='TRANSIT'),
                dict(event='joint_observation', qpos_deg=high)]
        result = check.minimum_heights(rows)
        self.assertAlmostEqual(result['LIFT'], 10.)
        self.assertAlmostEqual(result['TRANSIT'], 17.)

    def test_cancel_stops_and_disposes_without_claiming_success(self):
        robot = FakeRobot(HOME)
        robot.dispose = Mock()
        sdk = types.SimpleNamespace(Raccoon=Mock(return_value=robot))
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(setup=Path(td)/'setup.json', home_q=HOME,
                                      out=Path(td), port='fake')
            with patch.dict(sys.modules, roboid=sdk), patch.object(check, 'version', return_value=kin.SDK_VERSION), \
                    patch.object(check, 'Sampler', FakeSampler), \
                    patch.object(check, 'drive_stages', side_effect=KeyboardInterrupt('cancel')), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check.execute(args, config(), config()), 1)
            robot.dispose.assert_called_once()
            self.assertEqual(robot.mode, robot.JOINT_MODE_VELOCITY)
            meta = read_json(next(Path(td).glob('*/metadata.json')))
            self.assertEqual(meta['status'], 'aborted')
            self.assertEqual(meta['cup_contact'], 'unverified')
            self.assertFalse(meta['training_episode'])
            self.assertFalse(any(w[0] == robot.WRITE_EXT for w in robot.writes))


if __name__ == '__main__':
    unittest.main()

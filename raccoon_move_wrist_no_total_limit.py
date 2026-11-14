#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RaccoonBot 4-DOF: segmented Cartesian wrist moves in ROBOT FRAME (mm).

REQUIRES existing verified kinematics:
  ~/Raccoonbot_Cafe_Clean/raccoon_real_cup/cup_pick_real.py

Not a TCP controller. Coordinates produced by camera->robot calibration are
TCP points, and MUST NOT be passed to `wrist` without independent TCP->wrist
conversion and verification. No collision sensing / obstacle avoidance.
"""
import argparse
import math
import sys
import time
from pathlib import Path


def vec3(v):
    return '[' + ', '.join(f'{x:+.2f}' for x in v) + ']'


def main():
    parser = argparse.ArgumentParser(description='Explicitly approved segmented wrist XYZ move (millimeters).')
    parser.add_argument('--port', default='/dev/ttyACM0')
    parser.add_argument('--project-root', default=str(Path.home() / 'Raccoonbot_Cafe_Clean'))
    parser.add_argument('--speed', type=int, default=5)
    parser.add_argument('--max-step-mm', type=float, default=15.0, help='Intermediate waypoint spacing in mm')
    parser.add_argument('--execute', action='store_true', help='Enable real motion; otherwise previews only')
    args = parser.parse_args()
    if args.speed < 1 or args.speed > 10:
        parser.error('--speed must be between 1 and 10')
    if not math.isfinite(args.max_step_mm) or not (0 < args.max_step_mm <= 30):
        parser.error('--max-step-mm must be >0 and <=30')
    lib = Path(args.project_root) / 'raccoon_real_cup'
    if not (lib / 'cup_pick_real.py').exists():
        parser.error(f'Kinematics source missing: {lib / "cup_pick_real.py"}')
    sys.path.insert(0, str(lib))
    from cup_pick_real import Bench, wrist_fk, horizontal_ik, line_plan, pitch
    from roboid import Raccoon
    robot = None
    fault = False
    try:
        robot = Raccoon(port_name=args.port)
        deadline = time.monotonic() + 8
        while not robot.is_connected():
            if time.monotonic() > deadline:
                raise RuntimeError('Robot not connected: verify port/power and close other control programs')
            time.sleep(0.1)
        bench = Bench(robot, args.speed, lib / 'cup_pose.json', 2, 2)
        print('=== RACCOON WRIST XYZ (mm) ===')
        print('Commands: status | wrist X Y Z | jog x/y/z +/-millimeters | stop | quit')
        print('IMPORTANT: robot WRIST XYZ only, NOT camera/calibrated TCP XYZ!')
        print('No collision avoidance. Empty gripper, clear space, physical E-stop ready.')
        print('Motion mode:', 'ENABLED (approval required)' if args.execute else 'PREVIEW ONLY')
        while True:
            try:
                parts = input('\nraccoon> ').strip().split()
                if not parts:
                    continue
                cmd = parts[0].lower()
                if cmd in ('quit', 'q', 'exit'):
                    break
                if cmd == 'stop':
                    robot.stop()
                    fault = True
                    print('STOPPED. For safety, exit and reconnect after checking the robot.')
                    continue
                if fault:
                    print('Motion locked; exit and check hardware before restarting.')
                    continue
                if not robot.is_connected():
                    fault = True
                    raise RuntimeError('Robot connection lost; motion locked')
                q = [float(v) for v in bench.read()]
                if len(q) != 4 or not all(math.isfinite(v) for v in q):
                    raise RuntimeError('Invalid encoder reading')
                start_cm = [float(v) for v in wrist_fk(q)]
                start_mm = [10*v for v in start_cm]
                if cmd == 'status':
                    print('joints(deg):', vec3(q))
                    print('wrist(mm):', vec3(start_mm))
                    print('pitch(deg):', f'{pitch(q):+.2f}')
                    continue
                if cmd == 'wrist' and len(parts) == 4:
                    goal_mm = [float(v) for v in parts[1:]]
                elif cmd == 'jog' and len(parts) == 3:
                    axis = parts[1].lower()
                    if axis not in 'xyz' or len(axis) != 1:
                        raise ValueError('jog x/y/z DeltaMM')
                    delta = float(parts[2])
                    if not math.isfinite(delta) or abs(delta) > 5:
                        raise ValueError('jog limited to +/-5mm')
                    goal_mm = list(start_mm)
                    goal_mm['xyz'.index(axis)] += delta
                else:
                    print('Use: wrist 100 120 80  (robot wrist mm), or jog x 3')
                    continue
                if not all(math.isfinite(v) for v in goal_mm):
                    raise ValueError('Non-finite target')
                if abs(pitch(q)) > 3.0:
                    raise ValueError(f'Current pitch {pitch(q):.2f} deg, required horizontal 0 +/-3 deg')
                travel = math.dist(start_mm, goal_mm)
                # Split long paths into short waypoints; keep per-segment tracking checks.
                # No collision checking; all waypoint IK checks are approximate.
                nseg = max(1, math.ceil(travel / args.max_step_mm))
                if nseg > 300:
                    raise ValueError('More than 300 waypoints requested; check units and target coordinate frame')
                goals_mm = [
                    [a + (b-a)*i/nseg for a,b in zip(start_mm, goal_mm)]
                    for i in range(1,nseg+1)
                ]
                goals_cm = [[v/10.0 for v in point] for point in goals_mm]
                total_ik_points = 0
                for g in goals_cm:
                    horizontal_ik(g)
                    segment_path = line_plan(q, g)
                    if not segment_path:
                        raise RuntimeError('Empty IK path for a waypoint')
                    total_ik_points += len(segment_path)
                goal_cm = goals_cm[-1]
                print('START wrist (mm): ', vec3(start_mm))
                print('TARGET wrist (mm):', vec3(goal_mm))
                print(f'TRAVEL {travel:.2f} mm | SEGMENTS {nseg} (<= {args.max_step_mm:.0f}mm each) | planned IK points {total_ik_points}')
                print('CHECK: target is WRIST not TCP; path physically clear, no obstacle detection.')
                if not args.execute:
                    print('PREVIEW ONLY. --execute enables a separate, explicitly approved move.')
                    continue
                if input('For actual movement type EXACTLY MOVE: ').strip() != 'MOVE':
                    print('Cancelled')
                    continue
                # Always reacquire and ensure robot did not move since planning.
                q2 = [float(v) for v in bench.read()]
                current_mm = [v*10 for v in wrist_fk(q2)]
                if not robot.is_connected() or math.dist(current_mm, start_mm) > 3.0 or abs(pitch(q2)) > 3:
                    raise RuntimeError('Robot state changed since planning; move cancelled')
                previous_mm = current_mm
                for i, (g_cm, g_mm) in enumerate(zip(goals_cm,goals_mm), 1):
                    if not robot.is_connected():
                        raise RuntimeError('Connection lost during movement: STOP')
                    q_now = [float(v) for v in bench.read()]
                    now_mm = [10*v for v in wrist_fk(q_now)]
                    if math.dist(now_mm, previous_mm) > 5.0:
                        raise RuntimeError('Unexpected robot position drift: STOP')
                    horizontal_ik(g_cm)
                    print(f'[{i}/{nseg}] wrist target(mm):',vec3(g_mm),flush=True)
                    bench.move_line(g_cm)
                    measured_q = [float(v) for v in bench.read()]
                    measured_mm = [10*v for v in wrist_fk(measured_q)]
                    if math.dist(measured_mm, g_mm) > 5.0:
                        raise RuntimeError(f'Waypoint tracking error {math.dist(measured_mm,g_mm):.1f}mm: STOP')
                    previous_mm = measured_mm
                q_end = [float(v) for v in bench.read()]
                actual_mm = [v*10 for v in wrist_fk(q_end)]
                err = math.dist(actual_mm, goal_mm)
                print('Actual wrist FK (mm):', vec3(actual_mm))
                print(f'Wrist FK endpoint error: {err:.2f} mm (NOT camera TCP error)')
                if err > 3:
                    fault = True
                    print('OFF TARGET: further movements locked; inspect physical robot')
            except (ValueError, RuntimeError) as e:
                print('DENIED/ERROR:', e)
                if isinstance(e, RuntimeError):
                    fault = True
                    robot.stop()
            except (KeyboardInterrupt, EOFError):
                print('\nStopped by operator')
                break
    finally:
        if robot is not None:
            try: robot.stop()
            except Exception: pass
            try: robot.dispose()
            except Exception: pass


if __name__ == '__main__':
    main()

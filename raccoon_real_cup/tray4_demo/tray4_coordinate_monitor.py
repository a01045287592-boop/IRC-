#!/usr/bin/env python3
"""Live RealSense depth + Raccoon wrist monitor and calibrated camera-frame jogging.

Collect matching 3D points, fit a rigid camera-to-robot transform, then use
camgoto. OpenCV clicks only sample points; robot movement requires a terminal
command.
"""
import argparse
import json
import math
import queue
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from cup_pick_real import (  # noqa: E402
    Bench, SDK_VERSION, check_q, horizontal_ik, pitch, wrist_fk,
)

DEFAULT_SETUP = ROOT.parents[1] / "raccoon_dataset/tray4_setups/setup_20261108_onepour_reteach.json"
DEFAULT_CAL = ROOT.parents[1] / "raccoon_dataset/tray4_setups/tray4_camera_to_robot.json"
DEFAULT_EXTRINSIC = ROOT.parents[1] / "raccoon_dataset/tray4_setups/tray4_camera_to_robot_3d.json"
ROLES = ("plastic_1", "glass_1", "plate_1", "tissue_1")
LABELS = {"plastic_1": "플라스틱 컵", "glass_1": "유리컵", "plate_1": "접시", "tissue_1": "휴지"}


def map_pixel(calibration, role, pixel):
    row = calibration["homography_by_role"][role]
    u, v = pixel
    bounds = row["pixel_bounds"]
    margin = calibration.get("extrapolation_margin_px", 0)
    if not (bounds[0][0]-margin <= u <= bounds[0][1]+margin and
            bounds[1][0]-margin <= v <= bounds[1][1]+margin):
        raise ValueError("선택한 픽셀이 해당 물체의 보정 범위 밖입니다.")
    h = row["homography_pixel_to_robot_xy"]
    den = h[2][0]*u + h[2][1]*v + h[2][2]
    if abs(den) < 1e-9:
        raise ValueError("보정 변환의 분모가 0에 가깝습니다.")
    return [(h[0][0]*u+h[0][1]*v+h[0][2])/den,
            (h[1][0]*u+h[1][1]*v+h[1][2])/den]


def fit_rigid_transform(points):
    """Fit robot_xyz_cm = R @ camera_xyz_cm + t using Kabsch alignment."""
    import numpy as np
    if len(points) < 6:
        raise ValueError("3D 보정은 최소 6개 좌표 쌍이 필요합니다.")
    cam = np.asarray([p["camera_xyz_cm"] for p in points], dtype=np.float64)
    rob = np.asarray([p["robot_xyz_cm"] for p in points], dtype=np.float64)
    cam0, rob0 = cam.mean(axis=0), rob.mean(axis=0)
    ac, ar = cam-cam0, rob-rob0
    if np.linalg.matrix_rank(ac, tol=1e-4) < 2 or np.linalg.matrix_rank(ar, tol=1e-4) < 2:
        raise ValueError("보정점이 한 직선에 몰렸습니다. 공간에서 서로 다른 위치를 측정하세요.")
    u, _, vt = np.linalg.svd(ac.T @ ar)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = rob0 - rotation @ cam0
    pred = (rotation @ cam.T).T + translation
    errors = np.linalg.norm(pred-rob, axis=1)
    return rotation, translation, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CAL)
    parser.add_argument("--extrinsic", type=Path, default=DEFAULT_EXTRINSIC,
                        help="3D camera-to-robot transform JSON")
    parser.add_argument("--role", choices=ROLES, help="클릭한 픽셀을 이 물체의 평면 보정으로 XY 변환")
    parser.add_argument("--speed", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.speed <= 20:
        parser.error("--speed는 1~20 범위여야 합니다.")
    if version("roboid") != SDK_VERSION:
        parser.error(f"roboid=={SDK_VERSION} 환경에서 실행하세요.")

    cfg = json.loads(args.setup.read_text(encoding="utf-8"))
    calibration = None
    if args.calibration.is_file():
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if args.role and (not calibration or args.role not in calibration.get("homography_by_role", {})):
        print("주의: 해당 물체의 픽셀→로봇 XY 보정값이 없어 변환을 표시하지 않습니다.", flush=True)

    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
        from roboid import Raccoon
    except ImportError as exc:
        raise SystemExit(f"필요한 라이브러리를 현재 Python에서 불러오지 못했습니다: {exc}")

    robot = None
    pipeline = None
    camera_started = False
    input_queue = queue.Queue()
    result_queue = queue.Queue()
    robot_lock = threading.Lock()
    state = {"q": None, "xyz": None, "pitch": None, "busy": False, "click": None, "click_seq": 0, "cam_xyz": None,
             "depth_m": None, "mapped_xy": None, "mapped_xyz": None, "xyz_delta": None,
             "raw_target_cam_cm": None, "raw_error_cm": None, "error": None}
    calibration_points = []
    closing = threading.Event()

    def input_reader():
        while not closing.is_set():
            try:
                line = input("좌표 명령> ")
            except (EOFError, KeyboardInterrupt):
                input_queue.put("quit")
                return
            input_queue.put(line)

    def command_worker():
        while not closing.is_set():
            try:
                line = input_queue.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                run_command(line)
            except Exception as exc:
                state["error"] = str(exc)
                print("명령 거부/오류:", exc, flush=True)

    def run_command(line):
        fields = line.split()
        if not fields:
            return
        cmd = fields[0].lower()
        if cmd in ("q", "quit", "exit"):
            closing.set()
            return
        if cmd in ("help", "?"):
            print("명령: status | horz | jog x|y|z cm(최대 ±0.5) | goto X Y Z(로봇 cm) | record | pairs | fit | camgoto(보정 후) | quit", flush=True)
            print("좌표 대응 측정: 로봇 손목 기준점과 카메라에서 클릭할 물리 점을 일치시키고 클릭 → record. 공간상 다른 위치에서 6쌍 이상 모은 뒤 pairs와 fit을 실행하세요.", flush=True)
            print("goto/jog/camgoto는 손목 수평 pitch 0±3°일 때만, 이동당 최대 2cm로 동작합니다. 이동 공간을 비우고 작은 단계로 확인하세요.", flush=True)
            return
        if cmd == "status" and len(fields) == 1:
            with robot_lock:
                q = check_q(list(robot.encoder(-1)))
            xyz = wrist_fk(q)
            state.update(q=q, xyz=xyz, pitch=pitch(q))
            print("관절(deg):", [round(v, 2) for v in q], "손목 XYZ(cm):", [round(v, 3) for v in xyz], "pitch:", round(pitch(q), 2), flush=True)
            if args.role and args.role in cfg.get("pickups", {}):
                target = cfg["pickups"][args.role]
                err = [xyz[i]-target[i] for i in range(3)]
                print(f"설정 {LABELS[args.role]} 픽업 XYZ:", target, "현재-설정 오차(cm):", [round(v, 3) for v in err], flush=True)
            return
        if cmd == "horz" and len(fields) == 1:
            with robot_lock:
                bench.horizontal()
                q = check_q(list(robot.encoder(-1)))
            state.update(q=q, xyz=wrist_fk(q), pitch=pitch(q))
            print("수평 자세 완료. 손목 XYZ(cm):", [round(v, 3) for v in wrist_fk(q)], flush=True)
            return
        if cmd == "record" and len(fields) == 1:
            click = state.get("click")
            cam_xyz = state.get("cam_xyz")
            if click is None or cam_xyz is None:
                raise ValueError("카메라 창에서 보정점을 먼저 클릭하세요. 유효한 깊이가 필요합니다.")
            with robot_lock:
                q = check_q(list(robot.encoder(-1)))
            robot_xyz = wrist_fk(q)
            camera_xyz_cm = [float(v)*100. for v in cam_xyz]
            calibration_points.append({
                "pixel": [int(click[0]), int(click[1])],
                "camera_xyz_cm": camera_xyz_cm,
                "robot_xyz_cm": [float(v) for v in robot_xyz],
                "q_deg": [float(v) for v in q],
            })
            raw_delta = [robot_xyz[i]-camera_xyz_cm[i] for i in range(3)]
            print(f"좌표 쌍 {len(calibration_points)} 저장", flush=True)
            print("  observation camera XYZ (cm):", [round(v,3) for v in camera_xyz_cm], flush=True)
            print("  robot wrist XYZ (cm):      ", [round(v,3) for v in robot_xyz], flush=True)
            print("  단순 숫자 차 robot-camera (cm):", [round(v,3) for v in raw_delta],
                  "| 참고용: 좌표축/원점이 달라 이 차이만으로 보정하면 안 됩니다.", flush=True)
            print("  이 점이 실제로 같은 물리 위치의 대응점인지 확인하세요.", flush=True)
            return
        if cmd == "pairs" and len(fields) == 1:
            if not calibration_points:
                raise ValueError("저장된 좌표 쌍이 없습니다. 카메라에서 손목 기준점을 클릭하고 record 하세요.")
            print(f"저장된 대응점: {len(calibration_points)}개", flush=True)
            for i, point in enumerate(calibration_points, 1):
                camera_xyz = point["camera_xyz_cm"]
                robot_xyz = point["robot_xyz_cm"]
                raw_delta = [robot_xyz[j]-camera_xyz[j] for j in range(3)]
                print(f"{i:02d} pixel={point['pixel']} camera={ [round(v,3) for v in camera_xyz] } cm | "
                      f"robot={ [round(v,3) for v in robot_xyz] } cm | "
                      f"robot-camera 숫자 차={ [round(v,3) for v in raw_delta] } cm", flush=True)
            return
        if cmd == "fit" and len(fields) == 1:
            rotation, translation, errors = fit_rigid_transform(calibration_points)
            payload = {
                "schema": 1,
                "method": "rigid camera optical XYZ cm -> Raccoon wrist XYZ cm (Kabsch)",
                "camera_serial": camera_serial,
                "camera_axes": "RealSense optical frame: +X right, +Y down, +Z forward",
                "robot_axes": "Raccoon wrist frame: +X left, +Y forward, +Z up; cm",
                "rotation_camera_to_robot": rotation.tolist(),
                "translation_camera_to_robot_cm": translation.tolist(),
                "samples": calibration_points,
                "point_errors_cm": errors.tolist(),
                "rms_error_cm": float((sum(float(e*e) for e in errors)/len(errors))**.5),
                "max_error_cm": float(max(errors)),
            }
            args.extrinsic.parent.mkdir(parents=True, exist_ok=True)
            temp = args.extrinsic.with_name(args.extrinsic.name+".tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            temp.replace(args.extrinsic)
            print(f"3D 좌표 변환 저장: {args.extrinsic.resolve()} | RMS {payload['rms_error_cm']:.3f}cm | 최대 {payload['max_error_cm']:.3f}cm", flush=True)
            print("각 대응점의 변환 후 잔차 (예측 robot XYZ - 측정 robot XYZ):", flush=True)
            for i, (point, error) in enumerate(zip(calibration_points, errors), 1):
                predicted = rotation @ np.asarray(point["camera_xyz_cm"], dtype=np.float64) + translation
                residual = predicted - np.asarray(point["robot_xyz_cm"], dtype=np.float64)
                print(f"  {i:02d}: 잔차 XYZ={ [round(float(v),3) for v in residual] } cm | 거리={float(error):.3f} cm", flush=True)
            print("오차가 크면 이 파일로 camgoto하지 말고 점 부착 위치와 대응점을 다시 측정하세요.", flush=True)
            return
        if cmd == "camgoto" and len(fields) == 4:
            if not args.extrinsic.is_file():
                raise ValueError(f"3D 변환 파일이 없습니다. 먼저 record 6회 이상 후 fit 하세요: {args.extrinsic}")
            ext = json.loads(args.extrinsic.read_text(encoding="utf-8"))
            if ext.get("schema") != 1 or ext.get("camera_serial") != camera_serial:
                raise ValueError("3D 변환 파일의 카메라 serial이 현재 카메라와 다릅니다.")
            cam_point = [float(v) for v in fields[1:]]
            if not all(math.isfinite(v) for v in cam_point):
                raise ValueError("카메라 목표는 유한한 cm 숫자여야 합니다.")
            rotation = ext["rotation_camera_to_robot"]
            translation = ext["translation_camera_to_robot_cm"]
            robot_goal = [sum(rotation[i][j]*cam_point[j] for j in range(3))+translation[i] for i in range(3)]
            with robot_lock:
                q = check_q(list(robot.encoder(-1)))
                if abs(pitch(q)) > 3:
                    raise ValueError("camgoto는 수평 pitch 0±3° 자세에서만 동작합니다. 집게가 비었고 주변이 비었는지 확인한 뒤 horz를 사용하세요.")
                distance = math.dist(wrist_fk(q), robot_goal)
                if distance > 2.0:
                    raise ValueError(f"변환된 로봇 목표 { [round(v,3) for v in robot_goal] }cm가 현재 손목에서 {distance:.2f}cm입니다. 한 번 이동 한도 2cm를 넘습니다.")
                bench.move_line(robot_goal)
                new_q = check_q(list(robot.encoder(-1)))
            state.update(q=new_q, xyz=wrist_fk(new_q), pitch=pitch(new_q))
            print("카메라 목표 cm:", cam_point, "→ 로봇 손목 목표 cm:", [round(v,3) for v in robot_goal], flush=True)
            print("도착 손목 cm:", [round(v,3) for v in wrist_fk(new_q)], flush=True)
            return
        if cmd == "move_click" and len(fields) == 1:
            observed = state.get("cam_xyz")
            if observed is None:
                raise ValueError("먼저 카메라 창에서 목표 지점을 클릭해 유효한 깊이 XYZ를 읽으세요.")
            raw_target = [float(v)*100. for v in observed]
            fields = ["rawgoto", *[str(v) for v in raw_target]]
            print("클릭 지점 카메라 XYZ(cm)를 변환 없이 로봇 목표 숫자로 사용:", [round(v,3) for v in raw_target], flush=True)
            cmd = "rawgoto"
        if cmd == "rawgoto" and len(fields) == 4:
            # Deliberately bypass the camera-to-robot transform for a comparison trial.
            # Identical numeric XYZ values are interpreted as robot wrist centimeters.
            raw_target = [float(v) for v in fields[1:]]
            if not all(math.isfinite(v) for v in raw_target):
                raise ValueError("목표는 유한한 숫자여야 합니다.")
            state["raw_target_cam_cm"] = raw_target
            state["raw_error_cm"] = None
            with robot_lock:
                q = check_q(list(robot.encoder(-1)))
                if abs(pitch(q)) > 3:
                    raise ValueError("rawgoto는 수평 pitch 0±3° 자세에서만 동작합니다.")
                distance = math.dist(wrist_fk(q), raw_target)
                if distance > 2.0:
                    raise ValueError(f"보정 없이 같은 숫자를 로봇 XYZ(cm)로 보냈을 때 현재 위치에서 {distance:.2f}cm입니다. 1회 이동 제한 2cm를 넘으므로 더 가까운 중간 목표를 입력하세요.")
                bench.move_line(raw_target)
                new_q = check_q(list(robot.encoder(-1)))
            state.update(q=new_q, xyz=wrist_fk(new_q), pitch=pitch(new_q))
            print("보정 없이 입력값을 로봇 XYZ(cm)로 전송:", raw_target, flush=True)
            print("실제 손목 XYZ(cm):", [round(v,3) for v in wrist_fk(new_q)], flush=True)
            print("이제 카메라 창에서 그리퍼 기준 표시점을 클릭한 뒤 error를 입력하세요.", flush=True)
            return
        if cmd == "error" and len(fields) == 1:
            target = state.get("raw_target_cam_cm")
            observed = state.get("cam_xyz")
            if target is None:
                raise ValueError("먼저 rawgoto X Y Z로 비교 목표를 입력하세요.")
            if observed is None:
                raise ValueError("카메라 창에서 그리퍼 기준 표시점을 클릭해 깊이 좌표를 읽으세요.")
            observed_cm = [float(v)*100. for v in observed]
            delta = [observed_cm[i]-target[i] for i in range(3)]
            error_norm = math.sqrt(sum(v*v for v in delta))
            state["raw_error_cm"] = delta
            print("카메라 목표 XYZ(cm):", [round(v,3) for v in target], flush=True)
            print("카메라 관측 그리퍼 XYZ(cm):", [round(v,3) for v in observed_cm], flush=True)
            print("관측-목표 오차 XYZ(cm):", [round(v,3) for v in delta], "거리 오차(cm):", round(error_norm,3), flush=True)
            return
        if cmd in ("jog", "goto", "joint"):
            with robot_lock:
                q = check_q(list(robot.encoder(-1)))
                if abs(pitch(q)) > 3:
                    raise ValueError("현재 그리퍼 pitch가 수평 0±3°가 아닙니다. 안전한 수평 자세에서만 XYZ 이동을 허용합니다.")
                if cmd == "jog" and len(fields) == 3:
                    axis = fields[1].lower()
                    delta = float(fields[2])
                    if axis not in ("x", "y", "z") or not math.isfinite(delta) or abs(delta) > .5:
                        raise ValueError("사용법: jog x|y|z 거리_cm (각 명령 최대 ±0.5cm)")
                    goal = wrist_fk(q)
                    goal["xyz".index(axis)] += delta
                elif cmd == "goto" and len(fields) == 4:
                    goal = [float(v) for v in fields[1:]]
                    if not all(math.isfinite(v) for v in goal):
                        raise ValueError("좌표는 유한한 숫자여야 합니다.")
                    if math.dist(wrist_fk(q), goal) > 2.0:
                        raise ValueError("한 번의 goto는 현재 손목에서 최대 2cm입니다. jog 또는 가까운 중간 좌표를 사용하세요.")
                elif cmd == "joint" and len(fields) == 3:
                    joint, delta = int(fields[1]), float(fields[2])
                    if joint not in (1, 2, 3, 4) or not math.isfinite(delta) or abs(delta) > 2:
                        raise ValueError("사용법: joint 1~4 각도_deg (최대 ±2°)")
                    q[joint-1] += delta
                    check_q(q)
                    bench.move_q(q)
                    new_q = check_q(list(robot.encoder(-1)))
                    state.update(q=new_q, xyz=wrist_fk(new_q), pitch=pitch(new_q))
                    print("현재 손목 XYZ(cm):", [round(v, 3) for v in wrist_fk(new_q)], "pitch:", round(pitch(new_q), 2), flush=True)
                    return
                else:
                    raise ValueError("명령 형식을 확인하세요. help 입력 시 사용법을 표시합니다.")
                # Preflight through the same checked horizontal Cartesian planner.
                bench.move_line(goal)
                new_q = check_q(list(robot.encoder(-1)))
            state.update(q=new_q, xyz=wrist_fk(new_q), pitch=pitch(new_q))
            print("이동 완료. 손목 XYZ(cm):", [round(v, 3) for v in wrist_fk(new_q)], "pitch:", round(pitch(new_q), 2), flush=True)
            return
        raise ValueError("알 수 없는 명령입니다. help를 입력하세요.")

    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)
            state["click_seq"] += 1

    try:
        robot = Raccoon(port_name=args.port)
        deadline = time.monotonic() + 6
        while not robot.is_connected() and time.monotonic() < deadline:
            time.sleep(.05)
        if not robot.is_connected():
            raise RuntimeError(f"로봇 연결 실패: {args.port}")
        bench = Bench(robot, args.speed, args.setup.with_suffix(".json"), 2.0, 2.0)
        with robot_lock:
            q = check_q(list(robot.encoder(-1)))
        state.update(q=q, xyz=wrist_fk(q), pitch=pitch(q))

        pipeline = rs.pipeline()
        rs_config = rs.config()
        serial = cfg.get("camera_serial")
        if serial:
            rs_config.enable_device(serial)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipeline.start(rs_config)
        camera_started = True
        align = rs.align(rs.stream.color)
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        camera_serial = profile.get_device().get_info(rs.camera_info.serial_number)
        extrinsic_live = None
        if args.extrinsic.is_file():
            candidate = json.loads(args.extrinsic.read_text(encoding="utf-8"))
            if candidate.get("camera_serial") == camera_serial and candidate.get("schema") == 1:
                extrinsic_live = candidate
        print(f"연결: 카메라 {camera_serial}, 로봇 {args.port}", flush=True)
        print("카메라 창에서 지점을 클릭하면 그 지점의 깊이와 카메라 XYZ를 표시합니다.", flush=True)
        print("카메라 클릭은 로봇을 움직이지 않습니다. 로봇 이동은 터미널 명령으로만 합니다. help로 사용법을 확인하세요.", flush=True)

        cv2.namedWindow("TRAY4 coordinate monitor", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("TRAY4 coordinate monitor", mouse)
        threading.Thread(target=input_reader, daemon=True).start()
        threading.Thread(target=command_worker, daemon=True).start()
        last_robot_read = 0.0
        reported_click_seq = 0
        while not closing.is_set():
            frames = pipeline.wait_for_frames(1000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            image = np.asanyarray(color_frame.get_data()).copy()
            click = state.get("click")
            if click:
                u, v = click
                samples = []
                for yy in range(max(0, v-2), min(depth_frame.get_height(), v+3)):
                    for xx in range(max(0, u-2), min(depth_frame.get_width(), u+3)):
                        distance = depth_frame.get_distance(xx, yy)
                        if math.isfinite(distance) and distance > 0:
                            samples.append(distance)
                depth_m = float(np.median(samples)) if samples else 0.0
                cam_xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], depth_m) if depth_m else None
                mapped_xyz = None
                xyz_delta = None
                if cam_xyz and extrinsic_live:
                    camera_cm = [float(v)*100. for v in cam_xyz]
                    rotation = extrinsic_live["rotation_camera_to_robot"]
                    translation = extrinsic_live["translation_camera_to_robot_cm"]
                    mapped_xyz = [sum(rotation[i][j]*camera_cm[j] for j in range(3))+translation[i] for i in range(3)]
                    if state.get("xyz"):
                        xyz_delta = [state["xyz"][i]-mapped_xyz[i] for i in range(3)]
                state.update(depth_m=depth_m or None, cam_xyz=cam_xyz,
                             mapped_xyz=mapped_xyz, xyz_delta=xyz_delta)
                if cam_xyz and state.get("click_seq", 0) != reported_click_seq:
                    camera_cm = [float(v)*100. for v in cam_xyz]
                    print(f"카메라 클릭 pixel=({u},{v}), depth={depth_m:.4f}m, XYZ(cm)={[round(x,3) for x in camera_cm]}", flush=True)
                    reported_click_seq = state["click_seq"]
                if args.role and calibration and args.role in calibration.get("homography_by_role", {}):
                    try:
                        state["mapped_xy"] = map_pixel(calibration, args.role, [u, v])
                    except ValueError as exc:
                        state["mapped_xy"] = None
                        state["error"] = str(exc)
                else:
                    state["mapped_xy"] = None
                cv2.drawMarker(image, click, (255, 0, 255), cv2.MARKER_CROSS, 22, 3)

            now = time.monotonic()
            if now - last_robot_read >= .25 and robot_lock.acquire(blocking=False):
                try:
                    q = check_q(list(robot.encoder(-1)))
                    state.update(q=q, xyz=wrist_fk(q), pitch=pitch(q))
                except Exception as exc:
                    state["error"] = f"로봇 상태 읽기 실패: {exc}"
                finally:
                    robot_lock.release()
                last_robot_read = now

            y = 22
            lines = [
                f"Camera {camera_serial} | aligned depth | 640x480",
                f"Robot wrist XYZ cm: {state['xyz'] and [round(x,3) for x in state['xyz']]} | pitch {state['pitch'] and round(state['pitch'],2)} deg",
                f"Robot joints deg: {state['q'] and [round(x,2) for x in state['q']]}",
            ]
            if click:
                lines.append(f"Clicked pixel: {click} | depth: {state['depth_m'] and round(state['depth_m'],4)} m")
                camera_cm = [round(x*100.,2) for x in state["cam_xyz"]] if state.get("cam_xyz") else None
                lines.append(f"Camera XYZ cm: {camera_cm}")
                if state.get("mapped_xyz"):
                    lines.append(f"Mapped robot XYZ cm: {[round(x,3) for x in state['mapped_xyz']]}")
                    lines.append(f"Actual wrist - mapped XYZ cm: {[round(x,3) for x in state['xyz_delta']]}")
                if state.get("raw_target_cam_cm"):
                    lines.append(f"Raw target camera XYZ cm: {[round(x,3) for x in state['raw_target_cam_cm']]}")
                    if state.get("raw_error_cm"):
                        lines.append(f"Observed camera error XYZ cm: {[round(x,3) for x in state['raw_error_cm']]}")
                if state.get("mapped_xy"):
                    lines.append(f"Planar mapped robot XY cm ({args.role}): {[round(x,3) for x in state['mapped_xy']]}")
                    if state.get("xyz"):
                        delta = [state["mapped_xy"][i]-state["xyz"][i] for i in range(2)]
                        lines.append(f"Mapped XY - actual wrist XY cm: {[round(x,3) for x in delta]}")
            if args.role:
                target = cfg.get("pickups", {}).get(args.role)
                if target and state.get("xyz"):
                    err = [state["xyz"][i]-target[i] for i in range(3)]
                    lines.append(f"Setup {args.role} XYZ cm: {target} | actual-target: {[round(x,3) for x in err]}")
            if state.get("error"):
                lines.append(state["error"])
            lines.append("Click = depth sample | terminal: help / status / jog / goto / joint / quit")
            for line in lines:
                cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 3, cv2.LINE_AA)
                cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (80, 255, 120), 1, cv2.LINE_AA)
                y += 22
            cv2.imshow("TRAY4 coordinate monitor", image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                closing.set()
    except KeyboardInterrupt:
        closing.set()
    finally:
        closing.set()
        if pipeline is not None and camera_started:
            pipeline.stop()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        if robot is not None:
            try:
                robot.stop()
                time.sleep(.15)
                robot.dispose()
            except Exception:
                pass


if __name__ == "__main__":
    main()

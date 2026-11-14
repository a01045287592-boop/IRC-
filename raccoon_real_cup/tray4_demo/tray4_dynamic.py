#!/usr/bin/env python3
"""Calibrate planar camera-to-robot coordinates and run dynamic tray sorting."""
import argparse
import json
import math
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from core import LABELS, ROLES, KINDS, atomic_json, read_json, validate_config
from planner import kin, object_plan, validate_transitions

ROOT = Path(__file__).resolve().parent
DEFAULT_SETUP = Path(__file__).resolve().parents[2] / "raccoon_dataset/tray4_setups/setup_20261108_onepour_reteach.json"
DEFAULT_CAL = Path(__file__).resolve().parents[2] / "raccoon_dataset/tray4_setups/tray4_camera_to_robot.json"


def vision_env():
    env = os.environ.copy()
    libs = list((Path.home()/"yolo_ws/.venv/lib").glob("python*/site-packages/nvidia/cudnn/lib"))
    if libs:
        env["LD_LIBRARY_PATH"] = str(libs[0]) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def worker_python(cfg):
    py = Path(cfg["vision_python"]).expanduser()
    if not py.is_file():
        raise FileNotFoundError("YOLO Python 없음: " + str(py))
    return py


def wait_detection(folder, process, roles, timeout=120, stable=True, require_all=False, after_frame=-1):
    end = time.monotonic() + timeout
    last = None
    while time.monotonic() < end:
        if (folder/"camera_error.json").exists():
            raise RuntimeError(read_json(folder/"camera_error.json")["error"])
        if process.poll() is not None:
            raise RuntimeError("카메라 작업 종료. camera_status.json / worker.log를 확인하세요.")
        path = folder/"latest_detection.json"
        if path.exists():
            row = read_json(path)
            if time.monotonic_ns()-row["host_ns"] < 1_000_000_000:
                selection = row.get("selection")
                fresh = row["frame_index"] > after_frame
                if fresh and selection and (not require_all or set(selection["order"]) == set(roles)) and (not stable or row.get("ready")):
                    return row
                if row.get("status") and row["status"] != last:
                    print("인식 상태:", row["status"], flush=True)
                    last = row["status"]
                if fresh and selection and (not require_all or set(selection["order"]) == set(roles)) and row.get("ready"):
                    return row
        time.sleep(.05)
    raise TimeoutError("안정된 물체 인식을 기다리다 제한 시간을 넘었습니다.")


def fit_homography(points):
    """Normalized DLT fit for pixel->robot XY planar homography."""
    if len(points) < 4:
        raise ValueError("보정에는 서로 다른 위치 4점 이상이 필요합니다.")
    src = np.asarray([row["pixel"] for row in points], dtype=np.float64)
    dst = np.asarray([row["robot_xy_cm"] for row in points], dtype=np.float64)

    def normalize(values):
        center = values.mean(axis=0)
        mean_dist = np.linalg.norm(values-center, axis=1).mean()
        if mean_dist < 1e-8:
            raise ValueError("보정점이 한 위치에 겹쳤습니다.")
        scale = math.sqrt(2.)/mean_dist
        transform = np.asarray([[scale, 0., -scale*center[0]],
                                [0., scale, -scale*center[1]],
                                [0., 0., 1.]])
        hom = np.column_stack((values, np.ones(len(values))))
        return (transform @ hom.T).T[:, :2], transform

    src_n, t_src = normalize(src)
    dst_n, t_dst = normalize(dst)
    a = []
    for (u, v), (x, y) in zip(src_n, dst_n):
        a.extend([
            [u, v, 1, 0, 0, 0, -x*u, -x*v],
            [0, 0, 0, u, v, 1, -y*u, -y*v],
        ])
    b = dst_n.reshape(-1)
    coeff, _, rank, _ = np.linalg.lstsq(np.asarray(a, dtype=np.float64), b, rcond=None)
    if rank < 8:
        raise ValueError("보정점이 한 직선에 몰렸습니다. 트레이 네 방향에 분산하세요.")
    h_n = np.append(coeff, 1.).reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_n @ t_src
    if abs(h[2, 2]) < 1e-10:
        raise ValueError("보정점 배치로 안정된 변환을 만들 수 없습니다.")
    h = h/h[2, 2]
    predicted = []
    for row in points:
        u, v = row["pixel"]
        p = h @ np.asarray([u, v, 1.])
        predicted.append(p[:2]/p[2])
    errors = [float(np.linalg.norm(p-np.asarray(row["robot_xy_cm"]))) for p, row in zip(predicted, points)]
    return h, errors


def map_pixel(cal, role, pixel):
    if role not in cal.get("homography_by_role", {}):
        raise ValueError(f"{LABELS.get(role, role)} 보정값이 없습니다.")
    role_cal = cal["homography_by_role"][role]
    u, v = map(float, pixel)
    xmin, xmax = role_cal["pixel_bounds"][0]
    ymin, ymax = role_cal["pixel_bounds"][1]
    margin = cal.get("extrapolation_margin_px", 20)
    if not xmin-margin <= u <= xmax+margin or not ymin-margin <= v <= ymax+margin:
        raise ValueError(f"{LABELS[role]} 검출점 {pixel}이 해당 물체의 보정 범위 밖입니다. 자동 동작을 중단합니다.")
    h = np.asarray(role_cal["homography_pixel_to_robot_xy"], dtype=np.float64)
    p = h @ np.asarray([u, v, 1.])
    if abs(p[2]) < 1e-9:
        raise ValueError("좌표 변환 분모가 0에 가깝습니다.")
    xy = [float(p[0]/p[2]), float(p[1]/p[2])]
    bounds = role_cal.get("robot_xy_bounds_cm")
    if bounds:
        pad = cal.get("robot_extrapolation_margin_cm", .75)
        if not bounds[0][0]-pad <= xy[0] <= bounds[0][1]+pad or not bounds[1][0]-pad <= xy[1] <= bounds[1][1]+pad:
            raise ValueError(f"변환 좌표 {xy}cm가 보정 로봇 XY 범위 밖입니다. 로봇을 움직이지 않고 중단합니다.")
    return xy


def start_camera(cfg, folder, remaining):
    atomic_json(folder/"config.json", cfg)
    atomic_json(folder/"camera_control.json", {"mode": "detect", "remaining": list(remaining)})
    log = (folder/"worker.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(worker_python(cfg)), str(ROOT/"tray4_dynamic_camera.py"), "--session", str(folder)],
        env=vision_env(), stdout=log, stderr=subprocess.STDOUT,
    )
    return process, log


def stop_camera(folder, process, log):
    atomic_json(folder/"camera_control.json", {"mode": "stop", "remaining": []})
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    log.close()


def calibrate(args):
    cfg = read_json(args.setup)
    validate_config(cfg)
    py = worker_python(cfg)
    session = Path(tempfile.mkdtemp(prefix="tray4_calibration_"))
    process = log = None
    points = []
    try:
        process, log = start_camera(cfg, session, list(ROLES))
        print("카메라 화면에서 트레이의 물체가 보이게 하세요. 보정은 로봇을 움직이지 않습니다.", flush=True)
        print("물체 종류별로 그 물체를 트레이의 서로 다른 위치에 놓아 최소 4점씩 기록하세요 (권장 6점 이상, 총 16점 이상).", flush=True)
        print("측정은 물체를 새 위치에 놓은 뒤, 카메라가 다시 검출한 중심점과 그 위치의 로봇 손목 XY를 짝지어 기록합니다.", flush=True)
        print("그리퍼는 움직이지 않습니다. 그리퍼를 기존 데모와 같은 파지 지점에 맞춘 뒤, 별도 터미널의 cup_pick_real.py --read 결과를 입력하세요.", flush=True)
        last_frame = -1
        while True:
            row = wait_detection(session, process, list(ROLES), timeout=120, stable=True, after_frame=last_frame)
            last_frame = row["frame_index"]
            objs = row["selection"]["objects"]
            print("검출 중심(pixel):", " | ".join(f"{i+1}:{LABELS[o['id']]} {o['center_pixel']}" for i, o in enumerate(objs)), flush=True)
            raw = input("보정할 물체 번호 (끝내기: done): ").strip().lower()
            if raw in ("done", "d", "끝"):
                break
            try:
                selected = objs[int(raw)-1]
            except (ValueError, IndexError):
                print("목록 번호를 입력하세요.")
                continue
            role = selected["id"]
            print(f"{LABELS[role]}를 트레이의 새 측정 위치에 옮겨 놓고 손을 떼세요. 안정되면 Enter를 누르세요.", flush=True)
            input("새 위치에 놓은 뒤 Enter: ")
            row = wait_detection(session, process, list(ROLES), timeout=120, stable=True, after_frame=last_frame)
            last_frame = row["frame_index"]
            obj = next((o for o in row["selection"]["objects"] if o["id"] == role), None)
            if obj is None:
                print(f"{LABELS[role]}가 새 위치에서 검출되지 않았습니다. 다시 배치하고 다음 측정으로 진행하세요.", flush=True)
                continue
            pixel = list(map(float, obj["center_pixel"]))
            print(f"새 위치 검출 중심: {pixel}px. 이 위치에서 기존 데모와 같은 파지 기준점에 그리퍼를 맞추세요.", flush=True)
            print("별도 터미널에서 cup_pick_real.py --read를 실행해 현재 손목 XY를 확인하세요.", flush=True)
            xy = input("손목 XY cm (예: 4.21, 10.45): ").strip().replace(" ", "").split(",")
            if len(xy) != 2:
                print("X,Y 두 숫자를 쉼표로 구분해 입력하세요.")
                continue
            try:
                robot_xy = [float(xy[0]), float(xy[1])]
                if not all(math.isfinite(x) for x in robot_xy):
                    raise ValueError
            except ValueError:
                print("유효한 숫자가 아닙니다.")
                continue
            points.append({"pixel": pixel, "robot_xy_cm": robot_xy, "role": role})
            counts = {role: sum(point["role"] == role for point in points) for role in ROLES}
            print("현재 보정점:", ", ".join(f"{LABELS[r]} {counts[r]}" for r in ROLES), "(최소 각 4, 권장 각 6)", flush=True)
        fitted = {}
        all_errors = []
        for role in ROLES:
            role_points = [point for point in points if point["role"] == role]
            if len(role_points) < 4:
                raise ValueError(f"{LABELS[role]} 보정점이 {len(role_points)}개입니다. 물체마다 최소 4개, 권장 6개 이상 필요합니다.")
            role_pix = np.asarray([point["pixel"] for point in role_points])
            if np.ptp(role_pix[:, 0]) < 100 or np.ptp(role_pix[:, 1]) < 70:
                raise ValueError(f"{LABELS[role]} 보정점이 좁은 영역에 몰렸습니다. 트레이 네 방향으로 물체 위치를 바꿔주세요.")
            h, errors = fit_homography(role_points)
            rms = float(np.sqrt(np.mean(np.square(errors))))
            print(f"{LABELS[role]}: {len(role_points)}점, RMS {rms:.3f}cm, 최대 {max(errors):.3f}cm")
            if rms > args.max_rms_cm or max(errors) > args.max_point_error_cm:
                raise ValueError(f"{LABELS[role]} 보정 오차가 큽니다. 중심 정렬과 위치 입력을 확인하세요.")
            all_errors.extend(errors)
            try:
                import cv2
                hull = cv2.convexHull(role_pix.astype(np.float32)).reshape(-1, 2).tolist()
            except ImportError:
                hull = []
            fitted[role] = {
                "homography_pixel_to_robot_xy": h.tolist(),
                "pixel_bounds": [[float(role_pix[:,0].min()), float(role_pix[:,0].max())],
                                 [float(role_pix[:,1].min()), float(role_pix[:,1].max())]],
                "pixel_hull": hull,
                "robot_xy_bounds_cm": [[float(min(p["robot_xy_cm"][axis] for p in role_points)),
                                        float(max(p["robot_xy_cm"][axis] for p in role_points))] for axis in range(2)],
                "point_errors_cm": errors,
                "rms_error_cm": rms,
                "max_error_cm": max(errors),
                "samples": len(role_points),
            }
        rms = float(np.sqrt(np.mean(np.square(all_errors))))
        output = {
            "schema": 1,
            "method": "planar homography: camera pixel center -> robot wrist XY (cm)",
            "camera": {"width": 640, "height": 480,
                       "serial": read_json(session/"camera_info.json").get("serial")},
            "homography_by_role": fitted,
            "robot_extrapolation_margin_cm": .75,
            "extrapolation_margin_px": 20,
            "points": points,
            "point_errors_cm": all_errors,
            "rms_error_cm": rms,
            "max_error_cm": max(all_errors),
            "setup_file": str(args.setup.resolve()),
            "pickup_z_by_role_cm": {role: cfg["pickups"][role][2] for role in ROLES},
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        atomic_json(args.out, output)
        print("보정 파일 저장:", args.out.resolve(), flush=True)
    finally:
        if process:
            stop_camera(session, process, log)
        shutil.rmtree(session, ignore_errors=True)


def run(args):
    cfg = read_json(args.setup)
    validate_config(cfg)
    cal = read_json(args.calibration)
    if cal.get("schema") != 1 or cal.get("camera", {}).get("width") != 640:
        raise ValueError("보정 JSON 형식/해상도가 맞지 않습니다. calibrate를 먼저 실행하세요.")
    if version("roboid") != kin.SDK_VERSION:
        raise ValueError("roboid==1.7.3 환경에서 실행하세요.")
    worker_python(cfg)
    session = Path(tempfile.mkdtemp(prefix="tray4_dynamic_demo_"))
    process = camera_log = robot = port = None
    failure = None
    try:
        process, camera_log = start_camera(cfg, session, list(ROLES))
        print("네 물체가 트레이 위에 보이도록 준비하세요. Enter 한 번으로 시작합니다. 이후에는 자동으로 진행합니다. Q: 취소", flush=True)
        row = wait_detection(session, process, list(ROLES), stable=True, require_all=True)
        camera_serial = read_json(session/"camera_info.json").get("serial")
        calibrated_serial = cal.get("camera", {}).get("serial")
        if calibrated_serial and camera_serial != calibrated_serial:
            raise RuntimeError(f"보정 때와 다른 카메라입니다: 보정={calibrated_serial}, 현재={camera_serial}")
        for i, obj in enumerate(row["selection"]["objects"], 1):
            xy = map_pixel(cal, obj["id"], obj["center_pixel"])
            print(f"{i}. {LABELS[obj['id']]} 중심={obj['center_pixel']}px → 로봇 XY={xy}cm", flush=True)
        if input("위 물체와 로봇 주변을 확인하고 Enter: 시작 / Q: 취소 > ").strip().lower() == "q":
            return
        from roboid import Raccoon
        from control import NativePort, Controller
        robot = Raccoon(port_name=args.port)
        deadline = time.monotonic() + 8
        while not robot.is_connected() and time.monotonic() < deadline:
            time.sleep(.05)
        if not robot.is_connected():
            raise TimeoutError("로봇 연결 실패")
        from core import JsonLog
        log = JsonLog(session/"events.jsonl")
        port = NativePort(robot, cfg["speed"], log)
        home = port.read()
        if abs(kin.pitch(home)) > 3:
            raise RuntimeError("시작 자세 pitch가 수평 0±3도가 아닙니다.")
        remaining = list(ROLES)
        completed = []
        atomic_json(session/"run.json", {"state": "running", "completed": completed})
        while remaining:
            atomic_json(session/"camera_control.json", {"mode": "detect", "remaining": remaining})
            row = wait_detection(session, process, remaining, timeout=args.object_timeout, stable=True, require_all=True)
            obj = row["selection"]["objects"][0]
            role = obj["id"]
            xy = map_pixel(cal, role, obj["center_pixel"])
            z = float(cfg["pickups"][role][2])
            # Keep the taught grasp height and object-specific sequence; adapt only planar XY.
            cfg_run = json.loads(json.dumps(cfg))
            cfg_run["pickups"][role] = [xy[0], xy[1], z]
            current = port.read()
            if max(abs(a-b) for a, b in zip(current, home)) > 2.5 or math.dist(kin.wrist_fk(current), kin.wrist_fk(home)) > .5:
                raise RuntimeError("작업 사이에 시작 자세가 바뀌었습니다. 자동으로 밀어 움직이지 않고 중단합니다.")
            stages = object_plan(cfg_run, role, home, home)
            validate_transitions(stages, home)
            print(f"\n{LABELS[role]} 실행: pixel={obj['center_pixel']} → XY={xy}, Z={z:.2f}cm", flush=True)
            controller = Controller(port, stages, log, role)
            while not controller.finished:
                q = port.read()
                controller.tick(q, time.monotonic())
                if process.poll() is not None or (session/"camera_error.json").exists():
                    raise RuntimeError("동작 중 카메라/YOLO 작업이 종료됐습니다.")
                time.sleep(1/20)
            # Motion completion alone is not proof that the grasp succeeded.
            # Require the selected item to disappear from the tray for stable frames.
            absence_deadline = time.monotonic() + args.object_timeout
            absent_stable = 0
            last_detection_frame = -1
            while time.monotonic() < absence_deadline:
                if process.poll() is not None or (session/"camera_error.json").exists():
                    raise RuntimeError("완료 확인 중 카메라/YOLO 작업이 종료됐습니다.")
                detection = read_json(session/"latest_detection.json")
                if time.monotonic_ns()-detection["host_ns"] > 1_000_000_000:
                    absent_stable = 0
                elif detection["frame_index"] != last_detection_frame:
                    last_detection_frame = detection["frame_index"]
                    current_selection = detection.get("selection") or {}
                    if detection.get("ready") and role not in current_selection.get("order", []):
                        absent_stable += 1
                    else:
                        absent_stable = 0
                if absent_stable >= cfg["stable_frames"]:
                    break
                time.sleep(.05)
            if absent_stable < cfg["stable_frames"]:
                raise TimeoutError(f"{LABELS[role]}가 트레이에서 사라진 것을 확인하지 못했습니다. 성공으로 처리하지 않고 멈춥니다.")
            # Controller's final stage is RETURN_EXACT_HOME. Allow encoders to settle before re-ranking.
            settled = False
            for _ in range(80):
                q = port.read()
                if max(abs(a-b) for a, b in zip(q, home)) <= 2.5:
                    settled = True
                    break
                time.sleep(.05)
            if not settled:
                raise RuntimeError("물체 동작 뒤 시작 자세 복귀를 확인하지 못했습니다.")
            completed.append(role)
            remaining.remove(role)
            atomic_json(session/"run.json", {"state": "running", "completed": completed, "remaining": remaining})
            print(f"{LABELS[role]} 완료. 남은 물체를 다시 인식합니다.", flush=True)
            time.sleep(1.)
        atomic_json(session/"run.json", {"state": "complete", "completed": completed, "remaining": []})
        print("네 물체 동작을 마쳤습니다.", flush=True)
    except BaseException as exc:
        failure = exc
        atomic_json(session/"run.json", {"state": "aborted", "error": repr(exc)})
        print("중단:", exc, flush=True)
        traceback.print_exc()
    finally:
        if port:
            try:
                port.stop()
            except Exception:
                pass
        if robot:
            try:
                robot.dispose()
            except Exception:
                pass
        if process:
            stop_camera(session, process, camera_log)
        if "log" in locals():
            try:
                log.close()
            except Exception:
                pass
        shutil.rmtree(session, ignore_errors=True)
    if failure:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description="Dynamic tray demo using calibrated YOLO pixel-to-robot mapping")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("calibrate", help="collect tray-plane pixel/robot XY pairs and fit mapping")
    c.add_argument("--setup", type=Path, default=DEFAULT_SETUP)
    c.add_argument("--out", type=Path, default=DEFAULT_CAL)
    c.add_argument("--max-rms-cm", type=float, default=.6)
    c.add_argument("--max-point-error-cm", type=float, default=1.2)
    r = sub.add_parser("run", help="run one-touch autonomous dynamic demo")
    r.add_argument("--setup", type=Path, default=DEFAULT_SETUP)
    r.add_argument("--calibration", type=Path, default=DEFAULT_CAL)
    r.add_argument("--port", default="/dev/ttyACM2")
    r.add_argument("--object-timeout", type=float, default=120.)
    r.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.command == "calibrate":
        calibrate(a)
    elif not a.execute:
        raise SystemExit("실제 로봇 동작에는 --execute가 필요합니다.")
    else:
        run(a)


if __name__ == "__main__":
    main()

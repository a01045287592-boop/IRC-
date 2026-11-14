"""D435i 영상과 컵 데모의 로봇 이벤트를 함께 기록한다.

사용 예:
  python raccoon_real_cup/record_cup_demo.py --execute --empty-run --step

뒤의 인자는 fixed_cup_demo.py에 그대로 전달된다.
--execute가 없으면 로봇은 움직이지 않는다.
"""

import json
import runpy
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from PIL import Image

import cup_pick_real
import roboid


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DEMO_SCRIPT = ROOT / "plate_one_demo.py"

FPS = 10.0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30


def main():
    if not DEMO_SCRIPT.is_file():
        raise SystemExit(f"컵 동작 파일이 없어: {DEMO_SCRIPT}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    episode_dir = PROJECT / "raccoon_dataset" / "plate_raw" / stamp
    frame_dir = episode_dir / "rgb_frames"
    frame_dir.mkdir(parents=True, exist_ok=False)

    events_path = episode_dir / "events.jsonl"
    metadata_path = episode_dir / "metadata.json"

    metadata = {
        "episode_id": stamp,
        "status": "recording",
        "task": "plate_one_demo",
        "controller_arguments": sys.argv[1:],
        "camera": "Intel RealSense D435i",
        "camera_color": {
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
            "camera_fps": CAMERA_FPS,
            "saved_target_fps": FPS,
            "format": "RGB",
        },
        "time_base": "time.monotonic_ns",
        "joint_unit": "degree",
        "gripper_state": "commands only; not measured opening",
        "robot_state_sampling": (
            "Recorded when existing controller calls Bench.read; "
            "not fixed frequency."
        ),
        "success": "unverified",
        "notes": (
            "Raw event recording, not yet a LeRobot/ACT dataset. "
            "Camera timestamps and host monotonic timestamps have "
            "different clock domains."
        ),
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )

    event_file = events_path.open("w", encoding="utf-8")
    event_lock = threading.Lock()
    stop_camera = threading.Event()

    camera_error = []
    frame_count = [0]

    def log_event(kind, **values):
        row = {
            "kind": kind,
            "host_monotonic_ns": time.monotonic_ns(),
            **values,
        }

        line = json.dumps(
            row,
            ensure_ascii=False,
            default=lambda value: (
                value.tolist()
                if hasattr(value, "tolist")
                else repr(value)
            ),
        )

        with event_lock:
            event_file.write(line + "\n")
            event_file.flush()

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        rs.format.rgb8,
        CAMERA_FPS,
    )

    original_read = cup_pick_real.Bench.read
    original_raccoon = roboid.Raccoon

    def recorded_read(bench, *args, **kwargs):
        before_ns = time.monotonic_ns()

        result = original_read(bench, *args, **kwargs)

        after_ns = time.monotonic_ns()

        log_event(
            "joint_observation",
            read_start_ns=before_ns,
            read_end_ns=after_ns,
            joint_deg=list(result),
        )

        return result

    class RobotRecorder:
        """既存のRaccoonを包み、送信命令だけを記録する。"""

        def __init__(self, robot):
            self._robot = robot

        def __getattr__(self, name):
            return getattr(self._robot, name)

        def _call(self, name, *args, **kwargs):
            begin_ns = time.monotonic_ns()

            try:
                result = getattr(self._robot, name)(*args, **kwargs)

            except BaseException as exc:
                log_event(
                    "robot_command_error",
                    command=name,
                    args=list(args),
                    kwargs=kwargs,
                    call_start_ns=begin_ns,
                    error=repr(exc),
                )
                raise

            log_event(
                "robot_command",
                command=name,
                args=list(args),
                kwargs=kwargs,
                call_start_ns=begin_ns,
                call_end_ns=time.monotonic_ns(),
            )

            return result

        def set_degree(self, *args, **kwargs):
            return self._call("set_degree", *args, **kwargs)

        def open_gripper(self, *args, **kwargs):
            return self._call("open_gripper", *args, **kwargs)

        def close_gripper(self, *args, **kwargs):
            return self._call("close_gripper", *args, **kwargs)

        def lock_horz(self, *args, **kwargs):
            return self._call("lock_horz", *args, **kwargs)

        def lock_vert(self, *args, **kwargs):
            return self._call("lock_vert", *args, **kwargs)

        def unlock(self, *args, **kwargs):
            return self._call("unlock", *args, **kwargs)

        def stop(self, *args, **kwargs):
            return self._call("stop", *args, **kwargs)

        def dispose(self, *args, **kwargs):
            return self._call("dispose", *args, **kwargs)

    def recorded_raccoon(*args, **kwargs):
        robot = original_raccoon(*args, **kwargs)
        log_event("robot_instance_created")
        return RobotRecorder(robot)

    def camera_worker():
        interval_ns = int(1_000_000_000 / FPS)
        next_save_ns = 0

        try:
            while not stop_camera.is_set():
                frames = pipeline.wait_for_frames(5000)
                arrival_ns = time.monotonic_ns()

                color = frames.get_color_frame()
                if not color:
                    continue

                if arrival_ns < next_save_ns:
                    continue

                next_save_ns = arrival_ns + interval_ns

                image = np.asanyarray(color.get_data()).copy()

                index = frame_count[0]
                name = f"{index:06d}.png"
                path = frame_dir / name

                Image.fromarray(image).save(path)

                log_event(
                    "camera_frame",
                    frame_index=index,
                    file=f"rgb_frames/{name}",
                    host_arrival_ns=arrival_ns,
                    camera_timestamp_ms=color.get_timestamp(),
                    camera_timestamp_domain=str(
                        color.get_frame_timestamp_domain()
                    ),
                    frame_number=color.get_frame_number(),
                )

                frame_count[0] += 1

        except BaseException as exc:
            camera_error.append(repr(exc))
            log_event("camera_error", error=repr(exc))
            print("\n카메라 기록 오류:", exc, flush=True)

    camera_started = False
    camera_thread = None
    outcome = "interrupted_or_failed"

    try:
        print("카메라 연결 중...")
        pipeline.start(config)
        camera_started = True

        # 초기 영상이 들어오는지 먼저 확인
        for _ in range(10):
            pipeline.wait_for_frames(5000)

        cup_pick_real.Bench.read = recorded_read
        roboid.Raccoon = recorded_raccoon

        camera_thread = threading.Thread(
            target=camera_worker,
            name="d435i-recorder",
            daemon=True,
        )
        camera_thread.start()

        print("\n기록 폴더:", episode_dir)
        print("영상 저장 목표:", FPS, "fps")
        print("실행할 컵 코드:", DEMO_SCRIPT)
        print("전달 인자:", sys.argv[1:])

        if "--execute" in sys.argv[1:]:
            print("주의: --execute가 있으므로 실제 로봇이 움직여.")

        log_event(
            "recording_started",
            controller_arguments=sys.argv[1:],
        )

        # 기존 컵 프로그램에 인자를 그대로 전달
        sys.argv = [str(DEMO_SCRIPT)] + sys.argv[1:]
        runpy.run_path(str(DEMO_SCRIPT), run_name="__main__")

        # 프로그램 정상 종료만 의미. 실제 작업 성공은 미검증.
        outcome = "controller_returned"

    except KeyboardInterrupt:
        print("\n기록 중단 요청.")

    finally:
        stop_camera.set()

        if camera_started:
            pipeline.stop()

        if camera_thread is not None:
            camera_thread.join(timeout=6)

        cup_pick_real.Bench.read = original_read
        roboid.Raccoon = original_raccoon

        metadata["status"] = outcome
        metadata["frame_count"] = frame_count[0]
        metadata["camera_errors"] = camera_error

        if "--empty-run" in metadata["controller_arguments"]:
            metadata["success"] = "not_applicable_empty_run"

        log_event(
            "recording_finished",
            status=outcome,
            frame_count=frame_count[0],
            camera_errors=camera_error,
        )

        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )

        with event_lock:
            event_file.close()

        print("\n기록 종료")
        print("저장 폴더:", episode_dir)
        print("저장된 RGB 프레임:", frame_count[0])
        print("로봇 이벤트:", events_path)
        print("상태:", outcome)
        print("실제 파지 성공 여부는 별도로 확인해야 해.")


if __name__ == "__main__":
    main()

"""D435i 영상과 컵 데모의 로봇 이벤트를 함께 기록한다.

사용 예:
  python raccoon_real_cup/record_cup_demo.py --execute --empty-run --step

뒤의 인자는 fixed_cup_demo.py에 그대로 전달된다.
--execute가 없으면 로봇은 움직이지 않는다.
"""

import argparse
import json
import math
import statistics
import subprocess
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
from importlib.metadata import version
from packet_tap import PacketTap


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

FPS = 10.0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30


def main():
    parser = argparse.ArgumentParser(description='컵/접시/휴지/링 동작과 D435i 및 로봇 이벤트 기록')
    parser.add_argument('--task', default='tissue', choices=['tissue'])
    parser.add_argument('--target-id', default=None, help='실제 잡을 대상의 수동 이름')
    options, forwarded = parser.parse_known_args()
    task = options.task
    target_id = options.target_id or task + '_1'
    scripts = {'cup':'fixed_cup_demo.py', 'plate':'plate_one_demo.py',
               'tissue':'object_pick_demo.py', 'ring':'object_pick_demo.py'}
    demo_script = ROOT / scripts[task]
    if '--init' in forwarded:
        parser.error('--init은 object_pick_demo.py를 직접 실행하세요.')
    if task in ('tissue','ring'):
        forwarded = ['--kind', task] + forwarded
    if not demo_script.is_file():
        raise SystemExit(f"동작 파일이 없어: {demo_script}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    episode_dir = PROJECT / "raccoon_dataset" / (task + "_raw") / stamp
    frame_dir = episode_dir / "rgb_frames"
    frame_dir.mkdir(parents=True, exist_ok=False)

    events_path = episode_dir / "events.jsonl"
    metadata_path = episode_dir / "metadata.json"

    metadata = {
        "episode_id": stamp,
        "status": "recording",
        "task": task,
        "target_id": target_id,
        "target_source": "manually supplied; YOLO selection and overlay not recorded",
        "controller_arguments": forwarded,
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
    sdk_lock = threading.RLock()
    sampling_stop = threading.Event()
    sampler_thread = [None]
    packet_tap = [None]
    sample_count = [0]
    sample_errors = []

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

        if camera_error or sample_errors:
            raise RuntimeError('카메라 또는 관절 기록이 중단되었습니다.')
        with sdk_lock:
            result = original_read(bench, *args, **kwargs)

        after_ns = time.monotonic_ns()

        log_event(
            "controller_joint_read",
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
                with sdk_lock:
                    acquired_ns = time.monotonic_ns()
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
                call_acquired_ns=acquired_ns,
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
            sampling_stop.set()
            if sampler_thread[0] is not None:
                sampler_thread[0].join(timeout=2)
            return self._call("dispose", *args, **kwargs)

    def joint_sampler(robot):
        period_ns = 50_000_000
        deadline = time.monotonic_ns()

        while not sampling_stop.is_set():
            now = time.monotonic_ns()

            if now < deadline:
                if sampling_stop.wait((deadline - now) / 1e9):
                    break

            read_start = time.monotonic_ns()

            try:
                if not sdk_lock.acquire(timeout=0.025):
                    log_event(
                        "joint_sample_skipped",
                        reason="sdk_busy",
                    )
                else:
                    try:
                        if sampling_stop.is_set():
                            break

                        if robot.is_connected():
                            q = list(robot.encoder(-1))
                            read_end = time.monotonic_ns()

                            if len(q) != 4 or not all(
                                math.isfinite(float(v)) for v in q
                            ):
                                raise ValueError(
                                    "관절값이 유효하지 않습니다"
                                )

                            log_event(
                                "joint_observation",
                                read_start_ns=read_start,
                                read_end_ns=read_end,
                                joint_deg=q,
                                source="periodic_encoder_read",
                            )
                            sample_count[0] += 1

                    finally:
                        sdk_lock.release()

            except Exception as exc:
                sample_errors.append(repr(exc))
                log_event("joint_sample_error", error=repr(exc))
                break

            deadline += period_ns
            now = time.monotonic_ns()

            if deadline <= now:
                missed = (now - deadline) // period_ns + 1
                deadline += missed * period_ns
                log_event(
                    "joint_sample_skipped",
                    reason="deadline_missed",
                    count=missed,
                )

    def recorded_raccoon(*args, **kwargs):
        if version('roboid') != '1.7.3':
            raise RuntimeError('PacketTap requires roboid 1.7.3')
        robot = original_raccoon(*args, **kwargs)
        try:
            packet_tap[0] = PacketTap(robot._roboid, log_event)
        except BaseException:
            robot.dispose()
            raise
        log_event("robot_instance_created")

        sampler_thread[0] = threading.Thread(
            target=joint_sampler,
            args=(robot,),
            name="joint-20hz",
            daemon=True,
        )
        sampler_thread[0].start()

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

                if next_save_ns == 0:
                    next_save_ns = arrival_ns

                next_save_ns += (
                    (arrival_ns - next_save_ns) // interval_ns + 1
                ) * interval_ns

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
            if not stop_camera.is_set():
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
        print("실행할 동작 코드:", demo_script)
        print("전달 인자:", forwarded)

        if "--execute" in forwarded:
            print("주의: --execute가 있으므로 실제 로봇이 움직여.")

        log_event(
            "recording_started",
            controller_arguments=forwarded,
        )

        # 기존 컵 프로그램에 인자를 그대로 전달
        sys.argv = [str(demo_script)] + forwarded
        runpy.run_path(str(demo_script), run_name="__main__")

        # 프로그램 정상 종료만 의미. 실제 작업 성공은 미검증.
        outcome = "controller_returned_unverified"

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
        sampling_stop.set()
        if sampler_thread[0] is not None:
            sampler_thread[0].join(timeout=2)

        if packet_tap[0] is not None:
            packet_tap[0].close()
            metadata['packet_samples'] = packet_tap[0].count
            metadata['packet_dropped'] = packet_tap[0].dropped
            metadata['packet_errors'] = packet_tap[0].errors
            print('패킷 관절:', packet_tap[0].count,
                  '큐 누락:', packet_tap[0].dropped,
                  '기록 오류:', packet_tap[0].errors)
        metadata["camera_errors"] = camera_error
        metadata["joint_periodic_samples"] = sample_count[0]
        metadata["joint_sample_errors"] = sample_errors
        metadata["sync_note"] = (
            "host arrival vs encoder read completion; "
            "not exposure synchronization; encoder may be cached"
        )

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

        events = [
            json.loads(line)
            for line in events_path.read_text().splitlines()
            if line.strip()
        ]

        def print_rate(label, times):
            times = sorted(times)
            gaps = [
                (b-a)/1e9
                for a, b in zip(times, times[1:])
                if b > a
            ]

            if not gaps:
                print(label, "표본 부족:", len(times))
                return

            avg = (len(times)-1) / (
                (times[-1]-times[0])/1e9
            )

            print(
                label,
                "표본:", len(times),
                "평균 Hz:", round(avg, 2),
                "중앙 간격 ms:",
                round(statistics.median(gaps)*1000, 1),
                "최대 간격 ms:",
                round(max(gaps)*1000, 1),
                "150ms 초과:",
                sum(g > 0.15 for g in gaps),
            )

        print_rate(
            "카메라",
            [
                e["host_arrival_ns"] for e in events
                if e.get("kind") == "camera_frame"
            ],
        )

        print_rate(
            "수신 패킷 관절",
            [e["read_end_ns"] for e in events if e.get("kind") == "joint_packet"],
        )
        print_rate(
            "관절",
            [
                e["read_end_ns"] for e in events
                if e.get("kind") == "joint_observation"
            ],
        )

        print(
            "관절 샘플 건너뜀 이벤트:",
            sum(
                e.get("kind") == "joint_sample_skipped"
                for e in events
            ),
        )
        converter = ROOT / "act_preview_packet.py"
        if converter.is_file() and frame_count[0]:
            sys.stdout.flush()
            result = subprocess.run(
                [
                    sys.executable,
                    str(converter),
                    str(episode_dir),
                    "--target-class", task,
                    "--target-id", target_id,
                    "--max-age-ms", "100",
                    "--future-tolerance-ms", "50",
                ],
                check=False,
            )
            if result.returncode:
                print("시간 정렬 변환 실패. 원본 기록은 보존되었습니다.")
        else:
            print("프레임이나 변환기가 없어 시간 정렬을 건너뜁니다.")


if __name__ == "__main__":
    main()

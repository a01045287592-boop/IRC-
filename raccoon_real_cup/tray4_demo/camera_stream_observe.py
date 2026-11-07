#!/usr/bin/env python3
"""Observe the D435i color stream without connecting to or commanding a robot."""
import time


def observe(rs, clock=time.monotonic, emit=print):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        serial = profile.get_device().get_info(rs.camera_info.serial_number)
        emit(f'카메라 영상 수신 시작: {serial} / 컬러 640x480, 30fps', flush=True)
        emit('이 터미널은 켜 둔 채 다른 터미널에서 연결 검사를 실행하세요. 종료는 Ctrl+C입니다.', flush=True)
        started_at = clock()
        next_report = started_at + 15.
        frames = 0
        while True:
            if pipeline.wait_for_frames(1000).get_color_frame():
                frames += 1
            now = clock()
            if now >= next_report:
                elapsed = now - started_at
                emit(f'카메라 수신 중: {elapsed:.0f}초, {frames}프레임, 평균 {frames/elapsed:.1f}fps', flush=True)
                next_report += (int((now-next_report)//15.)+1)*15.
    finally:
        if started:
            pipeline.stop()


def main():
    import pyrealsense2 as rs
    try:
        observe(rs)
    except KeyboardInterrupt:
        print('\n카메라 수신을 종료했습니다.', flush=True)


if __name__ == '__main__':
    main()

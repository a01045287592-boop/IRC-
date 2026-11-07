#!/usr/bin/env python3
"""Observe SDK connection state while the robot is stationary; no motion calls."""
import argparse
import time


def observe(robot, seconds, poll_seconds=.01, report_seconds=15., clock=time.monotonic,
            sleep=time.sleep, emit=print):
    start = clock()
    deadline = start + seconds
    next_report = start
    previous = None
    lost_since = None
    observed_losses = 0
    samples = []
    while True:
        now = clock()
        if now >= deadline:
            break
        connected = bool(robot.is_connected())
        elapsed = now-start
        if connected != previous:
            if not connected:
                observed_losses += 1
                lost_since = now
                emit(f'{elapsed:.2f}초: 연결 끊김 상태 관찰', flush=True)
            elif previous is False:
                emit(f'{elapsed:.2f}초: 연결 복귀 상태 관찰 (관찰된 끊김 약 {now-lost_since:.3f}초)', flush=True)
            else:
                emit(f'{elapsed:.2f}초: 연결 상태 관찰', flush=True)
            previous = connected
        if now >= next_report:
            signal = robot.signal_strength() if connected else None
            if signal is not None:
                samples.append(signal)
            emit(f'{elapsed:.0f}초: 연결={"정상" if connected else "끊김"}, SDK 신호값={signal}', flush=True)
            next_report += (int((now-next_report)//report_seconds)+1)*report_seconds
        sleep(min(poll_seconds, max(0., deadline-clock())))
    elapsed = clock()-start
    emit(f'검사 종료: {elapsed:.1f}초, 상태 표본에서 관찰한 끊김 {observed_losses}회', flush=True)
    if samples:
        emit(f'SDK 신호값 범위: {min(samples)} ~ {max(samples)} (캐시 값)', flush=True)
    emit('짧은 끊김은 상태 표본에서 빠질 수 있으므로 SDK의 Connection lost 출력도 함께 확인하세요.', flush=True)
    return observed_losses


def main():
    parser = argparse.ArgumentParser(description='로봇 정지 상태의 연결 관찰')
    parser.add_argument('--port', default='/dev/ttyACM0')
    parser.add_argument('--seconds', type=float, default=300.)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800:
        parser.error('관찰 시간은 1~1800초여야 합니다.')
    from roboid import Raccoon
    robot = None
    try:
        print('이 도구는 이동·집게 함수를 직접 호출하지 않습니다.', flush=True)
        print('SDK 연결과 해제 과정에서 기본 패킷이 전송될 수 있으므로 물체를 잡은 상태에서는 실행하지 마세요.', flush=True)
        robot = Raccoon(port_name=args.port)
        deadline = time.monotonic()+8.
        while not robot.is_connected():
            if time.monotonic() >= deadline:
                raise RuntimeError('로봇 연결 실패')
            time.sleep(.05)
        observe(robot, args.seconds)
    except KeyboardInterrupt:
        print('\n사용자가 관찰을 중단했습니다.', flush=True)
    finally:
        if robot is not None:
            robot.dispose()


if __name__ == '__main__':
    main()

# 프로젝트 코드 안내

이 저장소에는 RaccoonBot 카페 트레이 정리 프로젝트의 실행 코드와 YOLO 모델 가중치가 있습니다. 수집된 영상/ACT episode 데이터와 학습 체크포인트는 저장소에 포함하지 않습니다.

## 주요 파일

- `raccoon_real_cup/tray4_demo/tray4.py`: 네 물체 트레이 전체 실행 및 전체 episode 수집 진입점
- `raccoon_real_cup/tray4_demo/tray4_partial.py`: 물체별 부분 episode 수집 진입점
- `raccoon_real_cup/tray4_demo/act_train.py`: ACT 학습 및 데이터 점검
- `raccoon_real_cup/tray4_demo/act_sequence.py`: ACT 추론을 이용한 순차 실행
- `raccoon_real_cup/tray4_demo/act_trial.py`: ACT 추론 시험 실행
- `raccoon_real_cup/tray4_demo/vision.py`, `camera_worker.py`: 카메라 영상과 YOLO 검출 연동
- `raccoon_real_cup/yolo_overlay_worker.py`, `record_live_yolo.py`: YOLO 오버레이 및 YOLO와 함께하는 수집 도구
- `models/best.pt`: 이 프로젝트의 YOLO 가중치
- `examples/teaching_configs/`: 실제 배치에서 사용한 로봇 티칭 설정 예시

## 데이터 및 모델 파일

수집 데이터, 변환 데이터셋, 학습 체크포인트, 카메라/로봇 로그는 용량과 장비 의존성이 있어 Git 저장소에 포함하지 않습니다. ACT 모델을 학습하거나 실행하려면 별도로 준비한 LeRobot 데이터셋과 체크포인트가 필요합니다.

## 추가 ACT 및 시뮬레이션 코드

- 저장소 최상위의 `act_*.py`와 `evaluate_act_episode.py`: 이전 로컬 ACT 추론·제어 코드
- `cafe_demo/*.py`: MuJoCo 카페 정리 시뮬레이션 및 ACT 조직 실험 코드
- `raccoon_move_wrist_*.py`: RaccoonBot 손목 XYZ 이동 도구

대형 ACT 체크포인트와 학습 상태는 GitHub 단일 파일 크기 제한 및 용량 때문에 포함하지 않았습니다. 원본 체크포인트는 로컬 `checkpoint/`에 남아 있습니다.

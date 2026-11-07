# 라쿤봇 카페 트레이 정리: 데모 데이터 수집

RealSense D435i 영상과 RaccoonBot 관절·명령을 함께 기록하는 우분투용 수집 프로젝트입니다. 현재 대상은 플라스틱 컵 1개, 유리컵 1개, 접시 1개, 휴지 1개입니다. 시작 시 YOLO가 물체를 감지해 화면상의 2D 픽셀 거리로 순서를 정하고, 사용자가 티칭한 좌표를 따라 정리한 뒤 각 물체의 성공 여부를 확인합니다.

**상태:** 실제 사용 중 무선 연결이 끊기는 문제가 남아 있습니다. 다른 노트북에서 연결 상태를 먼저 비교하세요. 네 물체가 끝까지 성공한 실물 에피소드 및 ACT 학습은 아직 검증되지 않았습니다. 시작 전에 로봇의 동작 범위와 배치, 접시와 컵의 간섭을 확인하세요. 로봇은 연결 종료·해제 시 움직이거나 집게가 열릴 수 있으므로 주변을 비우고 관찰하세요.

## 1. 준비물

- Ubuntu x86-64, 인터넷, Git, Python 3.11을 만들 수 있는 Miniforge/Conda. 기존 테스트 환경은 Python 3.11.18입니다. 다른 OS와 다른 장비는 검증하지 않았습니다.
- RaccoonBot과 전용 동글, RealSense D435i, 4개 물체와 실제 배치에 맞는 처리 용기. 직접 연결되는 그래픽 데스크톱이 필요합니다.
- 이 저장소의 `models/best.pt`: 트레이·컵·접시·휴지용 YOLO segmentation 모델. cup은 흰색 비율로 재질을 구분하며 오인식할 수 있습니다.
- 저장 공간: 원본 RGB가 많이 쌓입니다. 생성 데이터와 가상환경은 Git에 올리지 않습니다.

아래는 `~/Raccoonbot_Cafe_Clean`에 clone하는 예입니다. 기존 폴더가 있으면 덮어쓰지 말고 안전한 다른 위치를 선택하세요. 일부 기본 저장 경로는 이 폴더를 가정하므로 다른 이름을 쓰면 실행할 때 `--out`을 명시해야 합니다.

```bash
git clone https://github.com/a01045287592-boop/IRC-.git ~/Raccoonbot_Cafe_Clean
cd ~/Raccoonbot_Cafe_Clean
```

## 2. 가상환경 설치

[Miniforge 공식 설치 안내](https://github.com/conda-forge/miniforge)를 따라 설치하고 새 터미널을 엽니다. 다음 두 환경을 만듭니다. 기존 `raccoon-real` 환경이 있다면 이름 충돌에 주의하세요. Python 패키지 설치에는 인터넷이 필요합니다.

```bash
conda create -n raccoon-real python=3.11 -y
conda activate raccoon-real
python -m pip install -r requirements-robot.txt
```

YOLO와 카메라는 별도의 venv로 실행합니다. 기존 `~/yolo_ws/.venv`가 있다면 삭제·덮어쓰기 전에 확인하세요.

```bash
mkdir -p ~/yolo_ws
python -m venv ~/yolo_ws/.venv
~/yolo_ws/.venv/bin/python -m pip install --upgrade pip
~/yolo_ws/.venv/bin/python -m pip install -r requirements-vision.txt
mkdir -p ~/yolo_ws/runs/segment/angle3/weights
cp models/best.pt ~/yolo_ws/runs/segment/angle3/weights/best.pt
```

`ultralytics` 의존성으로 PyTorch가 설치됩니다. NVIDIA GPU를 사용한다면 드라이버/CUDA에 맞는 설치 명령을 [PyTorch 공식 설치 페이지](https://pytorch.org/get-started/locally/)에서 확인하세요. CPU도 가능할 수 있지만 이 모델의 인식 및 후처리 속도는 별도 확인이 필요합니다. 팀원 노트북에서 이 설치 조합은 아직 실제 검증하지 않았습니다.

```bash
python -c "import roboid, numpy; print('robot 환경 OK')"
~/yolo_ws/.venv/bin/python -c "import cv2, pyrealsense2, ultralytics, torch; print('vision 환경 OK; CUDA:', torch.cuda.is_available())"
python raccoon_real_cup/tray4_demo/tray4.py self-test
~/yolo_ws/.venv/bin/python raccoon_real_cup/tray4_demo/tray4.py self-test
```

모델 SHA-256: `4993f93560a3009126ebe5ec9d6f867927eb8e43139f4d4966b687ca23481ca8`.

## 3. 장치 연결과 권한

동글과 카메라를 연결한 뒤 확인합니다. `/dev/ttyACM0`은 예시이므로 실제 장치명을 사용하세요.

```bash
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
id -nG
```

장치 권한이 없고 장치 그룹이 `dialout`이면 `sudo usermod -aG dialout "$USER"` 후 로그아웃하고 다시 로그인하세요. RealSense USB 권한 문제가 있으면 [RealSense 공식 Linux 설치 안내](https://github.com/realsenseai/librealsense/blob/master/doc/installation.md)의 udev 규칙 설치를 참고하세요. 관리자 권한으로 Python 전체를 실행하는 방식은 사용하지 않습니다.

**연결 검사 전 집게를 비우고 로봇 주변을 비우세요.** SDK 연결/종료 과정에서도 로봇이 움직일 수 있습니다. 기존 PC에서도 5분간 안 끊긴 결과가 있었으므로 한 번의 성공만으로 장기 안정성을 보장할 수 없습니다.

터미널 1에서 카메라를 켜 둡니다.

```bash
cd ~/Raccoonbot_Cafe_Clean
~/yolo_ws/.venv/bin/python raccoon_real_cup/tray4_demo/camera_stream_observe.py
```

터미널 2에서 실제 포트를 넣어 동시에 검사합니다.

```bash
cd ~/Raccoonbot_Cafe_Clean
conda activate raccoon-real
python raccoon_real_cup/tray4_demo/link_observe.py --port /dev/ttyACM0 --seconds 300
```

결과를 확인한 뒤 터미널 1을 Ctrl+C로 종료합니다. `Connection lost`가 나타나면 데이터 수집 전에 연결 문제를 조사하세요. 단시간 끊김은 폴링에 빠질 수 있습니다.

## 4. 배치와 좌표 설정

`examples/setup_template.json`은 마지막에 사용한 목적지·속도 설정을 참고용으로 보존하며 **물체 좌표는 비워져 있습니다.** 팀원의 실제 물체/용기 배치가 같다는 보장은 없으므로 목적지까지 반드시 확인해야 합니다. 접시 이송 경로에는 최소 17cm가 코드에 적용됩니다. 기구학 검사는 실제 충돌을 검사하지 않습니다.

```bash
cd ~/Raccoonbot_Cafe_Clean
conda activate raccoon-real
mkdir -p setups
cp examples/setup_template.json setups/my_setup.json
```

각 물체의 파지 자세에서 한 번씩 다음을 실행해 손목 좌표 x y z(cm)를 확인합니다. 컵은 정면, 접시/휴지는 집게가 아래를 향한 자세입니다. 장치를 연결·해제할 때 움직일 수 있으므로 주의하세요.

```bash
python raccoon_real_cup/cup_pick_real.py --port /dev/ttyACM0 --read
```

좌표를 읽는 과정에서 물체가 움직였다면 다시 배치해 확인합니다. 네 좌표를 입력합니다.

```bash
python raccoon_real_cup/tray4_demo/tray4.py points --setup setups/my_setup.json
```

원하는 경우 로봇에 연결하지 않는 기구학 검사를 할 수 있습니다. 예시 시작 자세의 범위 검사이며 충돌·파지 성공을 보장하지 않습니다.

```bash
python raccoon_real_cup/tray4_demo/tray4.py check --setup setups/my_setup.json
```

템플릿의 유리컵 pitch 2.99도와 목적지 좌표는 원래 장비에서 티칭한 값이므로 실제 환경에 맞는지 확인하세요. 새 물체나 용기 배치가 달라졌다면 기존 설정을 그대로 재사용하지 마세요. 실행 중 로봇을 계속 관찰하고 이상이 있으면 중단합니다.

## 5. 수집과 검사

시작 자세에서 집게를 비우고, 모든 물체가 카메라에서 보이는지 확인합니다. 터미널에서 확인 후 실행합니다. 실제 로봇이 움직입니다.

```bash
python raccoon_real_cup/tray4_demo/tray4.py run \
  --setup setups/my_setup.json --port /dev/ttyACM0 --execute \
  --out "$HOME/Raccoonbot_Cafe_Clean/raccoon_dataset/tray4_raw"
```

물체 하나를 처리하고 시작 위치로 복귀할 때마다 실제 성공이면 `y` 후 Enter, 실패면 `n` 후 Enter를 입력합니다. 네 물체 성공 후 원본 영상에 YOLO 후처리와 데이터 검사가 실행됩니다. 연결 오류 등으로 중단한 에피소드는 원본을 남기지만 성공 학습 데이터로 내보내지 않습니다.

각 에피소드는 `raccoon_dataset/tray4_raw/날짜_시간/`에 저장됩니다. `rgb_frames/`, `frames.jsonl`, `steps.jsonl`, `events.jsonl`, `annotations.jsonl`, `frame_alignment.jsonl`, `metadata.json`, `audit.json`이 주요 기록입니다. 검증 통과한 경우에만 `training/episode.npz`를 만듭니다. 긴 원본 영상 데이터는 GitHub에 자동 업로드되지 않습니다.

후처리 재실행과 검사:

```bash
python raccoon_real_cup/tray4_demo/tray4.py annotate /실제/에피소드폴더
python raccoon_real_cup/tray4_demo/tray4.py audit /실제/에피소드폴더 --export
```

## 6. 데이터의 의미와 제한

- 원본 RGB와 호스트 수신 시간, SDK 캐시에서 읽은 관절각 4개(라디안), 마지막 그리퍼 명령을 기록합니다. 실제 그리퍼 개폐 폭과 센서 측정 시각은 확인할 수 없습니다.
- action은 목표 관절각 4개, 그리퍼 명령, 세 가지 제어 모드 one-hot의 8차원입니다. 제어 모드에 따라 직접 움직이는 관절이 다릅니다.
- YOLO bbox/중심은 사후 추정 결과이며 가림과 오인식 시 누락될 수 있습니다. 현재는 초기 순서 선택에만 YOLO를 사용하고 움직이는 중에는 물체를 보고 경로를 수정하지 않습니다.
- 성공 에피소드는 장차 ACT 변환을 위한 기반입니다. 실제 로봇의 8차원 명령과 모드에 맞는 ACT 학습·정책 실행은 아직 별도 구현 및 검증이 필요합니다. [상세 기록 설명](raccoon_real_cup/tray4_demo/README_KO.md)의 예전 zip 설치 절차는 이 문서의 clone 절차로 대체합니다.

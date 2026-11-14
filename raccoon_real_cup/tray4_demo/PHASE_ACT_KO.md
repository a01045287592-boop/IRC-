# 단계 조건부 ACT

영상과 기존 5차원 관측에 현재 단계 one-hot을 연결한다. 단계 순서는 성공 데모의 plan.json에서 검증하며, 원본 NPZ를 수정하지 않는다. action chunk는 다음 단계의 action을 포함하지 않도록 padding한다.

실행기는 단계 이름만 전달한다. 관절 목표, 모드, 집게는 ACT 출력이다. 이동/회전의 예상 모드와 학습 목표 범위 밖 예측은 중단한다. 관절 도착, 안정성, pitch 및 유지 시간을 검사해 단계를 진행한다. 검증된 목표로 한 명령당 최대 5도씩 접근한다. 단계별 30초 제한을 둔다. 실기에서 집기 성공·올바른 놓기까지 보장하는 판정은 아니다.

기존 5차원 모델은 이 실행기에 사용할 수 없다. 원본 데모와 기존 가중치는 보존된다.

접시 재학습:
```bash
.venv/bin/python raccoon_real_cup/tray4_demo/act_train_objects.py --phase-conditioned --role plate_1 --steps 3000 --out raccoon_dataset/tray4_phase_plate_v1
```
새 결과 폴더가 이미 있으면 덮어쓰지 않고 중단한다.

회전 시작 예측 검사:
```bash
.venv/bin/python raccoon_real_cup/tray4_demo/act_phase_diagnose.py raccoon_dataset/tray4_phase_plate_v1/plate_1
```

접시 단독 시험 (학습·검사 완료 후):
```bash
/home/min/miniforge3/envs/raccoon-real/bin/python raccoon_real_cup/tray4_demo/act_sequence.py --policies raccoon_dataset/tray4_phase_plate_v1/policies.json --target plate_1 --port /dev/ttyACM2 --execute
```

네 정책 모두 학습:
```bash
.venv/bin/python raccoon_real_cup/tray4_demo/act_train_objects.py --phase-conditioned --steps 3000 --out raccoon_dataset/tray4_phase_act_v1
```

물체별 완료는 모든 단계 종료 후 제거·복귀·집게 해제를 확인하는 휴리스틱이다. 잘못 놓인 물체나 일시적 가림을 완전히 판별하지 못한다. Q 중단은 GUI 이벤트 처리 시점에 적용되고 블로킹 카메라/추론 호출 동안 지연될 수 있다. 실행기를 무인 안전 장치로 간주하면 안 된다.

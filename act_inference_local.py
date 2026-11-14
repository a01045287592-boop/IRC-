from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


# ==================================================
# 1. 경로
# ==================================================

BASE = Path("/home/min/Raccoonbot_Cafe_Clean")

CHECKPOINT = BASE / "checkpoint/072329/pretrained_model"

EPISODE_DIR = (
    BASE
    / "raccoon_dataset/tray4_collected/glass/first"
    / "20261109_210048_613755"
)

NPZ_PATH = EPISODE_DIR / "training/act_h10.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 테스트할 프레임
FRAME = 49


# ==================================================
# 2. 데이터 로딩
# ==================================================

print("[1] Loading NPZ...")

with np.load(NPZ_PATH, allow_pickle=False) as data:
    state = data["observation_state"][FRAME].copy()
    gt_actions = data["action"][FRAME].copy()
    pad_mask = data["action_is_pad"][FRAME].copy()
    image_path = str(data["image"][FRAME])

print("State:", state.shape)
print("GT Action Chunk:", gt_actions.shape)
print("Image reference:", image_path)


# ==================================================
# 3. RGB 이미지 경로 확인
# ==================================================

print("\n[2] Loading Image...")

image_ref = Path(image_path)

candidates = []

if image_ref.is_absolute():
    candidates.append(image_ref)
else:
    candidates.extend([
        EPISODE_DIR / image_ref,
        NPZ_PATH.parent / image_ref,
        BASE / image_ref,
        EPISODE_DIR / "images" / image_ref,
        EPISODE_DIR / "rgb" / image_ref,
    ])

existing = [p for p in candidates if p.is_file()]

if not existing:
    print("이미지 파일을 찾지 못했습니다.")
    print("저장된 이미지 경로:", image_path)
    print("확인한 후보 경로:")
    for p in candidates:
        print(" -", p)
    raise SystemExit(1)

image_file = existing[0]

print("Resolved Image:", image_file)

with Image.open(image_file) as img:
    rgb = img.convert("RGB")
    print("Image Size:", rgb.size)
    image_tensor = to_tensor(rgb)

assert image_tensor.shape == (3, 480, 640), (
    f"Expected [3,480,640], got {image_tensor.shape}"
)


# ==================================================
# 4. ACT 모델 로딩
# ==================================================

print("\n[3] Loading ACT...")

policy = ACTPolicy.from_pretrained(
    str(CHECKPOINT),
    local_files_only=True
).to(DEVICE)

policy.eval()
policy.reset()

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=CHECKPOINT,
    preprocessor_overrides={
        "device_processor": {"device": DEVICE}
    },
)

print("Device:", DEVICE)
print("Chunk Size:", policy.config.chunk_size)


# ==================================================
# 5. Observation 구성
# ==================================================

observation = {
    "observation.images.front":
        image_tensor.unsqueeze(0),
    "observation.state":
        torch.from_numpy(state).float().unsqueeze(0),
}


# ==================================================
# 6. ACT Inference
# ==================================================

print("\n[4] Running Inference...")

with torch.inference_mode():

    processed = preprocessor(observation)

    predicted = policy.predict_action_chunk(processed)

    predicted_actions = []

    for t in range(predicted.shape[1]):

        action = predicted[:, t, :]

        physical_action = postprocessor(action)

        predicted_actions.append(
            physical_action.squeeze(0).detach().cpu().numpy()
        )

predicted_actions = np.stack(predicted_actions)


# ==================================================
# 7. 결과 출력
# ==================================================

print("\n========== ACT INFERENCE RESULT ==========")

print("Predicted Shape:", predicted_actions.shape)
print("Ground Truth Shape:", gt_actions.shape)

names = [
    "goal_q1_rad",
    "goal_q2_rad",
    "goal_q3_rad",
    "goal_q4_rad",
    "gripper_command_closed",
    "mode_horizontal",
    "mode_vertical",
    "mode_wrist",
]

valid = ~pad_mask

if not valid.any():
    raise RuntimeError("모든 GT Action이 padding입니다.")

joint_error = np.abs(
    predicted_actions[valid, :4] - gt_actions[valid, :4]
)

joint_mae = joint_error.mean()

print("\nJoint MAE (rad):", joint_mae)

print("\n========== ACTION COMPARISON ==========")

for t in range(len(predicted_actions)):

    if pad_mask[t]:
        print(f"\nAction {t+1}: PADDED - 평가 제외")
        continue

    print(f"\nAction {t+1}")

    for i, name in enumerate(names):

        gt = gt_actions[t, i]
        pred = predicted_actions[t, i]

        print(
            f"{name:28s} "
            f"GT={gt:9.5f} "
            f"PRED={pred:9.5f}"
        )


# ==================================================
# 8. CSV 저장
# ==================================================

OUTPUT = BASE / "act_inference_result.csv"

with open(OUTPUT, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "step",
        "action_name",
        "ground_truth",
        "prediction",
        "absolute_error",
        "is_pad",
    ])

    for t in range(len(predicted_actions)):
        for i, name in enumerate(names):

            gt = float(gt_actions[t, i])
            pred = float(predicted_actions[t, i])

            writer.writerow([
                t,
                name,
                gt,
                pred,
                abs(gt - pred),
                bool(pad_mask[t]),
            ])

print("\nCSV saved:", OUTPUT)
print("\n========== INFERENCE SUCCESS ==========")

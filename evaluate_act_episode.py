from pathlib import Path
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


# ============================================================
# 1. 설정
# ============================================================

BASE = Path("/home/min/Raccoonbot_Cafe_Clean")

CHECKPOINT = BASE / "checkpoint/072329/pretrained_model"

EPISODE_DIR = (
    BASE
    / "raccoon_dataset/tray4_collected/glass/first"
    / "20261109_210048_613755"
)

NPZ_PATH = EPISODE_DIR / "training/act_h10.npz"

OUT = BASE / "act_evaluation" / "glass_first_072329"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chunk 안에서 관절 목표 변화가 0.1rad를 초과하면 전환 구간
TRANSITION_THRESHOLD_RAD = 0.1

ACTION_NAMES = [
    "goal_q1_rad",
    "goal_q2_rad",
    "goal_q3_rad",
    "goal_q4_rad",
    "gripper_command_closed",
    "mode_horizontal",
    "mode_vertical",
    "mode_wrist",
]


# ============================================================
# 2. 데이터셋
# ============================================================

print("\n[1] Loading Episode...")

with np.load(NPZ_PATH, allow_pickle=False) as d:
    states = d["observation_state"].copy()
    gt_actions = d["action"].copy()
    pad_masks = d["action_is_pad"].copy()
    image_paths = d["image"].copy()
    segments = d["segment_id"].copy()
    objects = d["object_id"].copy()
    frame_indices = d["frame_index"].copy()

N = len(states)

assert states.shape == (N, 5)
assert gt_actions.shape == (N, 10, 8)
assert pad_masks.shape == (N, 10)

print("Episode Frames:", N)
print("GT Action Shape:", gt_actions.shape)
print("State Shape:", states.shape)


# ============================================================
# 3. ACT 모델
# ============================================================

print("\n[2] Loading ACT...")

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
print("Checkpoint:", CHECKPOINT)


# ============================================================
# 4. 이미지 로딩
# ============================================================

def load_image(image_ref):

    ref = Path(str(image_ref))

    if ref.is_absolute():
        candidates = [ref]
    else:
        candidates = [
            EPISODE_DIR / ref,
            NPZ_PATH.parent / ref,
            BASE / ref,
        ]

    image_path = next(
        (p for p in candidates if p.is_file()),
        None
    )

    if image_path is None:
        raise FileNotFoundError(
            f"이미지를 찾지 못했습니다: {image_ref}"
        )

    with Image.open(image_path) as img:
        tensor = to_tensor(img.convert("RGB"))

    if tensor.shape != (3, 480, 640):
        raise ValueError(
            f"Unexpected image shape: {tensor.shape}"
        )

    return tensor


# ============================================================
# 5. 전체 프레임 추론
# ============================================================

print("\n[3] Running Full Episode Inference...")

predictions = np.empty_like(gt_actions)
predictions[:] = np.nan

# ACT의 각 프레임에 대해 독립적인 10-step 예측을 수행.
# 실제 로봇으로 Action을 전송하지 않음.

for i in tqdm(range(N), desc="ACT Inference"):

    image_tensor = load_image(image_paths[i])

    observation = {
        "observation.images.front":
            image_tensor.unsqueeze(0),

        "observation.state":
            torch.from_numpy(
                states[i].copy()
            ).float().unsqueeze(0),
    }

    # 프레임별 독립적인 Chunk 예측
    policy.reset()

    with torch.inference_mode():

        processed = preprocessor(observation)

        chunk = policy.predict_action_chunk(processed)

        actions = []

        for t in range(chunk.shape[1]):

            physical_action = postprocessor(
                chunk[:, t, :]
            )

            actions.append(
                physical_action.squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

    actions = np.stack(actions)

    if actions.shape != (10, 8):
        raise RuntimeError(
            f"Frame {i}: unexpected prediction {actions.shape}"
        )

    predictions[i] = actions


# ============================================================
# 6. 기본 오차
# ============================================================

print("\n[4] Calculating Metrics...")

valid = ~pad_masks

gt_valid = gt_actions[valid]
pred_valid = predictions[valid]

if not np.isfinite(pred_valid).all():
    raise RuntimeError("Prediction contains NaN or Inf")

joint_abs_error = np.abs(
    gt_valid[:, :4] - pred_valid[:, :4]
)

joint_mae = float(joint_abs_error.mean())

joint_mae_deg = float(np.rad2deg(joint_mae))

per_joint_mae = joint_abs_error.mean(axis=0)


# ============================================================
# 7. Gripper / Mode 정확도
# ============================================================

# Gripper는 0/1 명령으로 해석
gt_gripper = gt_valid[:, 4] >= 0.5
pred_gripper = pred_valid[:, 4] >= 0.5

gripper_accuracy = float(
    np.mean(gt_gripper == pred_gripper)
)

# Mode는 세 값 중 하나만 활성화되는 one-hot 기준.
# 모든 값이 0인 GT는 여기서 평가하지 않음.
gt_mode_values = gt_valid[:, 5:8]
pred_mode_values = pred_valid[:, 5:8]

mode_active = (
    np.max(gt_mode_values, axis=1) >= 0.5
)

if np.any(mode_active):

    gt_modes = np.argmax(
        gt_mode_values[mode_active],
        axis=1
    )

    pred_modes = np.argmax(
        pred_mode_values[mode_active],
        axis=1
    )

    mode_accuracy = float(
        np.mean(gt_modes == pred_modes)
    )

else:
    mode_accuracy = float("nan")


# ============================================================
# 8. 프레임별 오차 + 전환 구간
# ============================================================

frame_mae = np.full(N, np.nan)
transition = np.zeros(N, dtype=bool)

for i in range(N):

    mask = valid[i]

    if not np.any(mask):
        continue

    gt = gt_actions[i, mask, :4]
    pred = predictions[i, mask, :4]

    frame_mae[i] = np.abs(gt - pred).mean()

    # 같은 Chunk 내부의 연속 목표값 차이
    if len(gt) >= 2:

        changes = np.linalg.norm(
            np.diff(gt, axis=0),
            axis=1
        )

        transition[i] = (
            np.max(changes) > TRANSITION_THRESHOLD_RAD
        )

transition_mae = (
    float(np.nanmean(frame_mae[transition]))
    if np.any(transition)
    else float("nan")
)

stable_mask = ~transition & np.isfinite(frame_mae)

stable_mae = (
    float(np.nanmean(frame_mae[stable_mask]))
    if np.any(stable_mask)
    else float("nan")
)


# ============================================================
# 9. 결과 요약
# ============================================================

summary = {
    "episode_frames": N,
    "valid_action_targets": int(valid.sum()),
    "joint_mae_rad": joint_mae,
    "joint_mae_deg": joint_mae_deg,
    "q1_mae_rad": float(per_joint_mae[0]),
    "q2_mae_rad": float(per_joint_mae[1]),
    "q3_mae_rad": float(per_joint_mae[2]),
    "q4_mae_rad": float(per_joint_mae[3]),
    "gripper_accuracy": gripper_accuracy,
    "mode_accuracy_active_only": mode_accuracy,
    "transition_frames": int(transition.sum()),
    "transition_mae_rad": transition_mae,
    "stable_mae_rad": stable_mae,
}

print("\n" + "=" * 55)
print("FULL EPISODE EVALUATION RESULT")
print("=" * 55)

for key, value in summary.items():
    print(f"{key:30s}: {value}")

with open(OUT / "summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerows(summary.items())


# ============================================================
# 10. 프레임별 결과 CSV
# ============================================================

with open(OUT / "frame_metrics.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "sample_index",
        "frame_index",
        "segment_id",
        "object_id",
        "joint_mae_rad",
        "is_transition",
        "valid_action_targets",
    ])

    for i in range(N):

        writer.writerow([
            i,
            int(frame_indices[i]),
            int(segments[i]),
            str(objects[i]),
            float(frame_mae[i]),
            bool(transition[i]),
            int(valid[i].sum()),
        ])


# ============================================================
# 11. 전체 Action 예측 CSV
# ============================================================

with open(OUT / "action_predictions.csv", "w", newline="") as f:

    writer = csv.writer(f)

    header = [
        "sample_index",
        "chunk_step",
        "is_pad",
    ]

    header += [
        f"gt_{name}" for name in ACTION_NAMES
    ]

    header += [
        f"pred_{name}" for name in ACTION_NAMES
    ]

    writer.writerow(header)

    for i in range(N):

        for t in range(10):

            writer.writerow(
                [i, t, bool(pad_masks[i, t])]
                + gt_actions[i, t].tolist()
                + predictions[i, t].tolist()
            )


# ============================================================
# 12. 그래프 1: 전체 프레임 MAE
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    np.arange(N),
    frame_mae,
    label="Joint MAE"
)

plt.xlabel("Frame Index")
plt.ylabel("Joint MAE (rad)")
plt.title("ACT Full Episode - Joint MAE")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "01_frame_mae.png",
    dpi=160
)

plt.close()


# ============================================================
# 13. 그래프 2: GT vs Prediction
# ============================================================

# 각 프레임에서 예측한 Chunk의 첫 Action만 시각화
plt.figure(figsize=(14, 5))

plt.plot(
    np.arange(N),
    gt_actions[:, 0, 1],
    label="GT q2"
)

plt.plot(
    np.arange(N),
    predictions[:, 0, 1],
    label="ACT q2",
    linestyle="--"
)

plt.xlabel("Frame Index")
plt.ylabel("Joint Target q2 (rad)")
plt.title("ACT Prediction vs Ground Truth - q2")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "02_q2_comparison.png",
    dpi=160
)

plt.close()


# ============================================================
# 14. 그래프 3: 관절별 MAE
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    ["q1", "q2", "q3", "q4"],
    per_joint_mae
)

plt.ylabel("MAE (rad)")
plt.title("ACT - Per Joint MAE")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUT / "03_per_joint_mae.png",
    dpi=160
)

plt.close()


# ============================================================
# 15. 종료
# ============================================================

print("\n" + "=" * 55)
print("EVALUATION SUCCESS")
print("=" * 55)

print("Results:", OUT)
print("summary.csv")
print("frame_metrics.csv")
print("action_predictions.csv")
print("01_frame_mae.png")
print("02_q2_comparison.png")
print("03_per_joint_mae.png")

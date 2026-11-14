import numpy as np
import torch
import cv2

from pathlib import Path
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ROOT = Path("/home/min/Raccoonbot_Cafe_Clean")

CHECKPOINT = ROOT / "checkpoint/072328/pretrained_model"

EPISODE = (
    ROOT
    / "raccoon_dataset/tray4_collected/glass/first"
    / "20261109_210048_613755"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading ACT...")

policy = ACTPolicy.from_pretrained(
    CHECKPOINT,
    local_files_only=True
).to(device)

policy.eval()

pre, post = make_pre_post_processors(
    policy.config,
    pretrained_path=CHECKPOINT,
    preprocessor_overrides={
        "device_processor": {"device": device}
    }
)

with np.load(
    EPISODE / "training/act_h10.npz",
    allow_pickle=False
) as data:

    states = data["observation_state"].copy()
    image_paths = data["image"].copy()

def load_image(frame):
    path = EPISODE / str(image_paths[frame])

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Image not found: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return torch.from_numpy(
        rgb.copy()
    ).permute(2, 0, 1).float().unsqueeze(0) / 255.0

@torch.inference_mode()
def predict(image_frame, state_frame):

    obs = {
        "observation.images.front": load_image(image_frame),
        "observation.state": torch.tensor(
            states[state_frame],
            dtype=torch.float32
        ).unsqueeze(0)
    }

    chunk = policy.predict_action_chunk(pre(obs))

    action = post(chunk[:, 0, :]).squeeze(0).cpu().numpy()

    q = np.rad2deg(action[:4])
    mode = ["H", "V", "W"][np.argmax(action[5:8])]

    return q, mode

tests = [
    ("A", 35, 35),
    ("B", 40, 40),
    ("C", 35, 40),
    ("D", 40, 35),
]

print("\n======= ACT TRANSITION TEST =======")

for name, image_frame, state_frame in tests:

    q, mode = predict(image_frame, state_frame)

    print(
        f"{name} | "
        f"Image={image_frame} | "
        f"State={state_frame} | "
        f"Q={np.round(q, 2)} | "
        f"Mode={mode}"
    )

print("===================================")

#!/usr/bin/env python3
"""Convert audited successful Tray4 ACT episodes to a local LeRobot dataset.

Source episodes are read only. RGB is stored as lossless PNG (not MP4) so the
post-conversion audit can compare every decoded image pixel-for-pixel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from review import audit  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collect_sources(root: Path, groups: list[str] | None = None):
    included, excluded = [], []
    if groups:
        candidates = sorted(
            npz_path
            for group in groups
            for npz_path in (root / group).rglob("training/episode.npz")
        )
    else:
        candidates = sorted(root.rglob("training/episode.npz"))
    for npz_path in candidates:
        folder = npz_path.parents[1]
        metadata_path = folder / "metadata.json"
        schema_path = folder / "training" / "schema.json"
        if not metadata_path.is_file() or not schema_path.is_file():
            excluded.append({"episode": str(folder), "reason": "missing metadata/schema"})
            continue
        metadata = json.loads(metadata_path.read_text())
        schema = json.loads(schema_path.read_text())
        is_partial = schema.get("format") == "raccoon_partial_act_v1"
        report = audit(folder, partial=is_partial)
        reasons = []
        if metadata.get("status") != "complete":
            reasons.append("status is not complete")
        if metadata.get("success") != "user_confirmed":
            reasons.append("success was not user-confirmed")
        if not report["data_checks_passed"]:
            reasons.extend(report["errors"])
        if reasons:
            excluded.append({"episode": str(folder), "reason": reasons})
            continue
        included.append({"folder": folder, "metadata": metadata, "schema": schema,
                         "partial": is_partial, "audit": report})
    return included, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "raccoon_dataset/tray4_collected/full_episode")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "raccoon_dataset/tray4_lerobot/full_success_v1")
    parser.add_argument("--groups", nargs="+", choices=["full_episode", "plate", "tissue", "glass", "plastic"])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    source_root = args.source.resolve()
    out = args.out.resolve()
    episodes, excluded = collect_sources(source_root, args.groups)
    if not episodes:
        raise SystemExit("검증을 통과한 성공 에피소드가 없습니다.")

    # Preflight every source row and image before creating any output.
    total = 0
    image_shape = None
    source_hashes = {}
    for ep in episodes:
        npz_path = ep["folder"] / "training/episode.npz"
        data = np.load(npz_path, allow_pickle=False)
        n = len(data["state"])
        if data["state"].shape != (n, 5) or data["action"].shape != (n, 8):
            raise ValueError(f"{ep['folder'].name}: state/action shape mismatch")
        if not np.isfinite(data["state"]).all() or not np.isfinite(data["action"]).all():
            raise ValueError(f"{ep['folder'].name}: non-finite state/action")
        if any(data[k].shape != (n,) for k in ("image", "segment_id", "object_id")):
            raise ValueError(f"{ep['folder'].name}: per-frame columns mismatch")
        for frame_idx, relative in enumerate(data["image"]):
            image_path = ep["folder"] / str(relative)
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if frame_idx == 0:
                with Image.open(image_path) as im:
                    im = im.convert("RGB")
                    shape = (im.height, im.width, 3)
                if image_shape is None:
                    image_shape = shape
                elif shape != image_shape:
                    raise ValueError(f"mixed image resolutions: {shape} != {image_shape}")
        total += n
        source_hashes[str(ep["folder"].relative_to(source_root))] = sha256(npz_path)

    if args.verify_only:
        if not out.exists():
            raise SystemExit(f"변환본이 없습니다: {out}")
        verify(out, episodes, total, source_hashes, excluded)
        return
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"출력 경로가 이미 비어 있지 않습니다. 덮어쓰지 않습니다: {out}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    object_names = sorted({str(x) for ep in episodes
                           for x in np.load(ep["folder"] / "training/episode.npz", allow_pickle=False)["object_id"]})
    object_to_code = {name: i for i, name in enumerate(object_names)}
    h, w, c = image_shape
    features = {
        "observation.images.front": {"dtype": "image", "shape": [h, w, c],
                                      "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": [5],
                              "names": ["q1_rad", "q2_rad", "q3_rad", "q4_rad", "previous_gripper_command_closed"]},
        "action": {"dtype": "float32", "shape": [8],
                   "names": ["goal_q1_rad", "goal_q2_rad", "goal_q3_rad", "goal_q4_rad",
                             "gripper_command_closed", "mode_horizontal", "mode_vertical", "mode_wrist"]},
        "segment_id": {"dtype": "int32", "shape": [1], "names": ["segment_id"]},
        "object_id_code": {"dtype": "int32", "shape": [1], "names": ["object_id_code"]},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset.create(
        repo_id="local/tray4-success-v1", root=out, fps=10,
        robot_type="raccoonbot_4dof", features=features, use_videos=False,
    )
    episode_map = []
    try:
        for ep_index, ep in enumerate(episodes):
            folder = ep["folder"]
            data = np.load(folder / "training/episode.npz", allow_pickle=False)
            task = ep["metadata"].get("instruction", "Tray4 object sorting")
            for i in range(len(data["state"])):
                image_path = folder / str(data["image"][i])
                with Image.open(image_path) as im:
                    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
                object_code = object_to_code[str(data["object_id"][i])]
                ds.add_frame({
                    "observation.images.front": rgb,
                    "observation.state": data["state"][i].astype(np.float32, copy=False),
                    "action": data["action"][i].astype(np.float32, copy=False),
                    "segment_id": np.asarray([data["segment_id"][i]], dtype=np.int32),
                    "object_id_code": np.asarray([object_code], dtype=np.int32),
                    "task": task,
                })
            ds.save_episode()
            episode_map.append({
                "lerobot_episode_index": ep_index,
                "source_episode": str(folder.relative_to(source_root)),
                "source_absolute_path": str(folder),
                "samples": int(len(data["state"])),
                "kind": "partial_object" if ep["partial"] else "full_four_object",
                "success": "user_confirmed",
                "source_training_npz_sha256": source_hashes[str(folder.relative_to(source_root))],
                "instruction": task,
            })
            print(f"변환 {ep_index + 1}/{len(episodes)}: {folder.name} ({len(data['state'])} frames)", flush=True)
        ds.finalize()
    except BaseException:
        try:
            ds.finalize()
        finally:
            raise

    manifest = {
        "conversion": "Raccoon Tray4 native-goal ACT samples to LeRobotDataset",
        "lerobot_version": "0.6.1",
        "source_root": str(source_root),
        "output_root": str(out),
        "fps": 10,
        "included_episode_count": len(episodes),
        "included_sample_count": total,
        "included_full_four_object_count": sum(not e["partial"] for e in episodes),
        "included_partial_object_count": sum(e["partial"] for e in episodes),
        "excluded": excluded,
        "features": features,
        "object_id_code_map": object_to_code,
        "image_storage": "lossless PNG, use_videos=False",
        "episode_map": episode_map,
        "source_npz_sha256": source_hashes,
    }
    (out / "conversion_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    verify(out, episodes, total, source_hashes, excluded)


def verify(out: Path, episodes, expected_total, source_hashes, excluded):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local/tray4-success-v1", root=out, download_videos=False)
    if len(ds) != expected_total:
        raise AssertionError(f"sample count mismatch: {len(ds)} != {expected_total}")
    manifest = json.loads((out / "conversion_manifest.json").read_text())
    if ds.meta.total_episodes != len(episodes):
        raise AssertionError(f"episode count mismatch: {ds.meta.total_episodes} != {len(episodes)}")

    # Compare every numeric row and every losslessly stored decoded RGB image.
    global_i = 0
    exact_states = exact_actions = exact_images = 0
    for ep_index, (ep, mapping) in enumerate(zip(episodes, manifest["episode_map"], strict=True)):
        src = np.load(ep["folder"] / "training/episode.npz", allow_pickle=False)
        if mapping["source_training_npz_sha256"] != source_hashes[str(ep["folder"].relative_to(Path(manifest["source_root"])))]:
            raise AssertionError(f"source hash changed: {ep['folder']}")
        for i in range(len(src["state"])):
            row = ds[global_i]
            state = row["observation.state"].detach().cpu().numpy()
            action = row["action"].detach().cpu().numpy()
            if not np.array_equal(state, src["state"][i]):
                raise AssertionError(f"state mismatch at episode {ep_index}, row {i}")
            if not np.array_equal(action, src["action"][i]):
                raise AssertionError(f"action mismatch at episode {ep_index}, row {i}")
            exact_states += 1
            exact_actions += 1
            src_img = np.asarray(Image.open(ep["folder"] / str(src["image"][i])).convert("RGB"))
            # LeRobot reader presents image tensors in C,H,W and float [0,1].
            dst_img = row["observation.images.front"].detach().cpu().numpy()
            if dst_img.shape[0] == 3:
                dst_img = dst_img.transpose(1, 2, 0)
            dst_img = np.rint(dst_img * 255).clip(0, 255).astype(np.uint8)
            if not np.array_equal(src_img, dst_img):
                raise AssertionError(f"image mismatch at episode {ep_index}, row {i}")
            exact_images += 1
            global_i += 1
        print(f"검증 {ep_index + 1}/{len(episodes)}: {ep['folder'].name}", flush=True)

    if global_i != expected_total:
        raise AssertionError(f"iterated sample count mismatch: {global_i} != {expected_total}")
    report = {
        "result": "PASS",
        "source_episode_count": len(episodes),
        "converted_episode_count": ds.meta.total_episodes,
        "sample_count": len(ds),
        "state_rows_exact": exact_states,
        "action_rows_exact": exact_actions,
        "decoded_rgb_images_pixel_exact": exact_images,
        "excluded_episode_count": len(excluded),
        "excluded": excluded,
        "source_npz_sha256_unchanged": True,
    }
    (out / "conversion_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

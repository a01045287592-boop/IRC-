#!/usr/bin/env python3
"""Live YOLO worker for calibrated, changing tray object positions."""
import argparse
import time
import traceback
from pathlib import Path

from core import atomic_json, read_json
from vision import detect, load_model, remaining_selection


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=Path, required=True)
    args = p.parse_args()
    folder = args.session
    cfg = read_json(folder / "config.json")
    pipe = None
    started = False
    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs

        atomic_json(folder / "camera_status.json", {"state": "loading_model", "host_ns": time.monotonic_ns()})
        model = load_model(cfg)
        pipe = rs.pipeline()
        rc = rs.config()
        if cfg.get("camera_serial"):
            rc.enable_device(cfg["camera_serial"])
        rc.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipe.start(rc)
        started = True
        atomic_json(folder / "camera_info.json", {
            "serial": profile.get_device().get_info(rs.camera_info.serial_number),
            "width": 640, "height": 480, "fps": 30,
        })
        previous = None
        stable = 0
        frame_index = 0
        last_number = None
        while True:
            control = read_json(folder / "camera_control.json")
            if control.get("mode") == "stop":
                break
            frame = pipe.wait_for_frames(1000).get_color_frame()
            if not frame:
                continue
            number = frame.get_frame_number()
            if number == last_number:
                continue
            last_number = number
            image = np.asanyarray(frame.get_data()).copy()
            dets = detect(model, image, cfg)
            roles = control.get("remaining", [])
            selection = None
            status = ""
            try:
                selection = remaining_selection(dets, roles)
            except ValueError as exc:
                status = str(exc)
            if selection is not None and previous is not None and selection["order"] == previous["order"]:
                old = {o["id"]: o for o in previous["objects"]}
                stable = stable + 1 if all(
                    o["id"] in old and ((o["center_pixel"][0]-old[o["id"]]["center_pixel"][0])**2+
                                         (o["center_pixel"][1]-old[o["id"]]["center_pixel"][1])**2)**.5 <= cfg["stable_pixels"]
                    for o in selection["objects"]
                ) else 1
            else:
                stable = 1 if selection else 0
            previous = selection
            atomic_json(folder / "latest_detection.json", {
                "frame_index": frame_index,
                "host_ns": time.monotonic_ns(),
                "ready": bool(selection) and stable >= cfg["stable_frames"],
                "stable_frames": stable,
                "status": status,
                "selection": selection,
            })
            # Keep the live view useful during calibration and the demo.
            rank = {o["id"]: i+1 for i, o in enumerate((selection or {}).get("objects", []))}
            view = image.copy()
            for d in dets:
                if d["kind"] == "tray":
                    continue
                role = next((r for r, k in {
                    "plate_1": "plate", "tissue_1": "tissue",
                    "glass_1": "glass", "plastic_1": "plastic",
                }.items() if k == d["kind"]), None)
                if role not in rank:
                    continue
                x, y = map(int, d["center"])
                x1, y1, x2, y2 = map(int, d["box"])
                color = (0, 255, 0) if rank[role] == 1 else (0, 190, 255)
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 2 if rank[role] == 1 else 1)
                cv2.circle(view, (x, y), 4, color, -1)
                cv2.putText(view, f"{rank[role]} {role}", (x1, max(20, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
            cv2.putText(view, f"REMAINING: {', '.join(roles)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,0), 2)
            cv2.imshow("TRAY4 calibrated dynamic demo - Q to stop", view)
            key = cv2.waitKey(1) & 255
            if key in (ord("q"), ord("Q"), 27):
                raise KeyboardInterrupt("카메라 창에서 취소")
            frame_index += 1
        atomic_json(folder / "camera_status.json", {"state": "stopped", "host_ns": time.monotonic_ns()})
    except BaseException as exc:
        atomic_json(folder / "camera_error.json", {
            "error": repr(exc), "traceback": traceback.format_exc(), "host_ns": time.monotonic_ns(),
        })
        raise
    finally:
        if started:
            pipe.stop()
        try:
            import cv2
            cv2.destroyAllWindows()
        except ImportError:
            pass


if __name__ == "__main__":
    main()

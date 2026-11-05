"""Capture the initial scene from the agreed opposite-side camera."""
from pathlib import Path
from types import SimpleNamespace
import mujoco
from PIL import Image
from nesting_sorting import full_scene, FullNesting

def main():
    repo = Path(__file__).resolve().parents[1]
    out = repo / "cafe_demo" / "vlm_input"
    out.mkdir(parents=True, exist_ok=True)
    scene, manifest = full_scene(repo, out)
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 640
    demo = FullNesting(model, SimpleNamespace(fps=30, fast=True, out=out), manifest)
    demo.hold(1.0)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, .145, .015]
    camera.distance = .40
    camera.azimuth = 270
    camera.elevation = -45
    with mujoco.Renderer(model, height=640, width=960) as renderer:
        renderer.update_scene(demo.data, camera=camera)
        Image.fromarray(renderer.render()).save(out / "scene.png")
    print("Saved:", out / "scene.png")

if __name__ == "__main__":
    main()

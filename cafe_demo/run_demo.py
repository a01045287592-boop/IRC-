"""RaccoonBot action-only MuJoCo demo. No camera, model server, or hardware I/O."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

UPSTREAM_COMMIT = "42c7339d8de2a55cb751dcce9b7a5d3cc644cb4c"
COLORS = [("red", ".90 .16 .16 1"), ("blue", ".15 .45 .94 1"),
          ("yellow", ".98 .70 .12 1"), ("green", ".12 .72 .46 1")]
CUP_RADIUS = .012
CUP_HEIGHT = .030
STACK_XY = np.array([.170, .080])
BIN_XY = np.array([-.170, .080])
TCP_LOCAL = [-.0056406, .0792, .0002]
HOME = np.deg2rad([0, -10, -140, 60])
OPEN, CLOSED = .15701, -.85


def vec(values):
    return " ".join(f"{float(x):.9g}" for x in values)


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def build_scene(repo: Path, destination: Path):
    """Read original robot locally; never modify its XML/mesh files."""
    source = repo / "Mujoco" / "RaccoonBot_S.xml"
    if not source.exists():
        raise FileNotFoundError(f"Robot XML not found: {source}. See README_KO.md.")
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "raccoon_cafe_action_demo")
    root.find("compiler").set("meshdir", str((source.parent / "assets").resolve()))
    option = root.find("option")
    option.set("timestep", "0.0005")
    option.set("noslip_iterations", "5")
    option.set("iterations", "80")
    # Keep upstream joint limits, masses and actuator force limits. Reduce its
    # large viscous gains for this scripted trajectory, with feedforward below.
    for actuator, kv in zip(root.find("actuator"), [.15, .35, .20, .06]):
        actuator.set("kv", str(kv))
    # Model already has explicit body inertias; new objects also get explicit ones.
    root.find(".//body[@name='Link4']").append(ET.Element(
        "site", name="task_tcp", pos=vec(TCP_LOCAL), size=".002", rgba="0 0 0 0"))
    asset = root.find("asset")
    add(asset, "texture", name="table_texture", type="2d", builtin="checker",
        rgb1=".16 .22 .29", rgb2=".18 .25 .32", width="256", height="256")
    add(asset, "material", name="table_material", texture="table_texture", texrepeat="8 8",
        reflectance=".05", shininess=".2")
    visual = add(root, "visual")
    add(visual, "global", offwidth="1280", offheight="960")
    add(visual, "headlight", ambient=".35 .35 .35", diffuse=".7 .7 .7", specular=".1 .1 .1")
    add(visual, "quality", shadowsize="512", offsamples="0")
    add(root, "statistic", center="0 .10 .10", extent=".48")
    world = root.find("worldbody")
    add(world, "light", pos=".1 .1 .9", dir="0 0 -1", diffuse=".9 .9 .9", directional="true")
    add(world, "geom", name="table", type="plane", size=".48 .42 .02", material="table_material",
        friction=".8 .01 .001", solref=".003 1")
    # Perspective looking toward the robot from the front right.
    camera_pos = np.array([.40, .57, .37])
    lookat = np.array([0, .10, .095])
    backward = camera_pos - lookat
    backward /= np.linalg.norm(backward)
    right = np.cross([0, 0, 1], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    add(world, "camera", name="overview", pos=vec(camera_pos), xyaxes=vec([*right, *up]), fovy="42")
    add(world, "camera", name="front", pos="0 .57 .30", xyaxes="-1 0 0 0 -.38 .925", fovy="43")
    # Visual regions are non-colliding and just above the table.
    add(world, "geom", name="source_zone", type="box", pos="0 .185 .0002", size=".125 .045 .0002",
        rgba=".25 .33 .42 1", contype="0", conaffinity="0")
    stack = add(world, "body", name="stack_base", pos=vec([*STACK_XY, .003]))
    add(stack, "geom", name="stack_platform", type="cylinder", size=".030 .003", rgba=".45 .70 .78 1",
        friction="1 .01 .001", condim="4", solref=".003 1")
    # Open hexagonal receptacle: six solid walls and a bottom; no fluid solver.
    bin_body = add(world, "body", name="hex_bin", pos=vec([*BIN_XY, 0]))
    add(bin_body, "geom", name="bin_bottom", type="cylinder", size=".052 .003", pos="0 0 .003",
        rgba=".18 .55 .64 1")
    for i in range(6):
        angle = i * math.pi / 3
        radius = .049
        add(bin_body, "geom", name=f"bin_wall_{i}", type="box",
            pos=vec([radius*math.cos(angle), radius*math.sin(angle), .025]),
            euler=vec([0, 0, angle]), size=".003 .030 .025", rgba=".22 .64 .70 1")
    for index, (name, color) in enumerate(COLORS):
        pos = [(-.09 + .06*index), .185, CUP_HEIGHT/2 + .0005]
        body = add(world, "body", name=f"cup_{name}", pos=vec(pos))
        add(body, "freejoint", name=f"free_{name}")
        mass = .008
        ixy = mass*(3*CUP_RADIUS**2+CUP_HEIGHT**2)/12
        iz = mass*CUP_RADIUS**2/2
        add(body, "inertial", pos="0 0 0", mass=str(mass), diaginertia=vec([ixy, ixy, iz]))
        add(body, "geom", name=f"cup_geom_{name}", type="cylinder", size=vec([CUP_RADIUS, CUP_HEIGHT/2]),
            rgba=color, friction="1.2 .015 .001", condim="4", solref=".003 1", solimp=".95 .99 .001")
        # Dark top makes tilting legible; collision geometry remains a solid cylinder.
        add(body, "geom", name=f"top_{name}", type="cylinder", pos=vec([0, 0, CUP_HEIGHT/2+.0002]),
            size=vec([CUP_RADIUS*.78, .0002]), rgba=".12 .16 .19 1", contype="0", conaffinity="0")
        add(body, "site", name=f"lip_{name}", pos=vec([0, 0, CUP_HEIGHT/2]), size=".001", rgba="0 0 0 0")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


class Demo:
    def __init__(self, model, args):
        self.model, self.args = model, args
        self.data, self.ikdata = mujoco.MjData(model), mujoco.MjData(model)
        self.arm_joints = [model.joint(f"Joint{i}").id for i in range(1, 5)]
        self.arm_q = np.array([model.jnt_qposadr[j] for j in self.arm_joints])
        self.arm_dof = np.array([model.jnt_dofadr[j] for j in self.arm_joints])
        self.arm_ctrl = np.array([model.actuator(f"Joint{i}").id for i in range(1, 5)])
        self.grip_ctrl = model.actuator("Gripper").id
        self.tcp_id = model.site("task_tcp").id
        self.lower = model.jnt_range[self.arm_joints, 0] + 1e-4
        self.upper = model.jnt_range[self.arm_joints, 1] - 1e-4
        self.setpoint = HOME.copy()
        self.grip = OPEN
        self.data.qpos[self.arm_q] = HOME
        self.data.qpos[model.joint("Gripper_L_Inner").qposadr[0]] = OPEN
        self.data.ctrl[self.arm_ctrl] = HOME
        self.data.ctrl[self.grip_ctrl] = OPEN
        mujoco.mj_forward(model, self.data)
        self.phase = "INITIALIZE"
        self.index = 0
        self.events, self.rows = [], []
        self.viewer = None
        self.renderer = None
        self.writer = None
        self.next_frame = 0.0
        self.frame_dt = 1 / args.fps
        self.wall_start = time.monotonic()
        self.held = None
        self.result = {"success": False, "grasp_mode": "physical", "liquid_simulated": False,
                       "upstream_commit": UPSTREAM_COMMIT, "cups": [], "robot_hardware_tested": False}

    def tcp(self):
        return self.data.site_xpos[self.tcp_id].copy()

    def ik(self, target, pitch=0.0):
        target = np.asarray(target, float)
        self.ikdata.qpos[:] = self.data.qpos
        def residual(q):
            self.ikdata.qpos[self.arm_q] = q
            mujoco.mj_kinematics(self.model, self.ikdata)
            rotation = self.ikdata.body("Link4").xmat.reshape(3, 3)
            forward = rotation[:, 1]
            actual_pitch = math.atan2(forward[2], np.linalg.norm(forward[:2]))
            return np.r_[self.ikdata.site_xpos[self.tcp_id] - target, .08*(actual_pitch-pitch)]
        seed = np.clip(self.setpoint, self.lower, self.upper)
        best = least_squares(residual, seed, bounds=(self.lower, self.upper),
                             xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=100)
        err = residual(best.x)
        if np.linalg.norm(err[:3]) > .0005 or abs(err[3]) > .0005:
            raise RuntimeError(f"Unreachable pose: {target.tolist()}, pitch={math.degrees(pitch):.1f}; residual {err}")
        return best.x

    def event(self, phase, **details):
        self.phase = phase
        item = {"time": round(self.data.time, 4), "cup": self.index+1, "phase": phase, **details}
        self.events.append(item)
        print(json.dumps(item), flush=True)

    def step(self, q, qvel=None):
        kp = self.model.actuator_gainprm[self.arm_ctrl, 0]
        kv = -self.model.actuator_biasprm[self.arm_ctrl, 2]
        ff = self.data.qfrc_bias[self.arm_dof] + kv*(qvel if qvel is not None else 0)
        self.data.ctrl[self.arm_ctrl] = np.clip(q + ff/kp,
            self.model.actuator_ctrlrange[self.arm_ctrl, 0], self.model.actuator_ctrlrange[self.arm_ctrl, 1])
        self.data.ctrl[self.grip_ctrl] = self.grip
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Nonfinite simulation state")
        if self.data.time + 1e-8 >= self.next_frame:
            self.next_frame += self.frame_dt
            self.output_frame()
            if self.viewer:
                if not self.viewer.is_running():
                    raise KeyboardInterrupt
                self.viewer.sync()
                if not self.args.fast:
                    delay = self.wall_start + self.data.time - time.monotonic()
                    if delay > 0:
                        time.sleep(min(delay, self.frame_dt))

    def hold(self, duration):
        for _ in range(round(duration/self.model.opt.timestep)):
            self.step(self.setpoint)

    def move_joints(self, target, duration):
        target = np.asarray(target)
        if np.any(target < self.lower) or np.any(target > self.upper):
            raise RuntimeError(f"Joint limit: {np.rad2deg(target).tolist()}")
        start = self.setpoint.copy()
        n = max(1, round(duration/self.model.opt.timestep))
        for i in range(1, n+1):
            s = i/n
            blend = 10*s**3 - 15*s**4 + 6*s**5
            rate = (30*s**2 - 60*s**3 + 30*s**4)/duration
            self.step(start + blend*(target-start), rate*(target-start))
        self.setpoint = target.copy()
        self.hold(.15)

    def move(self, target, duration=1.8, arc=False):
        # Interpolated Cartesian samples preserve level orientation while moving.
        start = self.tcp()
        target = np.asarray(target, float)
        radius0, radius1 = np.linalg.norm(start[:2]), np.linalg.norm(target[:2])
        angle0, angle1 = math.atan2(start[1], start[0]), math.atan2(target[1], target[0])
        angle_delta = (angle1-angle0+math.pi) % (2*math.pi)-math.pi
        n = max(1, int(duration*25))
        qs = [self.setpoint.copy()]
        for i in range(1, n+1):
            s = i/n
            blend = 10*s**3 - 15*s**4 + 6*s**5
            waypoint = start + blend*(target-start)
            if arc:
                radius, angle = radius0+blend*(radius1-radius0), angle0+blend*angle_delta
                waypoint[:2] = radius*np.array([math.cos(angle), math.sin(angle)])
            qs.append(self.ik(waypoint))
        for i in range(n):
            a, b = qs[i], qs[i+1]
            steps = max(1, round(duration/n/self.model.opt.timestep))
            for k in range(1, steps+1):
                self.step(a+(b-a)*k/steps, (b-a)/(steps*self.model.opt.timestep))
        self.setpoint = qs[-1]
        self.hold(.2)
        error = np.linalg.norm(self.tcp()-target)
        if error > .004:
            raise RuntimeError(f"TCP tracking error {error*1000:.2f} mm at {target}")

    def close_grip(self):
        start = self.grip
        n = round(.8/self.model.opt.timestep)
        for i in range(1, n+1):
            self.grip = start + (CLOSED-start)*i/n
            self.step(self.setpoint)
        self.hold(.3)

    def finger_contacts(self, name):
        target = self.model.body(f"cup_{name}").id
        fingers = {self.model.body(f"Gripper_{side}_Finger").id: side for side in ("L", "R")}
        touching = set()
        for c in self.data.contact[:self.data.ncon]:
            a, b = self.model.geom_bodyid[[c.geom1, c.geom2]]
            other = b if a == target else a if b == target else -1
            if other in fingers and c.dist < .0002:
                touching.add(fingers[other])
        return touching

    def verify_grasp(self, name):
        if self.finger_contacts(name) != {"L", "R"}:
            raise RuntimeError(f"Bilateral contact missing for {name}: {self.finger_contacts(name)}")
        # Logical state only: no weld, pose override, or attachment force.
        self.held = name

    def release(self, name):
        self.held = None
        self.grip = OPEN
        self.hold(.5)

    def object_pose(self, name):
        b = self.data.body(f"cup_{name}")
        tilt = math.degrees(math.acos(np.clip(b.xmat.reshape(3, 3)[2, 2], -1, 1)))
        return b.xpos.copy(), tilt

    def output_frame(self):
        self.rows.append({"time": float(self.data.time), "phase": self.phase,
                          "cup": self.index+1, "q_rad": self.data.qpos[self.arm_q].tolist(),
                          "tcp_m": self.tcp().tolist(),
                          "objects": {c: self.object_pose(c)[0].tolist() for c, _ in COLORS}})
        if self.renderer is None:
            return
        self.renderer.update_scene(self.data, camera="overview")
        # Fast and deterministic software/EGL rendering; physics is unchanged.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        image = self.renderer.render()
        from PIL import Image, ImageDraw, ImageFont
        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, 15) if Path(font_path).exists() else ImageFont.load_default()
        draw.rectangle((0, 0, image.shape[1], 68), fill=(18, 28, 41))
        draw.text((18, 10), "RACCOONBOT  |  CAFE RETURN STATION", fill=(245, 248, 250), font=font)
        draw.text((18, 38), f"Cup {min(self.index+1, 4)}/4  -  {self.phase.replace('_', ' ')}", fill=(101, 224, 204), font=font)
        draw.rectangle((0, image.shape[0]-34, image.shape[1], image.shape[0]), fill=(18, 28, 41))
        draw.text((18, image.shape[0]-29), "CONTACT GRASP  |  DRY POUR: NO LIQUID SIMULATION", fill=(215, 225, 236), font=font)
        frame = np.asarray(canvas)
        if self.writer:
            self.writer.append_data(frame)
        if self.phase in ("READY", "DONE"):
            canvas.save(self.args.out/f"{self.phase.lower()}.png")

    def run(self):
        self.hold(.6)
        self.event("READY")
        self.hold(.4)
        for index, (name, _) in enumerate(COLORS[:self.args.cups]):
            self.index = index
            pos, _ = self.object_pose(name)
            radial = np.r_[pos[:2]/np.linalg.norm(pos[:2]), 0.0]
            grasp = pos.copy()
            grasp[2] += .002
            pre = grasp - .035*radial
            self.event("APPROACH", color=name)
            self.move([*pre[:2], .11], 2.2, arc=True)
            self.move(pre, 1.7)
            self.move(grasp, 1.2)
            self.event("GRASP")
            self.close_grip()
            self.verify_grasp(name)
            self.event("LIFT")
            self.move([*self.tcp()[:2], .165], 2.1)
            lifted, tilt = self.object_pose(name)
            if lifted[2] < .12 or tilt > 12:
                raise RuntimeError(f"Lift verification failed: {name} z={lifted[2]}, tilt={tilt}")
            self.event("MOVE_TO_BIN")
            radial_bin = BIN_XY/np.linalg.norm(BIN_XY)
            self.move([*(BIN_XY+.040*radial_bin), .160], 2.2, arc=True)
            upright = self.setpoint.copy()
            down = upright.copy()
            down[3] -= math.pi/2
            for cycle in range(3):
                self.event("DRY_POUR", cycle=cycle+1)
                self.move_joints(down, 1.0)
                dumped, tilt = self.object_pose(name)
                lip = self.data.site(f"lip_{name}").xpos.copy()
                normals = np.array([[math.cos(i*math.pi/3), math.sin(i*math.pi/3)] for i in range(6)])
                inside_bin = bool(np.max(normals @ (lip[:2]-BIN_XY)) < .045)
                if tilt < 75 or not inside_bin or lip[2] < .052:
                    raise RuntimeError(f"Pour pose verification failed: tilt={tilt}, lip={lip}")
                self.hold(.20)
                self.event("RETURN_UPRIGHT")
                self.move_joints(upright, 1.0)
            self.event("MOVE_TO_STACK")
            self.move([*STACK_XY, .170], 3.5, arc=True)
            current_pos, _ = self.object_pose(name)
            offset = current_pos-self.tcp()
            target = np.r_[STACK_XY, .006+CUP_HEIGHT*(index+.5)+.001]
            tcp_target = target-offset
            self.event("LOWER_TO_STACK", level=index+1)
            self.move(tcp_target, 2.0)
            self.event("RELEASE")
            self.release(name)
            back = self.tcp().copy()
            back[:2] -= .04*STACK_XY/np.linalg.norm(STACK_XY)
            self.move(back, 1.2)
            self.move([*back[:2], .18], 1.5)
            self.hold(.7)
            placed, tilt = self.object_pose(name)
            expected_z = .006+CUP_HEIGHT*(index+.5)
            success = np.linalg.norm(placed[:2]-STACK_XY) < .004 and abs(placed[2]-expected_z) < .004 and tilt < 10
            self.result["cups"].append({"name": name, "success": bool(success), "xyz_m": placed.tolist(), "tilt_deg": tilt})
            if not success:
                raise RuntimeError(f"Stack verification failed: {name} position={placed}, tilt={tilt}")
            self.event("STACK_VERIFIED", color=name, level=index+1)
        self.event("DONE")
        self.hold(1.5)
        for i, (name, _) in enumerate(COLORS[:self.args.cups]):
            p, tilt = self.object_pose(name)
            if np.linalg.norm(p[:2]-STACK_XY) > .004 or abs(p[2]-(.006+CUP_HEIGHT*(i+.5))) > .004 or tilt > 10:
                raise RuntimeError(f"Final stack unstable: {name}")
        self.result["success"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent/"outputs")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Do not synchronize viewer to wall clock")
    parser.add_argument("--cups", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--fps", type=int, choices=range(1, 61), default=20)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    scene = build_scene(args.repo.resolve(), args.out/"scene.xml")
    model = mujoco.MjModel.from_xml_path(str(scene))
    demo = Demo(model, args)
    if args.video:
        import imageio.v2 as imageio
        demo.renderer = mujoco.Renderer(model, height=480, width=720)
        demo.writer = imageio.get_writer(args.out/"demo.mp4", fps=args.fps, codec="libx264", quality=8,
                                        ffmpeg_params=["-threads", "2"])
    if args.viewer:
        from mujoco import viewer as mj_viewer
        context = mj_viewer.launch_passive(model, demo.data)
    else:
        context = nullcontext(None)
    exit_code = 0
    try:
        with context as viewer:
            demo.viewer = viewer
            if viewer:
                viewer.cam.lookat[:] = [0, .10, .10]
                viewer.cam.distance = .63
                viewer.cam.azimuth = 55
                viewer.cam.elevation = -28
            demo.run()
    except (Exception, KeyboardInterrupt) as e:
        exit_code = 1
        demo.result["error"] = str(e) or "Viewer closed by user"
        print(f"STOP: {demo.result['error']}", file=sys.stderr, flush=True)
    finally:
        demo.result.update(sim_time=float(demo.data.time), mujoco_version=mujoco.__version__)
        (args.out/"result.json").write_text(json.dumps(demo.result, indent=2), encoding="utf-8")
        (args.out/"events.json").write_text(json.dumps(demo.events, indent=2), encoding="utf-8")
        with (args.out/"trajectory.jsonl").open("w", encoding="utf-8") as f:
            for row in demo.rows:
                f.write(json.dumps(row)+"\n")
        if demo.writer:
            demo.writer.close()
        if demo.renderer:
            demo.renderer.close()
    print(json.dumps(demo.result, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

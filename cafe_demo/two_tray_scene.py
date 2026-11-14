"""Two return trays and sorting stations. Scene preview only; no sorting policy."""
import argparse
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from run_demo import build_scene, add, vec, HOME, OPEN

TRAYS = {"tray_1": [-.095, .145], "tray_2": [.095, .145]}
TRAY_YAWS = {name: -math.atan2(xy[0],xy[1]) for name,xy in TRAYS.items()}
STATIONS = {
    "wet_waste": {"xy": [-.190, .040], "tag": "W", "label": "Drink + food waste"},
    "general_waste": {"xy": [.175, .030], "tag": "G", "label": "General waste"},
    "fork_bin": {"xy": [-.170, -.025], "tag": "F", "label": "Fork collection"},
    "cup_stack": {"xy": [.170, -.035], "tag": "C", "label": "Cup stack"},
    "plate_stack": {"xy": [-.150, -.080], "tag": "P", "label": "Plate stack"},
    "tray_stack": {"xy": TRAYS["tray_2"], "tag": "R", "label": "Empty trays: reuse T2"},
}
COLORS = {"cup_1": ".87 .23 .18 1", "cup_2": ".17 .49 .87 1", "cup_3": ".20 .69 .48 1"}


def geom(parent, **kwargs):
    defaults = dict(friction=".8 .01 .001", solref=".004 1", solimp=".95 .99 .001", condim="4")
    defaults.update(kwargs)
    return add(parent, "geom", **defaults)


def dynamic(world, name, xyz):
    body = add(world, "body", name=name, pos=vec(xyz))
    add(body, "freejoint", name=f"{name}_free")
    return body


def tray(world, name, xy, color):
    b = dynamic(world, name, [*xy, .0021])
    b.set("euler",vec([0,0,TRAY_YAWS[name]]))
    geom(b, name=f"{name}_base", type="box", size=".070 .0575 .002", rgba=color, mass=".018")
    for x in [-.069, .069]:
        geom(b, type="box", pos=vec([x, 0, .0035]), size=".001 .0575 .0035", rgba=color, mass=".001")
    for y in [-.0565, .0565]:
        geom(b, type="box", pos=vec([0, y, .0035]), size=".068 .001 .0035", rgba=color, mass=".001")
    geom(b,type="capsule",fromto="0 .0565 .005 0 .073 .026",size=".002",rgba=".32 .37 .42 1",mass=".001")
    geom(b,type="cylinder",pos="0 .073 .026",size=".007 .004",rgba=".86 .58 .20 1",mass=".002")
    add(b, "site", name=f"{name}_grasp", pos="0 .073 .026", size=".001", rgba="0 0 0 0")


def cup(world, name, xy, liquid):
    b = dynamic(world, name, [*xy, .0185])
    color = COLORS[name]
    geom(b, type="cylinder", pos="0 0 -.013", size=".010 .001", rgba=color, mass=".002")
    for i in range(16):
        a = 2*math.pi*i/16
        geom(b, type="box", pos=vec([.009*math.cos(a), .009*math.sin(a), .001]),
             euler=vec([0, 0, a]), size=".001 .00185 .013", rgba=color, mass=".000375")
    if liquid:
        geom(b, name=f"{name}_liquid_visual", type="cylinder", pos="0 0 .004", size=".0078 .0003",
             rgba=".14 .55 .74 .85", contype="0", conaffinity="0", mass="0")
    add(b, "site", name=f"{name}_grasp", pos="0 0 .008", size=".001", rgba="0 0 0 0")


def plate(world, name, xy):
    b = dynamic(world, name, [*xy, .0058])
    geom(b, type="cylinder", size=".017 .0015", rgba=".94 .94 .88 1", mass=".004")
    for i in range(20):
        a = 2*math.pi*i/20
        geom(b, type="box", pos=vec([.016*math.cos(a), .016*math.sin(a), .0075]),
             euler=vec([0, 0, a]), size=".001 .0026 .006", rgba=".90 .90 .86 1", mass=".00005")
    add(b, "site", name=f"{name}_grasp", pos="0 0 .010", size=".001", rgba="0 0 0 0")


def fork(world, name, xy):
    b = dynamic(world, name, [*xy, .0065])
    geom(b, type="box", pos="0 -.008 0", size=".004 .009 .002", rgba=".86 .50 .16 1", mass=".0007")
    geom(b, type="box", pos="0 .004 0", size=".0013 .005 .001", rgba=".72 .77 .82 1", mass=".0002")
    geom(b, type="box", pos="0 .010 0", size=".004 .002 .001", rgba=".72 .77 .82 1", mass=".0002")
    for x in [-.003, 0, .003]:
        geom(b, type="box", pos=vec([x, .015, 0]), size=".00065 .004 .001", rgba=".8 .84 .87 1", mass=".00005")
    geom(b,type="box",pos="0 .001 .007",size=".004 .004 .009",rgba=".90 .56 .22 1",mass=".0003")
    add(b, "site", name=f"{name}_grasp", pos="0 .001 .011", size=".001", rgba="0 0 0 0")


def tissue(world, name, xy):
    # Compact 18 mm cube; tray surface is at z=4 mm.
    b = dynamic(world, name, [*xy, .0135])
    geom(b,type="box",pos="0 0 0",size=".009 .009 .009",rgba=".97 .95 .88 1",mass=".001")
    # Downward grasp remains centered in x/y, slightly above the cube center.
    add(b,"site",name=f"{name}_grasp",pos="0 0 .004",size=".001",rgba="0 0 0 0")


def box_bin(world, name, xy, half, height, color):
    b = add(world, "body", name=name, pos=vec([*xy, 0]))
    geom(b, type="box", pos="0 0 .0015", size=vec([*half, .0015]), rgba=color)
    for x in [-half[0], half[0]]:
        geom(b, type="box", pos=vec([x, 0, height/2]), size=vec([.0015, half[1], height/2]), rgba=color)
    for y in [-half[1], half[1]]:
        geom(b, type="box", pos=vec([0, y, height/2]), size=vec([half[0], .0015, height/2]), rgba=color)


def make_scene(repo, out):
    scene = build_scene(repo, out/"two_tray_scene.xml")
    tree = ET.parse(scene)
    root, registry = tree.getroot(), []
    root.set("model", "raccoon_two_return_trays")
    root.find("compiler").set("inertiafromgeom", "auto")
    world = root.find("worldbody")
    for child in list(world):
        if child.get("name", "").startswith("cup_") or child.get("name") in ("source_zone", "stack_base", "hex_bin"):
            world.remove(child)
    camera = world.find("camera[@name='overview']")
    pos, target = np.array([.40, .57, .53]), np.array([0, .070, .035])
    z = pos-target; z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    camera.set("pos", vec(pos)); camera.set("xyaxes", vec([*x, *y])); camera.set("fovy", "44")
    add(world, "camera", name="top", pos="0 .060 .85", xyaxes="1 0 0 0 1 0", fovy="43")
    for name, xy in TRAYS.items():
        tray(world, name, xy, ".54 .40 .27 1" if name=="tray_1" else ".36 .49 .56 1")
    specs = [
        ("cup_1", "cup", "tray_1", [-.045, .032], "liquid"),
        ("cup_2", "cup", "tray_1", [.002, .032], "empty"),
        ("plate_1", "plate", "tray_1", [.040, -.005], "empty"),
        ("fork_1", "fork", "tray_1", [-.047, -.028], None),
        ("fork_2", "fork", "tray_1", [-.007, -.028], None),
        ("tissue_1", "tissue", "tray_1", [.018, -.042], None),
        ("cup_3", "cup", "tray_2", [-.038, .032], "liquid"),
        ("plate_2", "plate", "tray_2", [.027, .000], "cake"),
        ("fork_3", "fork", "tray_2", [-.035, -.028], None),
        ("tissue_2", "tissue", "tray_2", [.022, -.037], None),
    ]
    for name, kind, source, offset, contents in specs:
        yaw=TRAY_YAWS[source]
        rotation=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
        xy = np.asarray(TRAYS[source])+rotation@offset
        if kind=="cup": cup(world, name, xy, contents=="liquid")
        else: {"plate":plate, "fork":fork, "tissue":tissue}[kind](world, name, xy)
        world.find(f"body[@name='{name}']").set("euler",vec([0,0,yaw]))
        goal = {"cup":"cup_stack", "plate":"plate_stack", "fork":"fork_bin", "tissue":"general_waste"}[kind]
        registry.append(dict(name=name, kind=kind, tray=source, contents=contents, destination=goal, xy_m=xy.tolist()))
        if contents=="cake":
            cake = dynamic(world, "cake_piece", [*xy, .0118])
            geom(cake, type="box", size=".006 .006 .004", rgba=".45 .22 .10 1", mass=".0015", friction=".25 .00001 .00001", condim="3", priority="1")
            geom(cake, type="box", pos="0 0 .0042", size=".006 .006 .0002", rgba=".95 .86 .68 1", mass="0", contype="0", conaffinity="0")
    wet = add(world, "body", name="wet_waste", pos=vec([*STATIONS["wet_waste"]["xy"], 0]))
    geom(wet, type="cylinder", pos="0 0 .002", size=".031 .002", rgba=".16 .52 .56 1")
    for i in range(6):
        a = i*math.pi/3
        geom(wet, type="box", pos=vec([.027*math.cos(a), .027*math.sin(a), .022]),
             euler=vec([0,0,a]), size=".0015 .0165 .022", rgba=".18 .65 .66 1")
    for offset in [-.014,-.007,0,.007,.014]:
        for axis in [0,1]:
            p = [offset,0,.031] if axis==0 else [0,offset,.031]
            size = [.0004,.020,.0004] if axis==0 else [.020,.0004,.0004]
            geom(wet,type="box",pos=vec(p),size=vec(size),rgba=".56 .65 .67 1")
    box_bin(world, "general_waste", STATIONS["general_waste"]["xy"], [.025,.022], .035, ".72 .38 .18 1")
    box_bin(world, "fork_bin", STATIONS["fork_bin"]["xy"], [.027,.027], .012, ".60 .66 .73 1")
    for name,color in [("cup_stack",".19 .54 .78 1"),("plate_stack",".60 .42 .74 1")]:
        b = add(world,"body",name=name,pos=vec([*STATIONS[name]["xy"],.003]))
        geom(b,type="cylinder",size=".023 .003",rgba=color)
    add(world,"site",name="tray_stack_goal",pos=vec([*STATIONS["tray_stack"]["xy"],.002]),size=".001",rgba="0 0 0 0")
    from crumpled_tissue import install_crumpled_tissues
    install_crumpled_tissues(root)
    tree.write(scene,encoding="utf-8",xml_declaration=True)
    manifest = dict(trays=TRAYS, objects=registry, stations=STATIONS,
                    notes=["Scene only: no automatic sorting actions.", "Liquids are visual markers, not fluids.",
                           "Tissues are rigid compound bodies, not deformable paper.",
                           "Grasp sites are proposed; complete motion reachability has not been validated."],
                    contents_objects=[dict(name="cake_piece", container="plate_2", destination="wet_waste")])
    (out/"scene_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return scene, manifest


def hold_robot(model,data):
    kp=model.actuator_gainprm[:4,0]
    data.ctrl[:4]=np.clip(HOME+data.qfrc_bias[:4]/kp,model.actuator_ctrlrange[:4,0],model.actuator_ctrlrange[:4,1])
    data.ctrl[4]=OPEN
    mujoco.mj_step(model,data)


def check_scene(model,data,manifest):
    report={"counts":{},"objects":[],"success":True,"motion_validated":False}
    for obj in manifest["objects"]:
        kind=obj["kind"]; report["counts"][kind]=report["counts"].get(kind,0)+1
        body=data.body(obj["name"]); tb=data.body(obj["tray"])
        local=tb.xmat.reshape(3,3).T@(body.xpos-tb.xpos)
        ok=bool(abs(local[0])<.068 and abs(local[1])<.055 and 0<local[2]<.04)
        supported=False
        for c in data.contact[:data.ncon]:
            pair={int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])}
            if pair=={body.id,tb.id} and c.dist<.0002: supported=True
        ok=ok and supported
        report["objects"].append(dict(name=obj["name"],on_expected_tray=ok,position_m=body.xpos.tolist()))
        report["success"]=report["success"] and ok
    report["success"]=report["success"] and report["counts"]==dict(cup=3,plate=2,fork=3,tissue=2)
    return report


def render(model,data,out,camera_name):
    from PIL import Image,ImageDraw,ImageFont
    renderer=mujoco.Renderer(model,height=720,width=900)
    renderer.update_scene(data,camera=camera_name)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
    canvas=Image.new("RGB",(1250,720),(19,29,42)); canvas.paste(Image.fromarray(renderer.render()),(0,0));renderer.close()
    draw=ImageDraw.Draw(canvas)
    fontpath="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    def font(size): return ImageFont.truetype(fontpath,size) if Path(fontpath).exists() else ImageFont.load_default()
    draw.text((920,24),"CAFE RETURN",font=font(26),fill="white")
    draw.text((920,58),"2 trays / 10 items",font=font(20),fill="#7fe3cc")
    draw.text((920,112),"T1  Tray 1",font=font(20),fill="#efc08e")
    draw.text((920,143),"2 cups / 1 plate",font=font(17),fill="white")
    draw.text((920,170),"2 forks / 1 tissue",font=font(17),fill="white")
    draw.text((920,220),"T2  Tray 2",font=font(20),fill="#91cbe7")
    draw.text((920,251),"1 cup / 1 plate",font=font(17),fill="white")
    draw.text((920,278),"1 fork / 1 tissue",font=font(17),fill="white")
    labels={"T1": [*TRAYS["tray_1"], .035],"T2/R":[*TRAYS["tray_2"], .035]}
    for i, station in enumerate(STATIONS.values()):
        draw.text((920,343+i*42),station["tag"]+"  "+station["label"],font=font(17),fill="#d9e4ef")
        if station["tag"]!="R":labels[station["tag"]]=[*station["xy"], .05 if station["tag"] in ("W","G") else .012]
    camera=model.camera(camera_name); cid=camera.id
    f=360/math.tan(math.radians(float(camera.fovy[0]))/2)
    for tag,xyz in labels.items():
        p=data.cam_xmat[cid].reshape(3,3).T@(np.array(xyz)-data.cam_xpos[cid])
        x,y=450+f*p[0]/(-p[2]),360-f*p[1]/(-p[2])
        width=max(17,len(tag)*6)
        draw.rounded_rectangle((x-width,y-29,x+width,y-4),radius=5,fill="#132233",outline="#8fe1d0",width=1)
        draw.text((x-width+5,y-27),tag,font=font(16),fill="white")
    draw.text((920,650),"LAYOUT PREVIEW",font=font(19),fill="#7fe3cc")
    draw.text((920,680),"Run actions with run_sorting.py",font=font(14),fill="#a6b2c2")
    canvas.save(out/f"{camera_name}.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"two_tray_outputs")
    parser.add_argument("--viewer",action="store_true")
    parser.add_argument("--render",action="store_true")
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    scene,manifest=make_scene(args.repo.resolve(),args.out.resolve())
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    data.qpos[:4]=HOME;data.qpos[4]=OPEN;data.ctrl[:4]=HOME;data.ctrl[4]=OPEN
    mujoco.mj_forward(model,data)
    for _ in range(round(1.0/model.opt.timestep)): hold_robot(model,data)
    report=check_scene(model,data,manifest)
    (args.out/"layout_check.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
    if not report["success"]: return 1
    if args.render:
        for camera in ("overview","top"): render(model,data,args.out,camera)
    if args.viewer:
        from mujoco import viewer
        with viewer.launch_passive(model,data) as v:
            v.cam.lookat[:]=[0,.065,.03];v.cam.distance=.72;v.cam.azimuth=55;v.cam.elevation=-55
            while v.is_running():
                started=time.monotonic()
                for _ in range(round(1/60/model.opt.timestep)): hold_robot(model,data)
                v.sync();time.sleep(max(0,1/60-(time.monotonic()-started)))
    return 0


if __name__=="__main__":
    try: raise SystemExit(main())
    except KeyboardInterrupt: pass

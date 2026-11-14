"""Rule-based two-tray sorting. Physics contacts, explicit grasp poses, no VLM."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

from run_demo import Demo, HOME, OPEN
from two_tray_scene import make_scene, STATIONS, TRAYS, TRAY_YAWS

DOWN=-math.pi/2
ORDER=["tissue_1","fork_1","fork_2","cup_1","cup_2","plate_1",
       "tissue_2","fork_3","cup_3","plate_2"]


class SortingDemo(Demo):
    def __init__(self,model,args,manifest):
        super().__init__(model,args)
        self.manifest=manifest
        self.objects={o["name"]:o for o in manifest["objects"]}
        self.pitch=0.0
        self.active="initialization"
        self.counts={"cup":0,"plate":0,"fork":0,"tissue":0}
        self.result={"success":False,"completed":[],"uses_vlm":False,"grasp_mode":"physical",
                     "liquids":"visual-state update after dry tilt, not fluid simulation",
                     "paper":"rigid compound body","hardware_verified":False}

    def actual_pitch(self,data=None):
        data=self.data if data is None else data
        q1=float(data.qpos[self.arm_q[0]])
        radial=np.array([-math.sin(q1),math.cos(q1),0])
        forward=data.body("Link4").xmat.reshape(3,3)[:,1]
        return math.atan2(forward[2],np.dot(forward,radial))

    def ik(self,target,pitch=0.0,seed=None):
        target=np.asarray(target,float)
        self.ikdata.qpos[:]=self.data.qpos
        def residual(q):
            self.ikdata.qpos[self.arm_q]=q
            mujoco.mj_kinematics(self.model,self.ikdata)
            return np.r_[self.ikdata.site_xpos[self.tcp_id]-target,
                         .08*(self.actual_pitch(self.ikdata)-pitch)]
        initial=np.clip(self.setpoint if seed is None else seed,self.lower,self.upper)
        solution=least_squares(residual,initial,bounds=(self.lower,self.upper),
            xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=150)
        error=residual(solution.x)
        if np.linalg.norm(error[:3])>.0007 or abs(error[3])>.0007:
            raise RuntimeError(f"IK unavailable: xyz={target}, pitch={math.degrees(pitch):.1f}, residual={error}")
        return solution.x

    def move_pose(self,target,pitch=None,duration=1.8,arc=False):
        target=np.asarray(target,float)
        pitch=self.pitch if pitch is None else pitch
        start=self.tcp();p0=self.actual_pitch()
        r0,r1=np.linalg.norm(start[:2]),np.linalg.norm(target[:2])
        a0,a1=math.atan2(start[1],start[0]),math.atan2(target[1],target[0])
        if a0 < -math.pi/6-.01:a0+=2*math.pi
        if a1 < -math.pi/6-.01:a1+=2*math.pi
        da=a1-a0
        n=max(1,int(duration*25));qs=[self.setpoint.copy()]
        for i in range(1,n+1):
            s=i/n;b=10*s**3-15*s**4+6*s**5
            pt=start+b*(target-start)
            if arc:
                a=a0+b*da;r=r0+b*(r1-r0);pt[:2]=r*np.array([math.cos(a),math.sin(a)])
            qs.append(self.ik(pt,p0+b*(pitch-p0),seed=qs[-1]))
        for i in range(n):
            steps=max(1,round(duration/n/self.model.opt.timestep))
            velocity=(qs[i+1]-qs[i])/(steps*self.model.opt.timestep)
            for k in range(1,steps+1):self.step(qs[i]+(qs[i+1]-qs[i])*k/steps,velocity)
        self.setpoint=qs[-1];self.pitch=pitch;self.hold(.20)
        err=np.linalg.norm(self.tcp()-target)
        if err>.004:raise RuntimeError(f"Tracking error {err*1000:.2f}mm, target {target}, actual {self.tcp()}")

    def event(self,phase,**details):
        self.phase=phase
        event=dict(time=round(self.data.time,4),object=self.active,phase=phase,**details)
        self.events.append(event);print(json.dumps(event),flush=True)

    def pose(self,name):
        b=self.data.body(name)
        tilt=math.degrees(math.acos(np.clip(b.xmat.reshape(3,3)[2,2],-1,1)))
        return b.xpos.copy(),tilt

    def output_frame(self):
        names=list(self.objects) if hasattr(self,"objects") else []
        self.rows.append(dict(time=float(self.data.time),object=self.active,phase=self.phase,
            q_rad=self.data.qpos[self.arm_q].tolist(),qpos=self.data.qpos.tolist(),tcp_m=self.tcp().tolist(),
            objects={name:self.pose(name)[0].tolist() for name in names}))
        if self.renderer is None:return
        from PIL import Image,ImageDraw,ImageFont
        self.renderer.update_scene(self.data,camera="overview")
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
        canvas=Image.fromarray(self.renderer.render());draw=ImageDraw.Draw(canvas)
        fp="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font=ImageFont.truetype(fp,15) if Path(fp).exists() else ImageFont.load_default()
        draw.rectangle((0,0,720,57),fill="#132233")
        draw.text((12,8),f"RULE-BASED | {self.active} | {self.phase}",font=font,fill="white")
        draw.text((12,32),f"Completed: {len(self.result['completed'])}/11 | Plates: downward grasp",font=font,fill="#8fe1d0")
        draw.rectangle((0,453,720,480),fill="#132233")
        draw.text((12,457),"CONTACT PHYSICS | LIQUIDS VISUAL ONLY | NO VLM",font=font,fill="white")
        if self.writer:self.writer.append_data(np.asarray(canvas))
        if self.phase in ("READY","DONE"):canvas.save(self.args.out/f"{self.phase.lower()}.png")

    def finger_contacts(self,name):
        target=self.model.body(name).id
        fingers={self.model.body(f"Gripper_{side}_Finger").id:side for side in ("L","R")}
        touching=set()
        for c in self.data.contact[:self.data.ncon]:
            a,b=self.model.geom_bodyid[[c.geom1,c.geom2]]
            other=b if a==target else a if b==target else -1
            if other in fingers and c.dist<.0002:touching.add(fingers[other])
        return touching

    def open_to(self,target):
        start=self.grip;n=round(.35/self.model.opt.timestep)
        for i in range(1,n+1):
            self.grip=start+(target-start)*i/n;self.step(self.setpoint)
        self.hold(.15)

    def pickup(self,name,kind):
        vertical=kind in ("plate","fork","tissue")
        pitch=DOWN if vertical else 0.0
        opening={"cup":-.1,"plate":OPEN,"fork":-.35,"tissue":-.1,"tray":-.2}[kind]
        self.open_to(opening)
        target=self.data.site(f"{name}_grasp").xpos.copy()
        initial,_=self.pose(name)
        self.event("APPROACH",grasp_pitch_deg=math.degrees(pitch),grasp_point_m=target.tolist())
        if vertical:
            self.move_pose([*target[:2],.070],pitch,2.1,arc=True)
            self.event("DESCEND_VERTICAL")
            self.move_pose(target,pitch,1.6)
        else:
            radial=np.r_[target[:2]/np.linalg.norm(target[:2]),0]
            pre=target-.030*radial
            self.move_pose([*pre[:2],.105],pitch,2.1,arc=True)
            self.move_pose(pre,pitch,1.5)
            self.move_pose(target,pitch,1.2)
        self.event("CLOSE_GRIPPER")
        self.close_grip()
        self.verify_grasp(name)
        self.event("LIFT_VERTICAL")
        self.move_pose([*self.tcp()[:2],.070 if vertical else .145],pitch,1.9)
        after,tilt=self.pose(name)
        if after[2]<initial[2]+.025 or (kind!="tissue" and tilt>18):
            raise RuntimeError(f"Lift failed for {name}: before={initial}, after={after}, tilt={tilt}")

    def retract(self,vertical):
        p=self.tcp()
        if vertical:
            # Pull toward the base at release clearance before lifting at long reach.
            radial=p[:2]/np.linalg.norm(p[:2])
            self.move_pose([*(p[:2]-.018*radial),p[2]],DOWN,1.0)
            self.move_pose([*(p[:2]-.018*radial),.070],DOWN,1.2)
        else:
            radial=p[:2]/np.linalg.norm(p[:2])
            self.move_pose([*(p[:2]-.012*radial),p[2]],0.0,1.1)
            self.move_pose([*(p[:2]-.012*radial),.140],0.0,1.4)

    def pour_cup(self,name):
        xy=np.array(STATIONS["wet_waste"]["xy"]);radial=xy/np.linalg.norm(xy)
        self.event("MOVE_TO_WET_WASTE")
        self.move_pose([*(xy+.050*radial),.155],0.0,2.3,arc=True)
        upright=self.setpoint.copy();down=upright.copy();down[3]-=math.pi/2
        for i in range(3):
            self.event("DRY_POUR_CUP",cycle=i+1)
            self.move_joints(down,1.0);self.hold(.15)
            _,tilt=self.pose(name)
            if tilt<75:raise RuntimeError(f"Cup tilt failed: {tilt}")
            self.move_joints(upright,1.0)
        self.pitch=0.0
        gid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,f"{name}_liquid_visual")
        if gid>=0:self.model.geom_rgba[gid,3]=0
        self.event("LIQUID_VISUAL_CLEARED",fluid_simulation=False)

    def empty_plate(self,name):
        xy=np.array(STATIONS["wet_waste"]["xy"]);radial=xy/np.linalg.norm(xy)
        self.event("MOVE_PLATE_TO_WET_WASTE")
        self.move_pose([*(xy-.020*radial),.075],DOWN,2.3,arc=True)
        down=self.setpoint.copy()
        for i in range(3):
            self.event("TIP_PLATE",cycle=i+1)
            tip=down.copy();tip[3]+=math.radians(155)
            self.move_joints(tip,1.2);self.hold(.5)
            self.event("PLATE_TILT_MEASURE",tilt_deg=self.pose(name)[1],cake_position=self.pose("cake_piece")[0].tolist(),q4=float(self.data.qpos[3]))
            less=down.copy();less[3]+=math.radians(125)
            self.move_joints(less,.5)
        self.move_joints(down,1.2);self.pitch=DOWN;self.hold(.4)
        if self.objects[name]["contents"] != "cake":
            return
        cake=self.data.body("cake_piece").xpos.copy()
        normals=np.array([[math.cos(i*math.pi/3),math.sin(i*math.pi/3)] for i in range(6)])
        ok=np.max(normals@(cake[:2]-xy))<.027 and .001<cake[2]<.050
        self.event("CHECK_FOOD_REMOVAL",cake_position_m=cake.tolist(),success=bool(ok))
        if not ok:raise RuntimeError(f"Cake missed collector: {cake}")

    def place(self,name,kind):
        destination=self.objects[name]["destination"]
        xy=np.array(STATIONS[destination]["xy"]);level=self.counts[kind]
        vertical=kind!="cup"
        self.event("MOVE_TO_DESTINATION",destination=destination,level=level+1)
        if kind=="tissue":
            xy[1]+=(-.009 if level==0 else .009)
            p=self.tcp(); r=p[:2]/np.linalg.norm(p[:2])
            self.move_pose([*(.115*r),.085],DOWN,1.5)
            self.move_pose([*(xy/np.linalg.norm(xy)*.115),.085],DOWN,2.0,arc=True)
            self.move_pose([*xy,.055],DOWN,1.6)
        elif kind=="fork":
            p=self.tcp(); r=p[:2]/np.linalg.norm(p[:2])
            self.move_pose([*(.125*r),.075],DOWN,1.5)
            self.move_pose([*(xy/np.linalg.norm(xy)*.125),.075],DOWN,2.0,arc=True)
            self.move_pose([*xy,.055],DOWN,1.6)
        else:
            if vertical:
                p=self.tcp();r=p[:2]/np.linalg.norm(p[:2])
                self.move_pose([*(.115*r),.085],DOWN,1.5)
                self.move_pose([*(xy/np.linalg.norm(xy)*.115),.085],DOWN,2.0,arc=True)
                self.move_pose([*xy,.070],DOWN,1.6)
            else:self.move_pose([*xy,.140],0.0,2.5,arc=True)
            current,_=self.pose(name);offset=current-self.tcp()
            if kind=="cup":z=.006+.014+.028*level+.0015
            else:z=.006+.0015+.015*level+.0015
            self.event("LOWER_TO_STACK")
            self.move_pose(np.r_[xy,z]-offset,self.pitch,1.8)
        self.event("RELEASE")
        self.release(name)
        if kind=="plate":
            self.move_pose([*self.tcp()[:2],.070],DOWN,1.3)
        else:self.retract(vertical)
        self.hold(.6)
        p,tilt=self.pose(name)
        if kind in ("cup","plate"):
            z=.006+.014+.028*level if kind=="cup" else .006+.0015+.015*level
            ok=np.linalg.norm(p[:2]-xy)<.005 and abs(p[2]-z)<.005 and tilt<12
        elif kind=="fork":ok=np.max(np.abs(p[:2]-xy))<.025 and .001<p[2]<.05
        else:ok=abs(p[0]-STATIONS[destination]["xy"][0])<.024 and abs(p[1]-STATIONS[destination]["xy"][1])<.021 and .001<p[2]<.065
        self.event("VERIFY_PLACE",success=bool(ok),position_m=p.tolist(),tilt_deg=tilt)
        if not ok:raise RuntimeError(f"Placement failed for {name}: {p}, tilt={tilt}")
        self.counts[kind]+=1
        self.result["completed"].append(dict(name=name,kind=kind,destination=destination,position_m=p.tolist()))

    def stack_trays(self):
        self.active="tray_1"
        self.pickup("tray_1","tray")
        base,_=self.pose("tray_2")
        grip=self.data.site("tray_2_grasp").xpos.copy()
        self.event("MOVE_EMPTY_TRAY_TO_TRAY_2")
        self.move_pose([*grip[:2],.140],0,2.5,arc=True)
        pos,_=self.pose("tray_1");offset=pos-self.tcp()
        goal=base+np.array([0,0,.0125])
        self.move_pose(goal-offset,0,2.0)
        self.event("RELEASE_EMPTY_TRAY")
        self.release("tray_1");self.retract(False);self.hold(1)
        p1,t1=self.pose("tray_1");p2,t2=self.pose("tray_2")
        r1=self.data.body("tray_1").xmat.reshape(3,3);r2=self.data.body("tray_2").xmat.reshape(3,3)
        yaw_error=math.degrees(math.acos(np.clip(np.dot(r1[:,0],r2[:,0]),-1,1)))
        ok=np.linalg.norm(p1[:2]-p2[:2])<.007 and .005<p1[2]-p2[2]<.016 and t1<8 and t2<8 and yaw_error<8
        self.event("VERIFY_TRAY_STACK",success=bool(ok),yaw_error_deg=yaw_error)
        if not ok:raise RuntimeError(f"Tray stack failed: p1={p1}, p2={p2}, tilt={t1}, yaw error={yaw_error}")
        self.result["completed"].append(dict(name="tray_1",kind="tray",destination="tray_2"))

    def run(self):
        self.hold(1);self.event("READY");self.hold(.3)
        order=[self.args.only] if self.args.only else ORDER[:self.args.max_items]
        for i,name in enumerate(order):
            self.index=i;self.active=name;obj=self.objects[name]
            self.pickup(name,obj["kind"])
            if obj["kind"]=="cup":self.pour_cup(name)
            if obj["kind"]=="plate":self.empty_plate(name)
            self.place(name,obj["kind"])
        if not self.args.only and self.args.max_items>=10 and not self.args.skip_trays:self.stack_trays()
        self.event("DONE");self.hold(1)
        self.result["success"]=True
        self.result["complete_task"]=len(self.result["completed"])==11
        if self.result["complete_task"]:
            self.result["final_checks"]=self.verify_final()
            self.result["success"]=all(self.result["final_checks"].values())
            if not self.result["success"]:raise RuntimeError("Final arrangement changed after placement")

    def verify_final(self):
        checks={}
        for name,obj in self.objects.items():
            p,tilt=self.pose(name);xy=np.array(STATIONS[obj['destination']]['xy']);kind=obj['kind']
            if kind in ('cup','plate'):
                level=int(name.rsplit('_',1)[1])-1
                z=.006+.014+.028*level if kind=='cup' else .006+.0015+.015*level
                ok=np.linalg.norm(p[:2]-xy)<.005 and abs(p[2]-z)<.005 and tilt<12
            elif kind=='fork':ok=np.max(np.abs(p[:2]-xy))<.025 and .001<p[2]<.05
            else:ok=abs(p[0]-xy[0])<.024 and abs(p[1]-xy[1])<.021 and .001<p[2]<.065
            checks[name]=bool(ok)
        a,ta=self.pose('tray_1');b,tb=self.pose('tray_2')
        checks['trays']=bool(np.linalg.norm(a[:2]-b[:2])<.007 and .005<a[2]-b[2]<.016 and ta<8 and tb<8)
        self.event('FINAL_ARRANGEMENT',checks=checks)
        return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"sorting_outputs")
    p.add_argument("--viewer",action="store_true");p.add_argument("--video",action="store_true")
    p.add_argument("--fast",action="store_true");p.add_argument("--fps",type=int,default=12)
    p.add_argument("--only",choices=ORDER);p.add_argument("--max-items",type=int,default=10)
    p.add_argument("--skip-trays",action="store_true")
    args=p.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    scene,manifest=make_scene(args.repo.resolve(),args.out)
    m=mujoco.MjModel.from_xml_path(str(scene));demo=SortingDemo(m,args,manifest)
    if args.video:
        import imageio.v2 as imageio
        demo.renderer=mujoco.Renderer(m,height=480,width=720)
        demo.writer=imageio.get_writer(args.out/"sorting.mp4",fps=args.fps,codec="libx264",quality=8,ffmpeg_params=["-threads","2"])
    if args.viewer:
        from mujoco import viewer
        ctx=viewer.launch_passive(m,demo.data)
    else:ctx=nullcontext(None)
    code=0
    try:
        with ctx as v:
            demo.viewer=v
            if v:v.cam.lookat[:]=[0,.06,.05];v.cam.distance=.70;v.cam.azimuth=55;v.cam.elevation=-45
            demo.run()
    except (Exception,KeyboardInterrupt) as e:
        demo.result["error"]=str(e) or "interrupted";print("STOP:",e,flush=True);code=1
    finally:
        demo.result["sim_time"]=float(demo.data.time)
        for name,content in [("result.json",demo.result),("events.json",demo.events)]:
            (args.out/name).write_text(json.dumps(content,indent=2),encoding="utf-8")
        (args.out/"trajectory.jsonl").write_text("\n".join(json.dumps(row) for row in demo.rows),encoding="utf-8")
        np.savez_compressed(args.out/"final_state.npz",qpos=demo.data.qpos,qvel=demo.data.qvel,ctrl=demo.data.ctrl)
        if demo.writer:demo.writer.close()
        if demo.renderer:demo.renderer.close()
    print(json.dumps(demo.result,indent=2));return code


if __name__=="__main__":raise SystemExit(main())

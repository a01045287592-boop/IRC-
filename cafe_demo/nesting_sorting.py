"""Two-tray sorting with tapered nesting cups; all cups/plates visit waste bin."""
import argparse,json,math,time
from pathlib import Path
from contextlib import nullcontext
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import nesting_cups as nc
from run_sorting import SortingDemo,ORDER,DOWN
from two_tray_scene import make_scene,STATIONS,geom
from run_demo import vec

# Standalone test and full scene use different station layouts.
nc.STACK=np.array([.170,-.035])
STATIONS['cup_stack']['xy']=nc.STACK.tolist()
STATIONS['wet_waste']['xy']=[-.190,.040]

def full_scene(repo,out):
    path,manifest=make_scene(repo,out)
    tree=ET.parse(path);root=tree.getroot();world=root.find('worldbody')
    for obj in manifest['objects']:
        if obj['kind']!='cup':continue
        name=obj['name'];old=world.find(f"body[@name='{name}']")
        pos=np.fromstring(old.get('pos'),sep=' ');yaw=old.get('euler');world.remove(old)
        b=nc.tapered_cup(root,name,pos[:2]);b.set('pos',vec([*pos[:2],.004+nc.HEIGHT/2+.0005]))
        if yaw:b.set('euler',yaw)
        if obj['contents']=='liquid':
            geom(b,name=name+'_liquid_visual',type='cylinder',pos='0 0 0',size='.008 .0003',rgba='.14 .55 .74 .85',mass='0',contype='0',conaffinity='0')
    tree.write(path,encoding='utf-8',xml_declaration=True)
    return path,manifest

class FullNesting(nc.NestingDemo):
    def place(self,name,kind):
        if kind=='cup':
            self.nest(name,self.counts['cup']);self.counts['cup']+=1
        else:SortingDemo.place(self,name,kind)

    def empty_plate(self,name):
        # Same physical tipping action for every plate. Only food verification differs.
        xy=np.array(STATIONS['wet_waste']['xy']);radial=xy/np.linalg.norm(xy)
        self.event('MOVE_PLATE_TO_WET_WASTE')
        self.move_pose([*(xy-.020*radial),.075],DOWN,2.3,arc=True)
        down=self.setpoint.copy()
        for i in range(3):
            self.event('TIP_PLATE',cycle=i+1)
            tip=down.copy();tip[3]+=math.radians(155)
            self.move_joints(tip,1.2);self.hold(.5)
            less=down.copy();less[3]+=math.radians(125);self.move_joints(less,.5)
        self.move_joints(down,1.2);self.pitch=DOWN;self.hold(.4)
        if self.objects[name]['contents']!='cake':return
        cake=self.data.body('cake_piece').xpos.copy()
        normals=np.array([[math.cos(i*math.pi/3),math.sin(i*math.pi/3)] for i in range(6)])
        ok=np.max(normals@(cake[:2]-xy))<.027 and .001<cake[2]<.050
        self.event('CHECK_FOOD_REMOVAL',success=bool(ok),cake_position_m=cake.tolist())
        if not ok:raise RuntimeError('Cake missed collector')

    def run(self):
        self.hold(1);self.event('READY');self.hold(.3)
        for name in ORDER:
            self.active=name;obj=self.objects[name];self.pickup(name,obj['kind'])
            if obj['kind']=='cup':self.pour_cup(name)
            if obj['kind']=='plate':self.empty_plate(name)
            self.place(name,obj['kind'])
        self.stack_trays();self.hold(2)
        cup_checks=self.verify_nesting(3)
        saved=self.objects
        try:
            self.objects={k:v for k,v in saved.items() if v['kind']!='cup'}
            other_checks=SortingDemo.verify_final(self)
        finally:self.objects=saved
        if not all(other_checks.values()):raise RuntimeError('Final arrangement failed')
        self.result.update(success=True,complete_task=True,final_checks=dict(cups=cup_checks,other=other_checks))
        self.event('DONE');self.hold(.5)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--fast',action='store_true')
    p.add_argument('--fps',type=int,default=30);p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'nesting_sorting_outputs')
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(exist_ok=True,parents=True)
    path,manifest=full_scene(Path(__file__).resolve().parents[1],a.out)
    demo=FullNesting(mujoco.MjModel.from_xml_path(str(path)),a,manifest)
    if a.video:
        import imageio.v2 as imageio
        demo.renderer=mujoco.Renderer(demo.model,height=480,width=720)
        demo.writer=imageio.get_writer(a.out/'nesting_sorting.mp4',fps=a.fps,codec='libx264',ffmpeg_params=['-threads','2'])
    if a.viewer:
        from mujoco import viewer
        ctx=viewer.launch_passive(demo.model,demo.data)
    else:ctx=nullcontext(None)
    code=0;t=time.perf_counter()
    try:
        with ctx as v:
            demo.viewer=v
            if v:v.cam.lookat[:]=[0,.06,.05];v.cam.distance=.70;v.cam.azimuth=55;v.cam.elevation=-45
            demo.run()
    except (Exception,KeyboardInterrupt) as e:
        demo.result.update(success=False,error=str(e));print('STOP:',e,flush=True);code=1
    finally:
        demo.result.update(sim_seconds=float(demo.data.time),wall_seconds=time.perf_counter()-t)
        for name,data in [('result.json',demo.result),('events.json',demo.events)]: (a.out/name).write_text(json.dumps(data,indent=2))
        (a.out/'trajectory.jsonl').write_text('\n'.join(json.dumps(r) for r in demo.rows))
        np.savez_compressed(a.out/'final_state.npz',qpos=demo.data.qpos,qvel=demo.data.qvel,ctrl=demo.data.ctrl)
        if demo.writer:demo.writer.close()
        if demo.renderer:demo.renderer.close()
    print(json.dumps(demo.result,indent=2));return code
if __name__=='__main__':raise SystemExit(main())

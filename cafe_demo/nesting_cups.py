"""Isolated, contact-physics tapered cup nesting test. No hardware connection."""
import argparse,json,math,time
from pathlib import Path
from contextlib import nullcontext
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from run_demo import build_scene,add,vec
from two_tray_scene import geom,dynamic,COLORS,STATIONS,cup as old_cup
from run_sorting import SortingDemo

HEIGHT=.040
PITCH=.018
STACK=np.array([.170,.080])
SOURCES=[[-.10,.16],[0,.19],[.10,.16]]
STATIONS['cup_stack']['xy']=STACK.tolist()
STATIONS['wet_waste']['xy']=[-.170,.080]

def tapered_cup(root,name,xy):
    asset=root.find('asset');world=root.find('worldbody')
    b=dynamic(world,name,[*xy,HEIGHT/2+.0005])
    geom(b,name=name+'_bottom',type='cylinder',pos='0 0 -.019',size='.007 .001',rgba=COLORS[name],mass='.002')
    # Eight convex wall sectors: each is hollow as part of a compound body.
    for i in range(8):
        vertices=[]
        for z,outer in [(-.018,.00735),(.020,.014)]:
            for r in [outer-.001,outer]:
                for a in [i*math.pi/4,(i+1)*math.pi/4]:
                    vertices.extend([r*math.cos(a),r*math.sin(a),z])
        mesh=name+f'_sector_{i}'
        add(asset,'mesh',name=mesh,vertex=vec(vertices))
        geom(b,name=mesh,type='mesh',mesh=mesh,rgba=COLORS[name],mass='.00075')
    add(b,'site',name=name+'_grasp',pos='0 0 .014',size='.001',rgba='0 0 0 0')
    return b

def scene(repo,out,count=3,shape='tapered'):
    out.mkdir(parents=True,exist_ok=True);path=out/'nesting_scene.xml';build_scene(repo,path)
    tree=ET.parse(path);root=tree.getroot();root.find('compiler').set('inertiafromgeom','auto');world=root.find('worldbody')
    for c in list(world):
        if c.get('name','').startswith('cup_'):world.remove(c)
    objects=[]
    for i in range(count):
        name=f'cup_{i+1}'
        if shape=='tapered':tapered_cup(root,name,SOURCES[i])
        else:
            old_cup(world,name,SOURCES[i],False)
            world.find(f"body[@name='{name}']").set('pos',vec([*SOURCES[i],.0145]))
        objects.append(dict(name=name,kind='cup',contents='empty',destination='cup_stack'))
    # Closer view of source cups and nesting destination.
    camera=world.find("camera[@name='overview']")
    p=np.array([.39,.48,.34]);look=np.array([.035,.135,.06]);z=p-look;z/=np.linalg.norm(z)
    x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    camera.set('pos',vec(p));camera.set('xyaxes',vec([*x,*y]));camera.set('fovy','40')
    tree.write(path,encoding='utf-8',xml_declaration=True)
    return path,dict(objects=objects)

class NestingDemo(SortingDemo):
    def output_frame(self):
        self.rows.append(dict(time=float(self.data.time),object=self.active,phase=self.phase,qpos=self.data.qpos.tolist(),ncon=int(self.data.ncon)))
        if self.renderer is None:return
        from PIL import Image,ImageDraw
        self.renderer.update_scene(self.data,camera='overview')
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        im=Image.fromarray(self.renderer.render());draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,720,42),fill='#132233')
        draw.text((12,8),f'TAPERED CUPS | CONTACT PHYSICS | {self.active} | {self.phase}',fill='white')
        draw.text((12,25),'40mm cup / release then gravity nesting / no stops',fill='#8fe1d0')
        if self.writer:self.writer.append_data(np.asarray(im))
        if self.phase in ['READY','DONE']:im.save(self.args.out/(self.phase.lower()+'.png'))

    def nest(self,name,level):
        self.event('ALIGN_ABOVE_OPENING',level=level+1)
        self.move_pose([*STACK,.145],0,2.3,arc=True)
        p,_=self.pose(name);offset=p-self.tcp()
        release_z=.006+HEIGHT/2+.0008 if level==0 else self.pose(f'cup_{level}')[0][2]+PITCH
        target=np.r_[STACK,release_z]-offset
        self.event('INSERT_VERTICAL',target_tcp=target.tolist())
        self.move_pose(target,0,2.8)
        self.event('RELEASE');self.release(name)
        # Open fingers outward, then lift clear of both rims.
        self.move_pose([*self.tcp()[:2],.145],0,2.0)
        self.hold(.7)
        self.verify_nesting(level+1)
        self.result['completed'].append(dict(name=name,level=level+1))

    def verify_nesting(self,count):
        checks=[]
        for i in range(count):
            p,t=self.pose(f'cup_{i+1}')
            delta=None if i==0 else float(p[2]-self.pose(f'cup_{i}')[0][2])
            height_ok=abs(p[2]-(.006+HEIGHT/2))<.002 if i==0 else .003<delta<.016
            checks.append(dict(name=f'cup_{i+1}',position=p.tolist(),tilt_deg=t,rise_from_previous_mm=None if delta is None else delta*1000,success=bool(np.linalg.norm(p[:2]-STACK)<.003 and height_ok and t<8)))
        self.event('VERIFY_NESTING',checks=checks)
        if not all(c['success'] for c in checks):raise RuntimeError('Nesting position/tilt check failed')
        return checks

    def run(self):
        self.hold(1);self.event('READY');self.hold(.3)
        for i,name in enumerate(self.objects):
            self.active=name;self.pickup(name,'cup')
            self.pour_cup(name) # Always perform dry pouring, including empty cups.
            self.nest(name,i)
        self.event('SETTLE');self.hold(3)
        self.result['final_checks']=self.verify_nesting(len(self.objects))
        self.result.update(success=True,complete_task=True,height_mm=HEIGHT*1000,release_offset_mm=PITCH*1000)
        self.event('DONE');self.hold(.5)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cups',type=int,choices=[2,3],default=2)
    p.add_argument('--viewer',action='store_true');p.add_argument('--video',action='store_true')
    p.add_argument('--fps',type=int,default=30);p.add_argument('--fast',action='store_true')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'nesting_outputs')
    args=p.parse_args();args.out=args.out.resolve()
    path,manifest=scene(Path(__file__).resolve().parents[1],args.out,args.cups)
    demo=NestingDemo(mujoco.MjModel.from_xml_path(str(path)),args,manifest)
    if args.video:
        import imageio.v2 as imageio
        demo.renderer=mujoco.Renderer(demo.model,height=480,width=720)
        demo.writer=imageio.get_writer(args.out/'nesting.mp4',fps=args.fps,codec='libx264',ffmpeg_params=['-threads','2'])
    if args.viewer:
        from mujoco import viewer
        ctx=viewer.launch_passive(demo.model,demo.data)
    else:ctx=nullcontext(None)
    code=0;start=time.perf_counter()
    try:
        with ctx as v:
            demo.viewer=v
            if v:v.cam.lookat[:]=[.03,.13,.05];v.cam.distance=.55;v.cam.azimuth=55;v.cam.elevation=-35
            demo.run()
    except (Exception,KeyboardInterrupt) as e:
        demo.result.update(success=False,error=str(e));print('STOP',e,flush=True);code=1
    finally:
        demo.result.update(sim_seconds=float(demo.data.time),wall_seconds=time.perf_counter()-start)
        for fn,value in [('result.json',demo.result),('events.json',demo.events)]:
            (args.out/fn).write_text(json.dumps(value,indent=2))
        (args.out/'trajectory.jsonl').write_text('\n'.join(json.dumps(r) for r in demo.rows))
        np.savez_compressed(args.out/'final_state.npz',qpos=demo.data.qpos,qvel=demo.data.qvel,ctrl=demo.data.ctrl)
        if demo.writer:demo.writer.close()
        if demo.renderer:demo.renderer.close()
    print(json.dumps(demo.result,indent=2));return code
if __name__=='__main__':raise SystemExit(main())

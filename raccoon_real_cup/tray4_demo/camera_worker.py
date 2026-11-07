"""Runs in yolo_ws's venv. Owns the camera; never imports the robot SDK."""
import argparse
import time
import traceback
from pathlib import Path
from core import atomic_json,read_json,JsonLog,compatible_selections

def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);a=p.parse_args()
    folder=a.episode;cfg=read_json(folder/'config.json')
    pipe=None;started=False;log=None
    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
        from vision import load_model,detect,inventory,draw_preview
        atomic_json(folder/'camera_status.json',dict(state='loading_model',host_ns=time.monotonic_ns()))
        model=load_model(cfg)
        pipe=rs.pipeline();rc=rs.config()
        if cfg['camera_serial']:rc.enable_device(cfg['camera_serial'])
        rc.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
        profile=pipe.start(rc);started=True
        atomic_json(folder/'camera_info.json',dict(serial=profile.get_device().get_info(rs.camera_info.serial_number),
                    width=640,height=480,source_fps=30,saved_target_hz=10,color='RGB on PNG decode'))
        log=JsonLog(folder/'frames.jsonl');(folder/'rgb_frames').mkdir()
        previous=None;stable=0;running=False;armed=False;index=0;next_save=0;period=100_000_000
        win='TRAY4 - confirm in TERMINAL / Q cancel'
        last_number=None
        while True:
            control=read_json(folder/'camera_control.json')
            if control['mode']=='stop':break
            frame=pipe.wait_for_frames(1000).get_color_frame()
            now=time.monotonic_ns()
            if not frame:continue
            number=frame.get_frame_number()
            if number==last_number:continue
            last_number=number
            image=np.asanyarray(frame.get_data()).copy()
            if armed and control['mode']=='record':
                running=True;armed=False
                atomic_json(folder/'camera_status.json',dict(state='recording_raw_no_yolo',host_ns=now))
            if not running and not armed:
                dets=detect(model,image,cfg);selected=None;status=''
                try:selected=inventory(dets)
                except ValueError as exc:status=str(exc)
                stable=stable+1 if compatible_selections(previous,selected,cfg['stable_pixels']) else (1 if selected else 0)
                previous=selected
                ready=stable>=cfg['stable_frames']
                atomic_json(folder/'ready.json',dict(ready=ready,status=status,stable=stable,host_ns=time.monotonic_ns(),
                            selection=selected))
                if control['mode']=='freeze':
                    # WAIT_FOR_CONFIRMED_SCENE_V1
                    if not ready or not compatible_selections(selected,control['selection'],cfg['stable_pixels']):
                        cv2.imshow(
                            win,
                            draw_preview(
                                image, dets, selected,
                                'Waiting for confirmed scene...'
                            ),
                        )
                        key = cv2.waitKey(1) & 255
                        if key in (ord('q'), ord('Q'), 27):
                            raise KeyboardInterrupt('카메라 창에서 취소')
                        continue
                    selected.update(frozen_host_ns=now,confirmed_in_terminal=True)
                    atomic_json(folder/'selection.json',selected)
                    if not cv2.imwrite(str(folder/'initial_rgb.png'),image):raise OSError('초기 이미지 저장 실패')
                    armed=True;del model
                    atomic_json(folder/'camera_status.json',dict(state='waiting_for_joint_sampler',host_ns=now))
                else:
                    message='READY - press Enter in TERMINAL' if ready else 'Need plate1 tissue1 plastic1 glass1 - see terminal'
                    cv2.imshow(win,draw_preview(image,dets,selected,message))
            if armed:cv2.imshow(win,image)
            if running:
                # All saved images are unmodified camera pixels. No annotation is drawn here.
                if now>=next_save:
                    if next_save==0:next_save=now
                    skipped=max(0,(now-next_save)//period)
                    next_save+=(skipped+1)*period
                    file=f'rgb_frames/{index:06d}.png'
                    if not cv2.imwrite(str(folder/file),image):raise OSError('RGB 저장 실패')
                    row=dict(frame_index=index,file=file,host_arrival_ns=now,
                             camera_timestamp_ms=frame.get_timestamp(),camera_clock=str(frame.get_frame_timestamp_domain()),
                             camera_frame_number=number,save_deadlines_skipped=int(skipped))
                    log.write('camera_frame',**row);atomic_json(folder/'latest_frame.json',row);index+=1
                cv2.imshow(win,image)
            key=cv2.waitKey(1)&255
            if key in (ord('q'),ord('Q'),27):raise KeyboardInterrupt('카메라 창에서 취소')
        atomic_json(folder/'camera_status.json',dict(state='stopped',frames=index,host_ns=time.monotonic_ns()))
    except BaseException as exc:
        atomic_json(folder/'camera_error.json',dict(error=repr(exc),traceback=traceback.format_exc(),host_ns=time.monotonic_ns()))
        raise
    finally:
        if started:pipe.stop()
        if log:log.close()
        try:
            import cv2
            cv2.destroyAllWindows()
        except ImportError:pass

if __name__=='__main__':main()

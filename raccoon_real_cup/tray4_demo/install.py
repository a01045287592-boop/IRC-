"""Install in a new directory; do not change existing motion/config files."""
import argparse
import shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path.home()/'Raccoonbot_Cafe_Clean');a=p.parse_args()
    source=Path(__file__).resolve().parent;dest=a.project.expanduser()/'raccoon_real_cup/tray4_demo'
    if not (dest.parent/'cup_pick_real.py').is_file():raise SystemExit('프로젝트의 raccoon_real_cup/cup_pick_real.py를 찾지 못했습니다.')
    if dest.exists():raise SystemExit('이미 설치된 폴더입니다: '+str(dest)+'\n기존 파일을 덮어쓰지 않았습니다.')
    shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store'))
    print('설치 완료:',dest)
    print('기존 동작 파일/좌표 설정은 변경하지 않았습니다.')
    print('실행 진입점:',dest/'tray4.py')

if __name__=='__main__':main()

"""Create a lossless FFV1 review video from the ACT-input PNGs, without redrawing."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('episode',type=Path)
a=p.parse_args()
rows=[json.loads(s) for s in (a.episode/'observations.jsonl').read_text().splitlines() if s.strip()]
if not rows:raise SystemExit('기록된 입력 이미지가 없습니다.')
output=a.episode/'review_lossless.avi'
if output.exists():raise SystemExit('review_lossless.avi가 이미 있습니다. 덮어쓰지 않았습니다.')
first=cv2.imread(str(a.episode/rows[0]['image']))
if first is None:raise SystemExit('첫 PNG를 읽지 못했습니다.')
h,w=first.shape[:2]
writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'FFV1'),10.,(w,h))
if not writer.isOpened():raise SystemExit('FFV1 인코더를 열지 못했습니다.')
try:
    for r in rows:
        img=cv2.imread(str(a.episode/r['image']))
        if img is None or img.shape!=first.shape:raise ValueError(r['image'])
        writer.write(img)
finally:
    writer.release()
cap=cv2.VideoCapture(str(output));ok,decoded=cap.read();cap.release()
if not ok or not np.array_equal(first,decoded):
    raise SystemExit('첫 프레임 무손실 검증 실패. 학습 입력은 PNG를 사용하세요.')
print('영상:',output)
print('첫 프레임 PNG와 디코딩 픽셀 일치. 10fps 검토용이며 타이밍 원본은 observations.jsonl입니다.')

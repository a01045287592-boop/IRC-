import json
import sys
from pathlib import Path
from tray_core import validate_model_names
from ultralytics import YOLO

cfg=json.loads(Path(sys.argv[1]).read_text())
model=YOLO(str(Path(cfg['model']['weights']).expanduser()))
print('모델 클래스:',model.names)
validate_model_names(model.names,cfg['model']['class_names'])
if model.task != 'segment':
    raise SystemExit('segmentation 모델이 필요합니다.')
print('MODEL PASS: bowl/tissue/plastic_cup/glass_cup/tray 이름 매핑 확인')

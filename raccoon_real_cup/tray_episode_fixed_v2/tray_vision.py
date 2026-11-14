"""2-D tray selection and per-object, pre-grasp-only rendering."""
import math
import cv2
import numpy as np


def center(mask):
    ys, xs = np.nonzero(mask)
    return [float(xs.mean()), float(ys.mean())] if len(xs) else None


def bottom_center(mask):
    ys = np.flatnonzero(mask.any(axis=1))
    if not len(ys):
        raise ValueError('빈 트레이 마스크')
    band = max(int((ys[-1] - ys[0] + 1) * .1), 1)
    xs = np.flatnonzero(mask[max(ys[-1] - band, 0):ys[-1]+1].any(axis=0))
    x = int(xs[0] + xs[-1]) // 2
    col = np.flatnonzero(mask[:, x])
    return [x, int(col[-1]) if len(col) else int(ys[-1])]


def tray_footprint(mask):
    # Same occlusion-hull rule as the user's corrected mark_on_tray.
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None or len(points) < 3:
        raise ValueError('트레이 마스크가 너무 작습니다.')
    hull = cv2.convexHull(points)
    if cv2.contourArea(hull) <= 0:
        raise ValueError('트레이 면적이 0입니다.')
    result = np.zeros_like(mask, np.uint8)
    cv2.fillConvexPoly(result, hull, 1)
    return result.astype(bool)


def on_tray(mask, area):
    ys = np.flatnonzero(mask.any(axis=1))
    if not len(ys):
        return False
    base = mask.copy()
    base[:ys[-1] - int((ys[-1] - ys[0] + 1) * .3)] = False
    return bool(base.sum() and (base & area).sum() / base.sum() >= .6)


def cup_material(image, mask, cfg):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = (hsv[:,:,1] <= 60) & (hsv[:,:,2] >= 170)
    yy = np.nonzero(mask)[0]
    if not len(yy): return 'uncertain_cup', None
    region = cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)) > 0
    height = yy.max()-yy.min()+1
    region[:int(yy.min()+height*.2)] = False
    region[int(yy.min()+height*.85)+1:] = False
    if region.sum() < 20: return 'uncertain_cup', None
    ratio = float(white[region].mean())
    kind = ('glass' if ratio <= cfg['material']['glass_max'] else
            'plastic' if ratio >= cfg['material']['plastic_min'] else 'uncertain_cup')
    return kind, ratio


def detect(model, image, cfg):
    names = cfg['model']['class_names']; reverse = {v: k for k, v in names.items()}
    r = model.predict(image, conf=cfg['vision']['confidence'], imgsz=640,
                      retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return []
    out = []; h, w = image.shape[:2]
    for b, m in zip(r.boxes, r.masks.data.cpu().numpy()):
        cls = model.names[int(b.cls[0])]
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m = m > .5
        if not m.any():
            continue
        d = dict(kind=reverse.get(cls, 'unsupported:' + cls), mask=m,
                 box=list(map(float, b.xyxy[0].tolist())), center=center(m),
                 confidence=float(b.conf[0]))
        if d['kind'] == 'cup':
            d['kind'], d['white_ratio'] = cup_material(image, m, cfg)
        if d['kind'] == 'tray' and d['confidence'] < .5:
            continue
        out.append(d)
    # Same-class overlapping duplicates are filtered; conflicting classes are
    # retained so inventory validation cannot silently turn cup types into one another.
    keep = []
    for d in sorted(out, key=lambda d: -d['confidence']):
        if any(d['kind'] == k['kind'] and
               (d['mask'] & k['mask']).sum() >= .7 * min(d['mask'].sum(), k['mask'].sum())
               for k in keep):
            continue
        keep.append(d)
    return keep


def front_inventory(dets):
    trays = [d for d in dets if d['kind'] == 'tray']
    if not trays:
        raise ValueError('앞 트레이 미검출')
    tray = max(trays, key=lambda d: d['box'][3])
    area = tray_footprint(tray['mask'])
    objects = [d for d in dets if d['kind'] != 'tray' and on_tray(d['mask'], area)]
    return objects, bottom_center(tray['mask'])


def match_target(dets, target, gate_px):
    # Objects have not been picked yet: associate to their fixed start position,
    # not to mutable YOLO instance numbers or whichever object is closest now.
    candidates = sorted((d for d in dets if d['kind'] == target['kind'] and
                         math.dist(d['center'], target['center']) <= gate_px),
                        key=lambda d: math.dist(d['center'], target['center']))
    if not candidates:
        return None
    if len(candidates) > 1:
        d0 = math.dist(candidates[0]['center'], target['center'])
        d1 = math.dist(candidates[1]['center'], target['center'])
        if d1 - d0 < 5:
            return None
    return candidates[0]


def outline(image, mask):
    vis = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return vis
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) >= 9:
        contour = cv2.approxPolyDP(contour, 1., True)
        cv2.drawContours(vis, [contour], -1, (0, 0, 255), 2, cv2.LINE_8)
    return vis


def render_recorded(image, frame, selection, detector, gate_px):
    """The SAME output is written to PNG and shown. Raw phases never call detector."""
    state = frame['state']
    if state['phase'] != 'before_grasp':
        return image.copy(), False, False
    target = next(o for o in selection['objects'] if o['id'] == state['object_id'])
    matched = match_target(detector(image), target, gate_px)
    if matched is None:
        return image.copy(), True, False
    return outline(image, matched['mask']), True, True

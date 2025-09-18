import cv2, numpy as np
from core.helpers import odd, nms_merge
from detector.masks import make_hsv_mask
from detector.shapes import detect_shape_in_balloon, SHAPE_PARAMS

def detect_balloons(frame_bgr, preset, sens, min_area=1400,
                    k_base=11, require_shape=None):
    """
    Renk maskesi + morfoloji + dairesellik/solidlik + (opsiyonel) balon içi siyah şekil
    Döner: (dets, mask, proc)
      dets: [(ox,oy,orad), ...]  # orijinale ölçeklenmiş
      mask: binary maske (proc boyutunda)
      proc: küçültülmüş BGR (debug)
    """
    H, W = frame_bgr.shape[:2]

    PROC_W = 480
    scale  = PROC_W / float(W)
    proc = cv2.resize(frame_bgr, (PROC_W, int(round(H * scale))), interpolation=cv2.INTER_LINEAR)

    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (odd(5), odd(5)), 0)

    mask_color, _ = make_hsv_mask(hsv, preset, sens)

    k = odd(max(3, int(round(k_base + sens * 0.10))))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN,  ker)
    mask = cv2.morphologyEx(mask,       cv2.MORPH_CLOSE, ker)
    mask = cv2.medianBlur(mask, 5)

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    minA_scaled = min_area * (scale ** 2)

    for c in cnts:
        A = cv2.contourArea(c)
        if A < minA_scaled: continue

        P = cv2.arcLength(c, True) + 1e-6
        circ = 4*np.pi*A/(P*P)
        if circ < 0.75: continue

        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull) + 1e-6
        sol = A/ha
        if sol < 0.88: continue

        if require_shape is not None:
            want = require_shape if require_shape in ("circle", "triangle", "square") else "any"
            ok, shp, conf, dbg = detect_shape_in_balloon(proc, c, want_shape=want)  # 4 değer
            if not ok:
                continue
            # (istersen burada conf/dbg'yi kullanabilirsin; şimdilik gerek yok)
        else:
            shp = None

        (cx_, cy_), r = cv2.minEnclosingCircle(c)
        r = int(r)
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])

        ox = int(round(cx / scale)); oy = int(round(cy / scale))
        orad = int(round(r / scale))
        dets.append((ox, oy, orad))

    dets = nms_merge(dets, dist_thr=22)
    return dets, mask, proc

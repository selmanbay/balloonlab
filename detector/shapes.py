import cv2, numpy as np

def classify_shape(c):
    A = cv2.contourArea(c)
    if A < 1: return "unknown"
    P = cv2.arcLength(c, True) + 1e-6
    approx = cv2.approxPolyDP(c, 0.03 * P, True)
    if len(approx) == 3: return "triangle"
    if len(approx) == 4: return "square"
    circ = 4 * np.pi * A / (P * P)
    if circ > 0.78: return "circle"
    return "unknown"

def find_black_shape_in_balloon(proc_bgr, contour, want_shape="any"):
    """
    Balon konturunun içini maskeleyip siyah şekil arar.
    want_shape: 'any' | 'circle' | 'triangle' | 'square'
    Döner: (bool found, shape_name_or_None)
    """
    x,y,w,h = cv2.boundingRect(contour)
    if w*h <= 0: return False, None

    roi = proc_bgr[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    black = cv2.inRange(hsv, (0, 0, 0), (179, 170, 70))
    black = cv2.medianBlur(black, 5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, k)

    mask_ball = np.zeros((h, w), np.uint8)
    c_shift = contour - [x, y]
    cv2.drawContours(mask_ball, [c_shift], -1, 255, thickness=-1)

    black_in = cv2.bitwise_and(black, mask_ball)

    cnts,_ = cv2.findContours(black_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return False, None

    ball_area = cv2.contourArea(contour) + 1e-6
    best = None
    for c in cnts:
        A = cv2.contourArea(c)
        if A < 0.01*ball_area or A > 0.65*ball_area:
            continue
        shp = classify_shape(c)
        if shp == "unknown": continue
        if want_shape != "any" and shp != want_shape:
            continue
        if best is None or A > best[0]:
            best = (A, shp)
    if best is None: return False, None
    return True, best[1]

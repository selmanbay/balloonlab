import cv2, os

def ensure_window(name, w=960, h=540):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)

def build_controls():
    ctrl = "Controls"
    ensure_window(ctrl, 480, 200)
    cv2.createTrackbar("Color", ctrl, 0, 4,  lambda v: None)   # 0..4 (Blue..White)
    cv2.createTrackbar("Sensitivity", ctrl, 35, 100, lambda v: None)
    cv2.createTrackbar("MinArea", ctrl, 1400, 30000, lambda v: None)
    cv2.createTrackbar("ShowMask", ctrl, 0, 1, lambda v: None)
    cv2.createTrackbar("Shape", ctrl, 0, 3, lambda v: None)    # 0:Any 1:Circle 2:Triangle 3:Square
    cv2.createTrackbar("Mode",  ctrl, 0, 1, lambda v: None)    # 0:ColorOnly 1:Color+Shape
    return ctrl

def read_controls(ctrl, color_list, shape_names):
    pidx  = cv2.getTrackbarPos("Color", ctrl)
    sens  = cv2.getTrackbarPos("Sensitivity", ctrl)
    minA  = cv2.getTrackbarPos("MinArea", ctrl)
    show_mask = cv2.getTrackbarPos("ShowMask", ctrl) == 1
    shidx = cv2.getTrackbarPos("Shape", ctrl)
    mode  = cv2.getTrackbarPos("Mode",  ctrl)
    color_name = color_list[pidx % len(color_list)]
    shape_name = shape_names[shidx] if shidx < len(shape_names) else "Any"
    return color_name, sens, minA, show_mask, shape_name, mode

def save_frame(out_dir, img, prefix="frame", idx=0):
    if not out_dir: return idx
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{prefix}_{idx:05d}.jpg")
    cv2.imwrite(fname, img)
    return idx + 1
def build_shape_params():
    win = "ShapeParams"
    ensure_window(win, 360, 260)
    # isimler SHAPE_PARAMS anahtarlarıyla eşleşiyor (yaklaşık)
    cv2.createTrackbar("canny_lo", win, 60, 400, lambda v: None)
    cv2.createTrackbar("canny_hi", win, 140, 400, lambda v: None)
    cv2.createTrackbar("approx_eps_x100", win, 4, 20, lambda v: None)   # 0.04 -> 4
    cv2.createTrackbar("close_k", win, 3, 9, lambda v: None)            # tek sayı yapacağız
    cv2.createTrackbar("dilate_k", win, 1, 5, lambda v: None)
    return win

def read_shape_params(win, params_dict):
    canny_lo = cv2.getTrackbarPos("canny_lo", win)
    canny_hi = cv2.getTrackbarPos("canny_hi", win)
    eps_x100 = cv2.getTrackbarPos("approx_eps_x100", win)
    close_k  = cv2.getTrackbarPos("close_k", win)
    dilate_k = cv2.getTrackbarPos("dilate_k", win)
    if close_k % 2 == 0: close_k += 1
    params_dict.update({
        "canny_lo": canny_lo,
        "canny_hi": canny_hi,
        "approx_eps": max(0.01, eps_x100/100.0),
        "close_k": max(1, close_k),
        "dilate_k": max(0, dilate_k),
    })
def build_inner_shape_controls():
    win = "InnerShape"
    ensure_window(win, 420, 220)
    # 0: auto_dark, 1: auto_contrast, 2: color
    cv2.createTrackbar("InnerMode", win, 0, 2, lambda v: None)
    # InnerColor index: 0..5  (Red,Green,Blue,Yellow,Black,White)
    cv2.createTrackbar("InnerColor", win, 4, 5, lambda v: None)  # default Black
    cv2.createTrackbar("InnerSens", win, 40, 100, lambda v: None)
    cv2.createTrackbar("ContrastThr", win, 20, 60, lambda v: None)  # ΔE threshold
    # Morfoloji (çizgi kalınlığı)
    cv2.createTrackbar("CloseK", win, 3, 9, lambda v: None)
    cv2.createTrackbar("DilateK", win, 1, 5, lambda v: None)
    return win

def read_inner_shape_controls(win, SHAPE_PARAMS):
    mode_i = cv2.getTrackbarPos("InnerMode", win)
    color_i = cv2.getTrackbarPos("InnerColor", win)
    sens    = cv2.getTrackbarPos("InnerSens", win)
    thr     = cv2.getTrackbarPos("ContrastThr", win)
    close_k = cv2.getTrackbarPos("CloseK", win)
    dil_k   = cv2.getTrackbarPos("DilateK", win)

    mode = ["auto_dark","auto_contrast","color"][mode_i]
    colors = ["Red","Green","Blue","Yellow","Black","White"]
    color_name = colors[color_i]

    if close_k % 2 == 0: close_k += 1
    SHAPE_PARAMS.update({
        "inner_mode": mode,
        "inner_color": color_name,
        "inner_sens": sens,
        "contrast_thr": max(5, thr),
        "close_k": max(1, close_k),
        "dilate_k": max(0, dil_k),
    })

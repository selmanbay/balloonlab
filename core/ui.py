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

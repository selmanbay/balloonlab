import cv2, numpy as np
from core.helpers import odd, lerp

COLOR_PRESETS = [
    {"name":"Blue",   "type":"single", "h":115, "w":15, "s_min":60, "v_min":50},
    {"name":"Red",    "type":"split",  "h":None,"w":10, "s_min":65, "v_min":55},
    {"name":"Green",  "type":"single", "h":60,  "w":18, "s_min":55, "v_min":45},
    {"name":"Yellow", "type":"single", "h":28,  "w":12, "s_min":70, "v_min":60},
    {"name":"White",  "type":"white",  "h":None,"w":0,  "s_min":35, "v_min":180}
]

def red_mask_hsv(hsv, s0, v0, w):
    w = max(5, w)
    low1  = np.array([0,   s0, v0], np.uint8); high1 = np.array([min(179, w), 255, 255], np.uint8)
    low2  = np.array([max(0, 179 - w), s0, v0], np.uint8); high2 = np.array([179, 255, 255], np.uint8)
    return cv2.inRange(hsv, low1, high1) | cv2.inRange(hsv, low2, high2)

def white_mask_hsv(hsv, s_max, v_min):
    low  = np.array([0, 0, v_min], np.uint8)
    high = np.array([179, s_max, 255], np.uint8)
    return cv2.inRange(hsv, low, high)

def make_hsv_mask(hsv, preset, sens):
    """
    sens: 0..100 (t arttıkça tolerans genişler)
    Döner: (mask, (w,s0,v0))  # white için (0,0,0)
    """
    t = sens / 100.0
    if preset["type"] == "white":
        s_max = int(round(lerp(preset["s_min"], min(120, preset["s_min"] + 40), t)))
        v_min = int(round(lerp(preset["v_min"], max(140, preset["v_min"] - 30), t)))
        mask = white_mask_hsv(hsv, s_max, v_min)
        return mask, (0,0,0)
    else:
        w  = int(round(lerp(preset["w"], preset["w"] + 18, t)))
        s0 = int(round(lerp(preset["s_min"], max(20, preset["s_min"] - 30), t)))
        v0 = int(round(lerp(preset["v_min"], max(20, preset["v_min"] - 35), t)))
        if preset["type"] == "single":
            h = preset["h"]
            low  = np.array([max(0, h - w), s0, v0], np.uint8)
            high = np.array([min(179, h + w), 255, 255], np.uint8)
            mask = cv2.inRange(hsv, low, high)
        else:
            mask = red_mask_hsv(hsv, s0, v0, w)
        return mask, (w, s0, v0)

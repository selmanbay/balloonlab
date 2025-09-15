import cv2, json, time, argparse
import numpy as np

from core.helpers import ema_update, window_exists
from core.ui import ensure_window, build_controls, read_controls, save_frame
from core.telemetry import TelemetryWriter
from detector.masks import COLOR_PRESETS
from detector.detection import detect_balloons

COLOR_LIST  = [p["name"] for p in COLOR_PRESETS]
SHAPE_NAMES = ["Any", "Circle", "Triangle", "Square"]

def load_config(path):
    # BOM’lu/ BOM’suz UTF-8’i sorunsuz okur
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def camera_open(width=640, height=480, fps=30):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    return cap

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="configs/pc-dev.json")
    ap.add_argument("--record", type=str, help="Çıktıları kaydet (klasör)")
    ap.add_argument("--video", type=str, help="Video dosyası ile test")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.profile)

    color_name   = cfg["color"]
    sensitivity  = cfg["sensitivity"]
    min_area     = cfg["min_area"]
    require_shape = cfg.get("shape", "any")
    mode_name    = cfg.get("mode", "color_only")
    show_mask    = bool(cfg.get("show_mask", False))

    cap = camera_open() if not args.video else cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Kamera/Video açılamadı."); return

    win = "Balloon Detector (Modular)"
    ensure_window(win, 960, 540)
    ctrl = build_controls()

    writer = TelemetryWriter(args.record)

    tracks = []
    save_idx = 0
    prev = time.time()
    cv2.setUseOptimized(True)

    while True:
        ok, frame = cap.read()
        if not ok: break

        color_name, sensitivity, min_area, show_mask, shape_name, mode = read_controls(ctrl, COLOR_LIST, SHAPE_NAMES)
        require_shape = (shape_name.lower() if mode==1 else None)

        pidx = [p["name"] for p in COLOR_PRESETS].index(color_name)
        preset = COLOR_PRESETS[pidx]

        dets, mask, proc = detect_balloons(
            frame, preset, sens=sensitivity, min_area=min_area,
            k_base=cfg["morph_k_base"], require_shape=require_shape
        )

        tracks = ema_update(tracks, dets, alpha=0.35, match_thr=40)

        out = frame.copy()
        for (x, y, r, last) in tracks:
            cv2.circle(out, (x, y), 6, (0, 255, 255), -1)
            if r > 0: cv2.circle(out, (x, y), r, (0, 200, 255), 2)
            cv2.putText(out, f"({x},{y})", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        now = time.time(); fps = 1.0 / max(1e-3, (now - prev)); prev = now
        hud = f"{color_name} | Mode:{'C+S' if mode==1 else 'Color'} | Shape:{shape_name} | Sens:{sensitivity} | MinA:{min_area} | FPS:{fps:.1f}"
        cv2.putText(out, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (50, 255, 50), 2)

        cv2.imshow(win, out)

        if show_mask:
            if not window_exists("Mask"):
                cv2.namedWindow("Mask", cv2.WINDOW_NORMAL); cv2.resizeWindow("Mask", 480, 270)
            vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            vis_proc = cv2.resize(proc, (vis_mask.shape[1], vis_mask.shape[0]))
            cv2.imshow("Mask", np.hstack([vis_proc, vis_mask]))
        else:
            if window_exists("Mask"):
                try: cv2.destroyWindow("Mask")
                except cv2.error:
                    pass

        if args.record:
            if save_idx % 10 == 0:
                save_idx = save_frame(args.record, out, prefix="frame", idx=save_idx)
            else:
                save_idx += 1
            writer.write({
                "mode": "color_shape" if mode==1 else "color_only",
                "fps": round(fps,1),
                "params": {"color": color_name, "sensitivity": sensitivity, "min_area": min_area, "shape": shape_name},
                "dets": [{"x":x,"y":y,"r":r} for (x,y,r) in dets]
            })

        k = cv2.waitKey(1) & 0xFF
        if k == 27: break
        elif k in (ord('c'), ord('C')):
            pidx = (pidx + 1) % len(COLOR_PRESETS)
            cv2.setTrackbarPos("Color", "Controls", pidx)
        elif k in (ord('m'), ord('M')):
            show_mask = not show_mask
            cv2.setTrackbarPos("ShowMask", "Controls", 1 if show_mask else 0)
        elif k in (ord('s'), ord('S')) and args.record:
            save_idx = save_frame(args.record, out, prefix="snap", idx=save_idx)

    cap.release()
    writer.close()
    if window_exists("Mask"):
        try: cv2.destroyWindow("Mask")
        except cv2.error:
            pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2, numpy as np

# ---- Parametre varsayılanları (UI'dan güncellenebilir) ----
SHAPE_PARAMS = {
    "canny_lo": 60,        # Canny düşük eşik
    "canny_hi": 140,       # Canny yüksek eşik
    "min_side_px": 10,     # çok küçük üçgen/kare kenarlarını elemek için
    "approx_eps": 0.04,    # approxPolyDP oransal hata
    "min_frac": 0.003,     # ROI alanına göre minimum şekil alanı oranı
    "max_frac": 0.90,      # ROI alanına göre maksimum şekil alanı oranı
    "dilate_k": 1,         # çizgi kalınlaştırma miktarı (px)
    "close_k": 3           # morfolojik close kernel boyu (tek sayı)
}

def classify_shape_by_poly(c, eps_ratio=0.04):
    """Kontura approxPolyDP uygula, çokgen köşe sayısına göre sınıf döndür."""
    A = cv2.contourArea(c)
    if A < 1: return "unknown"
    P = cv2.arcLength(c, True) + 1e-6
    approx = cv2.approxPolyDP(c, eps_ratio * P, True)
    n = len(approx)
    if n == 3: return "triangle"
    if n == 4: return "square"   # kare/dikdörtgen birlikte
    circ = (4*np.pi*A)/(P*P)
    if circ > 0.78: return "circle"
    return "unknown"

def _morph_strengthen(bin_img, close_k=3, dilate_k=1):
    """İnce çizgileri güçlendir: close + dilate."""
    if close_k < 1: close_k = 1
    if close_k % 2 == 0: close_k += 1
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    out = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k1)
    if dilate_k > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
        out = cv2.dilate(out, k2, iterations=1)
    return out

def _dark_mask_roi(roi_bgr):
    """ROI içinde V+OTSU ile koyu maskesi, S düşük bölgelerle kesi; çizgileri güçlendir."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    Vb = cv2.GaussianBlur(hsv[:,:,2], (5,5), 0)
    _, inv = cv2.threshold(Vb, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # karanlık beyaz
    _, s_inv = cv2.threshold(hsv[:,:,1], 40, 255, cv2.THRESH_BINARY_INV)         # S düşük bölgeler beyaz
    dark = cv2.bitwise_and(inv, s_inv)
    dark = cv2.medianBlur(dark, 3)
    dark = _morph_strengthen(dark, close_k=SHAPE_PARAMS["close_k"], dilate_k=SHAPE_PARAMS["dilate_k"])
    return dark

def _edge_mask_roi(roi_bgr):
    """ROI içinde Canny kenar maskesi; kenar kalınlaştırma."""
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)
    e = cv2.Canny(g, threshold1=SHAPE_PARAMS["canny_lo"], threshold2=SHAPE_PARAMS["canny_hi"])
    e = _morph_strengthen(e, close_k=SHAPE_PARAMS["close_k"], dilate_k=SHAPE_PARAMS["dilate_k"])
    return e

def _mask_inside_balloon(mask, contour, xy):
    """Balon konturunun içini doldurup mask ile kesiştir."""
    x,y = xy
    h, w = mask.shape[:2]
    mask_ball = np.zeros((h, w), np.uint8)
    c_shift = contour - [x, y]
    cv2.drawContours(mask_ball, [c_shift], -1, 255, thickness=-1)
    return cv2.bitwise_and(mask, mask_ball)

def _best_shape_from_contours(cnts, ball_area, want_shape="any", eps_ratio=0.04, min_frac=0.003, max_frac=0.90):
    best = None
    for c in cnts:
        A = cv2.contourArea(c)
        if A < min_frac*ball_area or A > max_frac*ball_area:
            continue
        shp = classify_shape_by_poly(c, eps_ratio=eps_ratio)
        if shp == "unknown":
            continue
        if want_shape != "any" and shp != want_shape:
            continue
        if best is None or A > best[0]:
            best = (A, shp)
    return best

def detect_shape_in_balloon(proc_bgr, contour, want_shape="any"):
    """
    İki yollu tespit + oy birleştirme:
      (1) 'Koyu bölge' maskesi konturları → approxPolyDP
      (2) Canny kenar konturları → approxPolyDP
    Dönüş: (found:bool, shape:str|None, conf:float, debug:dict)
    """
    x,y,w,h = cv2.boundingRect(contour)
    if w*h <= 0:
        return False, None, 0.0, {}
    roi = proc_bgr[y:y+h, x:x+w]
    ball_area = cv2.contourArea(contour) + 1e-6

    # 1) Dark (region-based)
    dark = _dark_mask_roi(roi)
    dark_in = _mask_inside_balloon(dark, contour, (x,y))
    cnts_d,_ = cv2.findContours(dark_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_d = _best_shape_from_contours(
        cnts_d, ball_area, want_shape=want_shape,
        eps_ratio=SHAPE_PARAMS["approx_eps"],
        min_frac=SHAPE_PARAMS["min_frac"], max_frac=SHAPE_PARAMS["max_frac"]
    )

    # 2) Edge (Canny-based)
    edge = _edge_mask_roi(roi)
    edge_in = _mask_inside_balloon(edge, contour, (x,y))
    cnts_e,_ = cv2.findContours(edge_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_e = _best_shape_from_contours(
        cnts_e, ball_area, want_shape=want_shape,
        eps_ratio=SHAPE_PARAMS["approx_eps"],
        min_frac=SHAPE_PARAMS["min_frac"], max_frac=SHAPE_PARAMS["max_frac"]
    )

    # Oy birleştirme
    votes = {}
    if best_d: votes[best_d[1]] = votes.get(best_d[1], 0) + 1
    if best_e: votes[best_e[1]] = votes.get(best_e[1], 0) + 1

    if not votes:
        return False, None, 0.0, {"dark_in": dark_in, "edge_in": edge_in}

    # En çok oyu alan + baseline güven (1 yol = 0.6, 2 yol = 0.9)
    shape = max(votes.items(), key=lambda kv: kv[1])[0]
    conf = 0.6 if votes[shape] == 1 else 0.9
    # 'want_shape' kısıtı varsa ona göre final kararı ver
    if want_shape != "any" and shape != want_shape:
        # farklı oy geldiyse, yine de exact eşleşme var mı bak
        if best_d and best_d[1] == want_shape:
            shape, conf = want_shape, conf
        elif best_e and best_e[1] == want_shape:
            shape, conf = want_shape, conf
        else:
            return False, None, 0.0, {"dark_in": dark_in, "edge_in": edge_in}

    return True, shape, conf, {"dark_in": dark_in, "edge_in": edge_in}

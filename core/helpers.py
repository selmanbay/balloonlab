import cv2
from math import hypot

def odd(n: int) -> int:
    """Tek sayıya zorla (çekirdek boyutları için)."""
    return n if n % 2 == 1 else n + 1

def lerp(a: float, b: float, t: float) -> float:
    """Doğrusal ara değer (yumuşatma)."""
    return a + (b - a) * t

def nms_merge(points, dist_thr=22):
    """
    Yakın merkezleri tek tespitte birleştir.
    points: [(x,y,r), ...]
    """
    merged = []
    for (x, y, r) in points:
        placed = False
        for i, (ux, uy, ur) in enumerate(merged):
            if hypot(x - ux, y - uy) <= dist_thr:
                merged[i] = ((ux + x) // 2, (uy + y) // 2, max(ur, r))
                placed = True
                break
        if not placed:
            merged.append((x, y, r))
    return merged

def ema_update(tracks, detections, alpha=0.35, match_thr=40):
    """
    Basit iz takibi + yumuşatma.
    tracks: [(tx,ty,tr,last), ...]
    detections: [(dx,dy,dr), ...]
    """
    new_tracks = []
    used = [False] * len(detections)
    for (tx, ty, tr, last) in tracks:
        best_i, best_d = -1, 1e9
        for i, (dx, dy, dr) in enumerate(detections):
            if used[i]: continue
            d = hypot(dx - tx, dy - ty)
            if d < best_d: best_d, best_i = d, i
        if best_i != -1 and best_d <= match_thr:
            dx, dy, dr = detections[best_i]; used[best_i] = True
            sx = int(lerp(tx, dx, alpha)); sy = int(lerp(ty, dy, alpha)); sr = int(max(tr, dr))
            new_tracks.append((sx, sy, sr, 0))
        else:
            if last < 5: new_tracks.append((tx, ty, tr, last + 1))
    for i, (dx, dy, dr) in enumerate(detections):
        if not used[i]: new_tracks.append((dx, dy, dr, 0))
    return new_tracks

def window_exists(name: str) -> bool:
    """Pencere güvenli kontrol."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 0
    except cv2.error:
        return False

# -*- coding: utf-8 -*-
"""
PySide6 ile tek pencereli, kaygan bir arayüz:
- Sol: video önizleme
- Sağ: sekmeli kontrol paneli (Color/Detection & InnerShape)
- Alt: canlı log paneli

Projendeki mevcut modülleri kullanır:
 - detector.detection.detect_balloons
 - detector.masks.COLOR_PRESETS
 - detector.shapes.SHAPE_PARAMS
 - core.helpers.ema_update

Kurulum:
    pip install PySide6 opencv-python numpy
Çalıştırma (proje kökünden):
    python apps/desktop_ui.py
"""
import sys, os, time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np

# Proje içi importlar
from core.helpers import ema_update
from detector.masks import COLOR_PRESETS
from detector.detection import detect_balloons
from detector.shapes import SHAPE_PARAMS

APP_TITLE = "Balloon Lab – Desktop UI"
COLOR_LIST = [p["name"] for p in COLOR_PRESETS]
SHAPE_LIST = ["Any", "Circle", "Triangle", "Square"]
INNER_COLORS = ["Red","Green","Blue","Yellow","Black","White"]
INNER_MODES = ["auto_dark","auto_contrast","color"]

class VideoWorker(QtCore.QObject):
    """QTimer ile kameradan frame alır, işleyip sinyal yollar."""
    frame_ready = QtCore.Signal(np.ndarray, dict)  # (bgr_frame, meta)

    def __init__(self, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(0)  # olabildiğince hızlı
        self._timer.timeout.connect(self._on_tick)
        self.running = False
        self.prev_ts = time.time()

        # algılama durumları
        self.selected_color = COLOR_LIST[0]
        self.sensitivity = 35
        self.min_area = 1400
        self.mode = 0  # 0: ColorOnly, 1: Color+Shape
        self.shape = "Any"

        # takip için
        self.tracks = []

    def start(self):
        self.running = True
        self._timer.start()

    def stop(self):
        self.running = False
        self._timer.stop()
        if self.cap.isOpened():
            self.cap.release()

    # UI'dan parametre güncelleme
    def update_params(self, *, selected_color, sensitivity, min_area, mode, shape):
        self.selected_color = selected_color
        self.sensitivity = int(sensitivity)
        self.min_area = int(min_area)
        self.mode = int(mode)
        self.shape = shape

    def _on_tick(self):
        if not self.running: return
        ok, frame = self.cap.read()
        if not ok:
            return
        now = time.time();
        fps = 1.0 / max(1e-3, (now - self.prev_ts));
        self.prev_ts = now

        # preset seç
        try:
            pidx = COLOR_LIST.index(self.selected_color)
        except ValueError:
            pidx = 0
        preset = COLOR_PRESETS[pidx]

        require_shape = None
        if self.mode == 1:
            require_shape = self.shape.lower() if self.shape != "Any" else "any"

        dets, mask, proc = detect_balloons(
            frame, preset, sens=self.sensitivity, min_area=self.min_area,
            k_base=11, require_shape=require_shape
        )
        self.tracks = ema_update(self.tracks, dets, alpha=0.35, match_thr=40)

        # overlay çiz
        out = frame.copy()
        for (x, y, r, last) in self.tracks:
            cv2.circle(out, (x, y), 6, (0, 255, 255), -1)
            if r > 0: cv2.circle(out, (x, y), r, (0, 200, 255), 2)
            cv2.putText(out, f"({x},{y})", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        hud = f"{self.selected_color} | Mode:{'C+S' if self.mode==1 else 'Color'} | Shape:{self.shape} | Sens:{self.sensitivity} | MinA:{self.min_area} | FPS:{fps:.1f}"
        cv2.putText(out, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (50, 255, 50), 2)

        meta = {
            "fps": fps,
            "dets": dets,
            "color": self.selected_color,
            "mode": "color_shape" if self.mode==1 else "color_only",
            "shape": self.shape,
        }
        self.frame_ready.emit(out, meta)

class VideoCanvas(QtWidgets.QLabel):
    """QLabel üzerinde OpenCV BGR görüntüsünü gösterir."""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 450)
        self.setScaledContents(True)
        self.setStyleSheet("background:#111;border-radius:8px")

    @staticmethod
    def _bgr2qimage(bgr: np.ndarray) -> QtGui.QImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

    def show_frame(self, bgr: np.ndarray):
        qi = self._bgr2qimage(bgr)
        self.setPixmap(QtGui.QPixmap.fromImage(qi))

class LogPanel(QtWidgets.QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(1000)
        self.setStyleSheet("QPlainTextEdit{background:#0b0f12;color:#dfe7ef;font-family:Consolas,Monaco,monospace;font-size:12px;border-radius:8px;padding:8px}")

    def log(self, text: str):
        self.appendPlainText(text)

class ControlPanel(QtWidgets.QWidget):
    params_changed = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()
        self.setStyleSheet("QGroupBox{border:1px solid #223244;border-radius:10px;margin-top:8px} QGroupBox::title{subcontrol-origin: margin; left:10px; padding:0 4px;background:#0b0f12;color:#9fc7ff}")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6,6,6,6)
        layout.setSpacing(8)

        # Sekmeler
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet("QTabWidget::pane{border:1px solid #1d2a36;border-radius:10px} QTabBar::tab{background:#0b0f12;color:#b9c6d3;padding:6px 10px;border:1px solid #1d2a36;border-bottom:none;margin-right:2px;border-top-left-radius:6px;border-top-right-radius:6px} QTabBar::tab:selected{background:#14202b;color:#e7f1ff}")
        layout.addWidget(tabs, 1)

        # ---- Tab 1: Detection ----
        det = QtWidgets.QWidget(); det_l = QtWidgets.QFormLayout(det)
        det_l.setLabelAlignment(QtCore.Qt.AlignRight)
        # Color combo
        self.cmb_color = QtWidgets.QComboBox(); self.cmb_color.addItems(COLOR_LIST)
        # Sensitivity
        self.sld_sens = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_sens.setRange(0,100); self.sld_sens.setValue(35)
        # MinArea
        self.sld_minA = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_minA.setRange(200, 30000); self.sld_minA.setValue(1400)
        # Mode
        self.cmb_mode = QtWidgets.QComboBox(); self.cmb_mode.addItems(["ColorOnly","Color+Shape"]) ; self.cmb_mode.setCurrentIndex(0)
        # Shape
        self.cmb_shape = QtWidgets.QComboBox(); self.cmb_shape.addItems(SHAPE_LIST)

        det_l.addRow("Color", self.cmb_color)
        det_l.addRow("Sensitivity", self.sld_sens)
        det_l.addRow("MinArea", self.sld_minA)
        det_l.addRow("Mode", self.cmb_mode)
        det_l.addRow("Shape", self.cmb_shape)
        tabs.addTab(det, "Detection")

        # ---- Tab 2: InnerShape ----
        inn = QtWidgets.QWidget(); inn_l = QtWidgets.QFormLayout(inn)
        self.cmb_inner_mode = QtWidgets.QComboBox(); self.cmb_inner_mode.addItems(["auto_dark","auto_contrast","color"]); self.cmb_inner_mode.setCurrentIndex(1)
        self.cmb_inner_color = QtWidgets.QComboBox(); self.cmb_inner_color.addItems(INNER_COLORS); self.cmb_inner_color.setCurrentIndex(4)  # Black
        self.sld_inner_sens = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_inner_sens.setRange(0,100); self.sld_inner_sens.setValue(40)
        self.sld_contrast = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_contrast.setRange(5,60); self.sld_contrast.setValue(20)
        self.sld_closek = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_closek.setRange(1,9); self.sld_closek.setValue(3)
        self.sld_dilatek = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sld_dilatek.setRange(0,5); self.sld_dilatek.setValue(1)

        inn_l.addRow("InnerMode", self.cmb_inner_mode)
        inn_l.addRow("InnerColor", self.cmb_inner_color)
        inn_l.addRow("InnerSens", self.sld_inner_sens)
        inn_l.addRow("ContrastThr", self.sld_contrast)
        inn_l.addRow("CloseK", self.sld_closek)
        inn_l.addRow("DilateK", self.sld_dilatek)
        tabs.addTab(inn, "InnerShape")

        # Apply button
        btn_apply = QtWidgets.QPushButton("Apply Params")
        btn_apply.setStyleSheet("QPushButton{background:#1771e6;color:white;padding:8px;border-radius:8px} QPushButton:hover{background:#1e7ff0}")
        layout.addWidget(btn_apply)

        # Signals
        btn_apply.clicked.connect(self._emit_params)

    def _emit_params(self):
        params = {
            "selected_color": self.cmb_color.currentText(),
            "sensitivity": self.sld_sens.value(),
            "min_area": self.sld_minA.value(),
            "mode": self.cmb_mode.currentIndex(),
            "shape": self.cmb_shape.currentText(),
            # inner params -> SHAPE_PARAMS globaline yazacağız
            "inner_mode": self.cmb_inner_mode.currentText(),
            "inner_color": self.cmb_inner_color.currentText(),
            "inner_sens": self.sld_inner_sens.value(),
            "contrast_thr": self.sld_contrast.value(),
            "close_k": self.sld_closek.value() if self.sld_closek.value()%2==1 else self.sld_closek.value()+1,
            "dilate_k": self.sld_dilatek.value(),
        }
        self.params_changed.emit(params)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1280, 720)
        self._central = QtWidgets.QWidget(); self.setCentralWidget(self._central)

        root = QtWidgets.QVBoxLayout(self._central)
        top = QtWidgets.QHBoxLayout(); root.addLayout(top, 1)

        # sol: video
        self.canvas = VideoCanvas()
        top.addWidget(self.canvas, 3)

        # sağ: kontroller
        self.panel = ControlPanel()
        self.panel.setFixedWidth(380)
        top.addWidget(self.panel, 1)

        # alt: log paneli
        self.logs = LogPanel(); self.logs.setFixedHeight(160)
        root.addWidget(self.logs)

        # video işçisi
        self.worker = VideoWorker(camera_index=0)
        self.worker.frame_ready.connect(self.on_frame)

        # sinyaller
        self.panel.params_changed.connect(self.on_params_changed)

        # başlangıç parametreleri uygula ve başlat
        self.panel._emit_params()
        self.worker.start()

        # tema
        self.setStyleSheet("""
            QMainWindow{background:#0b0f12}
            QLabel{color:#e7f1ff}
            QComboBox, QSlider, QLineEdit{color:#dfe7ef}
        """)

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self.worker.stop()
        return super().closeEvent(e)

    @QtCore.Slot(dict)
    def on_params_changed(self, params: dict):
        # VideoWorker parametreleri
        self.worker.update_params(
            selected_color=params["selected_color"],
            sensitivity=params["sensitivity"],
            min_area=params["min_area"],
            mode=params["mode"],
            shape=params["shape"],
        )
        # SHAPE_PARAMS (global) güncelle
        SHAPE_PARAMS.update({
            "inner_mode": params["inner_mode"],
            "inner_color": params["inner_color"],
            "inner_sens": params["inner_sens"],
            "contrast_thr": params["contrast_thr"],
            "close_k": params["close_k"],
            "dilate_k": params["dilate_k"],
        })
        self.logs.log(f"[APPLY] {params}")

    @QtCore.Slot(np.ndarray, dict)
    def on_frame(self, bgr: np.ndarray, meta: dict):
        self.canvas.show_frame(bgr)
        dets_txt = ", ".join([f"({x},{y},r={r})" for (x,y,r) in meta["dets"]]) or "-"
        self.logs.log(f"{meta['mode']}  | color={meta['color']} shape={meta['shape']}  | dets={len(meta['dets'])} [{dets_txt}]  | fps={meta['fps']:.1f}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

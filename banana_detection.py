import sys
import cv2
import joblib
import numpy as np
import os
from datetime import datetime
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO
import subprocess
import platform
from skimage.feature import graycomatrix, graycoprops


# === Folder setup ===
os.makedirs("roi_debug", exist_ok=True)
os.makedirs("captures", exist_ok=True)

# === Load SVM ===
try:
    svm_model = joblib.load("svm_model.pkl")
    print("✅ SVM loaded. classes:", getattr(svm_model, "classes_", "N/A"))
except FileNotFoundError:
    svm_model = None
    print("⚠️ File 'svm_model.pkl' tidak ditemukan!")

try:
    yolo_model = YOLO("best.pt")
    print("✅ YOLO model berhasil dimuat.")
except Exception as e:
    yolo_model = None
    print("⚠️ YOLO model gagal dimuat:", e)

import joblib

try:
    scaler = joblib.load("scaler.pkl")
    print("✅ Scaler loaded")
except Exception as e:
    print("⚠️ Gagal load scaler:", e)
    scaler = None

try:
    pca = joblib.load("pca.pkl")
    print("✅ PCA loaded")
except Exception as e:
    print("⚠️ Gagal load PCA:", e)
    pca = None


def extract_features_live(roi, bins=(8,8,8)):
    """
    roi: BGR image (numpy array)
    returns: feature vector HSV + GLCM + shape
    """
    if roi is None or roi.size == 0:
        return None

    # Resize & blur
    roi_resized = cv2.resize(roi, (256, 256))
    roi_blur = cv2.GaussianBlur(roi_resized, (5,5), 0)

    # HSV histogram
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0,180,0,256,0,256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    # GLCM features
    gray = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    glcm = graycomatrix(gray_eq, [5], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm,'contrast')[0,0]
    energy = graycoprops(glcm,'energy')[0,0]
    homo = graycoprops(glcm,'homogeneity')[0,0]

    # Shape features (area + circularity)
    lower = np.array([15,60,60])
    upper = np.array([40,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        circ = 4*np.pi*area/(peri*peri) if peri>0 else 0
    else:
        area = circ = 0

    features = np.hstack([hist.flatten(), contrast, energy, homo, area, circ]).astype(np.float32)
    return features

def make_preview_widget(title):
    vbox = QtWidgets.QVBoxLayout()
    vbox.setSpacing(10)
    lbl_title = QtWidgets.QLabel(title)
    lbl_title.setAlignment(QtCore.Qt.AlignCenter)
    lbl_title.setStyleSheet("font-weight:bold; font-size:16px; color:#333;")
    img_lbl = QtWidgets.QLabel()
    img_lbl.setFixedSize(260, 200)
    img_lbl.setStyleSheet("border:1px solid #aaa; background-color:#f9f9f9;")
    vbox.addWidget(lbl_title, alignment=QtCore.Qt.AlignHCenter)
    vbox.addWidget(img_lbl, alignment=QtCore.Qt.AlignCenter)

    return vbox, img_lbl

class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(int, int)

    def mousePressEvent(self, event):
        self.clicked.emit(event.x(), event.y())

# === App utama ===
class BananaApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Banana Detection AI (YOLO + SVM)")
        self.showFullScreen()
        self.stats = {"unripe": 0, "ripe": 0, "overripe": 0, "rotten": 0}
        self.total_detected = 0
        self.is_paused = False
        self.roi_list = []
        self.current_roi_index = 0

        # debug counter
        self.debug_save_counter = 0

        # === Layout utama ===
        layout = QtWidgets.QHBoxLayout(self)
        left_panel = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel(self)
        self.label.setMinimumSize(1000, 700)
        self.label.setStyleSheet("background-color:#000;")
        self.label.setAlignment(QtCore.Qt.AlignCenter)  # <-- tambahkan ini
        left_panel.addWidget(self.label, stretch=1)


        button_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("▶️ Start")
        self.pause_btn = QtWidgets.QPushButton("⏸️ Pause")
        self.stop_btn = QtWidgets.QPushButton("⏹️ Stop")
        self.capture_btn = QtWidgets.QPushButton("📸 Capture")
        self.reset_btn = QtWidgets.QPushButton("🔄 Reset Chart")

        for btn in [self.start_btn, self.pause_btn, self.stop_btn, self.capture_btn, self.reset_btn]:
            btn.setFixedHeight(50)
            btn.setStyleSheet("font-size:18px;")
            button_layout.addWidget(btn)
        left_panel.addLayout(button_layout)

        # === Panel kanan ===
        right_frame = QtWidgets.QFrame()
        right_frame.setStyleSheet("background-color:white; border:2px solid #aaa; border-radius:5px;")
        right_frame.setMinimumWidth(300)
        right_layout = QtWidgets.QVBoxLayout(right_frame)
        right_layout.setSpacing(12)

        # === Label info ===
        label_font = "font-size:24px; font-weight:bold; color:black;"
        info_font = "font-size:20px; color:black;"
        self.result_label = QtWidgets.QLabel("Prediction: -")
        self.result_label.setStyleSheet(label_font)
        self.conf_label = QtWidgets.QLabel("Confidence: -")
        self.conf_label.setStyleSheet(info_font)
        self.area_label = QtWidgets.QLabel("Area: -")
        self.area_label.setStyleSheet(info_font)
        self.rasio_label = QtWidgets.QLabel("Rasio: -")
        self.rasio_label.setStyleSheet(info_font)

        right_layout.addWidget(self.result_label)
        right_layout.addWidget(self.conf_label)
        right_layout.addWidget(self.area_label)
        right_layout.addWidget(self.rasio_label)

        # === Pie chart ===
        self.figure, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas, stretch=6)

        # === Preview ROI & Histogram 2x3 ===
        roi_box, self.mask_label = make_preview_widget("ROI Preview")
        hsv_box, self.hsv_label = make_preview_widget("HSV Histogram")
        mask_morph_box, self.mask_morph_label = make_preview_widget("Morph Mask")
        hue_box, self.hue_blur_label = make_preview_widget("Hue Channel")
        sv_box, self.sv_label = make_preview_widget("Saturation Channel")
        contour_box, self.contour_label = make_preview_widget("Contour + Texture")
        extra1_box, self.extra1_label = make_preview_widget("Value Channel")
        extra2_box, self.extra2_label = make_preview_widget("CLAHE Grayscale")
        extra3_box, self.extra3_label = make_preview_widget("Canny Edge")


        grid_preview = QtWidgets.QGridLayout()
        grid_preview.setSpacing(10)
        grid_preview.addLayout(roi_box, 0, 0)
        grid_preview.addLayout(hsv_box, 0, 1)
        grid_preview.addLayout(mask_morph_box, 0, 2)
        grid_preview.addLayout(hue_box, 1, 0)
        grid_preview.addLayout(sv_box, 1, 1)
        grid_preview.addLayout(contour_box, 1, 2)
        grid_preview.addLayout(extra1_box, 2, 0)
        grid_preview.addLayout(extra2_box, 2, 1)
        grid_preview.addLayout(extra3_box, 2, 2)

        right_layout.addLayout(grid_preview, stretch=5)
        right_layout.addStretch(1)

        layout.addLayout(left_panel, stretch=4)
        layout.addWidget(right_frame, stretch=2)

        # tombol
        self.start_btn.clicked.connect(self.start_camera)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.capture_btn.clicked.connect(self.capture_frame)
        self.reset_btn.clicked.connect(self.reset_stats)

        self.cap = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.update_chart()

    def center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def update_chart(self):
        self.ax.clear()
        labels = list(self.stats.keys())
        values = list(self.stats.values())
        colors = ['#4CAF50', '#FFEB3B', '#FF9800', '#795548']
        total = sum(values)

        if total == 0:
            circle = plt.Circle((0, 0), 1, color='white', ec='lightgray', lw=2)
            self.ax.add_artist(circle)
        else:
            self.ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
            self.ax.text(0, 0, f"Detected {self.total_detected}", ha='center', va='center', fontsize=12, weight='bold')

        self.ax.set_aspect('equal')
        self.canvas.draw()

    def update_hsv_hist(self, roi):
        if roi is None or roi.size == 0:
            self.hsv_label.clear()
            return
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        cv2.normalize(hist_h, hist_h, 0, 180, cv2.NORM_MINMAX)
        hist_img = np.zeros((180, 256, 3), dtype=np.uint8)
        for i in range(1, 180):
            hue_color = cv2.cvtColor(np.uint8([[[i, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
            cv2.line(hist_img, (i-1, 180 - int(hist_h[i-1][0])), (i, 180 - int(hist_h[i][0])), hue_color.tolist(), 1)
        qimg = QtGui.QImage(cv2.resize(hist_img, (260, 200)).data, 260, 200, 260*3, QtGui.QImage.Format_RGB888)
        self.hsv_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def update_mask_morph_and_hue(self, frame):
        if frame is None or frame.size == 0:
            self.mask_morph_label.clear()
            self.hue_blur_label.clear()
            return

        # Gaussian blur
        frame_blur = cv2.GaussianBlur(frame, (5,5), 0)

        # --- Mask morfologi seluruh frame ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([15,60,60])
        upper = np.array([40,255,255])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        qmask = QtGui.QImage(cv2.resize(mask_disp, (260, 200)).data,
                             260, 200, 260*3, QtGui.QImage.Format_RGB888)
        self.mask_morph_label.setPixmap(QtGui.QPixmap.fromImage(qmask))

        # --- Hue channel seluruh frame TANPA blur ---
        hue = hsv[:,:,0]
        hue_color = cv2.cvtColor(cv2.merge([hue,hue,hue]), cv2.COLOR_HSV2BGR)
        qhue = QtGui.QImage(cv2.resize(hue_color, (260,200)).data,
                            260,200,260*3, QtGui.QImage.Format_RGB888)
        self.hue_blur_label.setPixmap(QtGui.QPixmap.fromImage(qhue))

    def update_saturation(self, frame):
        if frame is None or frame.size == 0:
            self.sv_label.clear()
            return
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        s_color = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
        qimg = QtGui.QImage(cv2.resize(s_color, (260,200)).data,
                            260,200,260*3, QtGui.QImage.Format_RGB888)
        self.sv_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def update_contour_edge_texture(self, frame=None):
        """
        Sekarang menampilkan contour + saturation full frame, bukan ROI
        """
        if frame is None:
            if not hasattr(self, "cap") or self.cap is None:
                self.contour_label.clear()
                return
            ret, frame = self.cap.read()
            if not ret:
                self.contour_label.clear()
                return

        frame_blur = cv2.GaussianBlur(frame, (5,5), 0)
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        # --- Mask untuk kontur ---
        lower = np.array([15,60,60])
        upper = np.array([40,255,255])
        mask = cv2.inRange(hsv, lower, upper)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = frame.copy()
        cv2.drawContours(contour_img, cnts, -1, (0,0,255), 2)

        # --- Saturation channel full frame ---
        s = hsv[:,:,1]
        s_color = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)

        # --- Gabungkan contour + saturation ---
        combined = cv2.addWeighted(contour_img, 0.7, s_color, 0.3, 0)

        qimg = QtGui.QImage(cv2.resize(combined, (260,200)).data,
                            260,200,260*3, QtGui.QImage.Format_RGB888)
        self.contour_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def update_value(self, frame):
        if frame is None or frame.size == 0:
            self.extra1_label.clear()
            return
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        v_color = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
        qimg = QtGui.QImage(cv2.resize(v_color, (260,200)).data,
                            260,200,260*3, QtGui.QImage.Format_RGB888)
        self.extra1_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def update_clahe(self, frame):
        if frame is None or frame.size == 0:
            self.extra2_label.clear()
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray)
        gray_color = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
        qimg = QtGui.QImage(cv2.resize(gray_color, (260,200)).data,
                            260,200,260*3, QtGui.QImage.Format_RGB888)
        self.extra2_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def update_extra3(self, frame):
        if frame is None or frame.size == 0:
            self.extra3_label.clear()
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        qimg = QtGui.QImage(cv2.resize(edges_color, (260,200)).data,
                            260,200,260*3, QtGui.QImage.Format_RGB888)
        self.extra3_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def start_camera(self):
        if not (svm_model and yolo_model):
            QtWidgets.QMessageBox.critical(self, "Error", "Model belum lengkap (YOLO/SVM)!")
            return

        # === Pakai kamera HP (Iriun Webcam) di index 2 ===
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            # fallback ke webcam laptop kalau HP belum tersambung
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                print("⚠️ Gagal pakai Iriun Webcam, fallback ke webcam laptop.")
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Error",
                    "Tidak bisa membuka kamera!\nPastikan kamera HP (Iriun) tersambung via USB dan aplikasi aktif."
                )
                return
        else:
            print("✅ Menggunakan kamera HP (Iriun Webcam via USB, index 0).")

        # === Mulai update frame setiap 30 ms ===
        self.timer.start(30)

    def toggle_pause(self):
        if self.cap:
            self.is_paused = not self.is_paused

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.label.clear()
        self.timer.stop()

    def capture_frame(self):
        if self.cap:
            for _ in range(3):
                self.cap.grab()
            ret, frame = self.cap.read()

            if ret:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                pred_text = self.result_label.text().replace("Prediction: ", "")
                filename = f"captures/capture_{pred_text}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                QtWidgets.QMessageBox.information(self, "Captured", f"Frame saved:\n{filename}")

    def reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0
        self.update_chart()

    @staticmethod
    def resize_with_padding(img, target_size=(260, 200)):
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=[249, 249, 249])

    def update_frame(self):
        if not self.cap or self.is_paused:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        # --- Simpan frame asli untuk preview ---
        frame_orig = frame.copy()

        detections = []
        if yolo_model:
            results = yolo_model(frame, imgsz=640, conf=0.5, verbose=False)
            for r in results:
                names = r.names
                boxes = r.boxes
                if boxes is not None:
                    for box, conf, cls_id in zip(
                        boxes.xyxy.cpu().numpy(),
                        boxes.conf.cpu().numpy(),
                        boxes.cls.cpu().numpy()
                    ):
                        x1, y1, x2, y2 = map(int, box)
                        label = names[int(cls_id)].lower()
                        if "banana" in label and conf > 0.5 and (x2 - x1) > 10 and (y2 - y1) > 10:
                            detections.append((x1, y1, x2, y2, float(conf)))

        # Reset stats
        self.stats = {"unripe": 0, "ripe": 0, "overripe": 0, "rotten": 0}
        roi_list = []

        if detections:
            total_conf, total_area, total_ratio = 0, 0, 0

            for (x1, y1, x2, y2, conf) in detections:
                roi = frame[y1:y2, x1:x2].copy()
                if roi is None or roi.size == 0 or svm_model is None:
                    continue

                # --- Gaussian blur ---
                roi_blur = cv2.GaussianBlur(roi, (5,5), 0)

                # --- Ekstraksi fitur lengkap ---
                features = extract_features_live(roi=roi_blur)
                if features is None:
                    continue

                # --- Transform scaler + PCA ---
                X_input = features.reshape(1, -1)
                X_input = scaler.transform(X_input)
                X_input = pca.transform(X_input)

                # --- Prediksi SVM ---
                try:
                    if hasattr(svm_model, "predict_proba"):
                        probs = svm_model.predict_proba(X_input)[0]
                        idx = int(np.argmax(probs))
                        pred_label = svm_model.classes_[idx]
                        conf_svm = probs[idx]
                        conf_display = conf_svm
                    else:
                        pred_label = svm_model.predict(X_input)[0]
                        conf_display = 1.0
                except Exception as e:
                    print("SVM predict error:", e)
                    pred_label = "unknown"
                    conf_display = conf

                # --- Bersihkan label ---
                label_map = {
                    "unripe": "unripe", "unripe_banana": "unripe", "unripe banana": "unripe",
                    "ripe": "ripe", "ripe_banana": "ripe", "ripe banana": "ripe",
                    "overripe": "overripe", "overripe_banana": "overripe", "overripe banana": "overripe",
                    "rotten": "rotten", "rotten_banana": "rotten", "rotten banana": "rotten",
                    "0": "unripe", "1": "ripe", "2": "overripe", "3": "rotten"
                }
                pred_clean = label_map.get(str(pred_label).lower().strip(), "unknown")

                # --- Warna bounding box ---
                color = {
                    "unripe": (76, 175, 80),
                    "ripe": (0, 215, 255),
                    "overripe": (0, 140, 255),
                    "rotten": (60, 60, 60),
                    "unknown": (200, 200, 200)
                }.get(pred_clean, (255, 255, 255))

                # --- Gambar bounding box & label ---
                label_text = f"{pred_clean} {conf_display:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 6, y1), color, -1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                cv2.putText(frame, label_text, (x1 + 3, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                roi_list.append(roi)
                if pred_clean in self.stats:
                    self.stats[pred_clean] += 1

                total_conf += conf_display if isinstance(conf_display, (int, float)) else 0
                area = (x2 - x1) * (y2 - y1)
                total_area += area
                total_ratio += area / (frame.shape[0] * frame.shape[1])

            # --- Update info panel ---
            n = len(roi_list)
            if n > 0:
                avg_conf = total_conf / n
                avg_area = total_area / n
                avg_ratio = total_ratio / n
                self.result_label.setText(f"Detected: {n} banana(s)")
                self.conf_label.setText(f"Avg Confidence: {avg_conf:.2f}")
                self.area_label.setText(f"Avg Area: {avg_area:.0f}")
                self.rasio_label.setText(f"Avg Ratio: {avg_ratio:.3f}")
        else:
            self.result_label.setText("Detected: -")
            self.conf_label.setText("Confidence: -")
            self.area_label.setText("Area: -")
            self.rasio_label.setText("Rasio: -")

        # --- ROI Preview & 2x3 panel ---
        self.roi_list = roi_list
        if roi_list:
            self.current_roi_index = 0
            first_roi = roi_list[0]
            preview = self.resize_with_padding(first_roi)
            qroi = QtGui.QImage(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB).data,
                                preview.shape[1], preview.shape[0],
                                preview.shape[1]*3, QtGui.QImage.Format_RGB888)
            self.mask_label.setPixmap(QtGui.QPixmap.fromImage(qroi))
            self.update_hsv_hist(first_roi)
            self.update_mask_morph_and_hue(frame_orig)
            self.update_saturation(frame_orig)
            self.update_contour_edge_texture(frame_orig)
            self.update_value(frame_orig)
            self.update_clahe(frame_orig)
            self.update_extra3(frame_orig)
        else:
            self.mask_label.clear()
            self.hsv_label.clear()
            self.mask_morph_label.clear()
            self.hue_blur_label.clear()
            self.sv_label.clear()
            self.contour_label.clear()
            self.extra1_label.clear()
            self.extra2_label.clear()
            self.extra3_label.clear()

        # --- Update pie chart ---
        self.total_detected = len(roi_list)
        self.update_chart()

        # --- Tampilkan frame utama ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0],
                            frame_rgb.shape[1]*3, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            self.label.size(),
            QtCore.Qt.KeepAspectRatioByExpanding,
            QtCore.Qt.SmoothTransformation
        )
        self.label.setPixmap(scaled_pixmap)




    def on_mask_click(self, x, y):
        if not hasattr(self, 'roi_list') or not self.roi_list:
            return
        self.current_roi_index = (self.current_roi_index + 1) % len(self.roi_list)
        roi = self.roi_list[self.current_roi_index]
        preview = self.resize_with_padding(roi)
        qroi = QtGui.QImage(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB).data,
                            preview.shape[1], preview.shape[0],
                            preview.shape[1]*3, QtGui.QImage.Format_RGB888)
        self.mask_label.setPixmap(QtGui.QPixmap.fromImage(qroi))
        self.update_hsv_hist(roi)
        print(f"Klik ROI → tampilkan ROI ke-{self.current_roi_index+1}")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

# === Main ===
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = BananaApp()
    window.show()
    sys.exit(app.exec_())

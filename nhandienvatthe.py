import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, Canvas
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os
import numpy as np
import urllib.request
import tensorflow as tf
from nsfw_detector import predict

# ================== KHỞI TẠO MODEL NSFW ==================
H5_NAME = "nsfw_mobilenet2.224x224.h5"
SAVED_DIR = "nsfw_mobilenet2.224x224_saved"
NSFW_MODEL = None


def download_h5(destination):
    url = "https://github.com/GantMan/nsfw_model/releases/download/1.0.0/nsfw_mobilenet2.224x224.h5"
    print("🔽 Đang tải model NSFW...")
    urllib.request.urlretrieve(url, destination)
    print("✅ Tải xong:", destination)


def ensure_nsfw_model():
    global NSFW_MODEL
    if os.path.isdir(SAVED_DIR):
        try:
            NSFW_MODEL = predict.load_model(SAVED_DIR)
            print("✅ Đã load NSFW model từ thư mục SavedModel.")
            return
        except Exception as e:
            print("⚠️ Lỗi load SavedModel:", e)

    if os.path.isfile(H5_NAME):
        try:
            print("🔄 Chuyển file .h5 sang SavedModel...")
            keras_model = tf.keras.models.load_model(H5_NAME)
            if os.path.isdir(SAVED_DIR):
                import shutil
                shutil.rmtree(SAVED_DIR)
            keras_model.save(SAVED_DIR)
            NSFW_MODEL = predict.load_model(SAVED_DIR)
            print("✅ Đã load NSFW model từ file .h5.")
            return
        except Exception as e:
            print("⚠️ Lỗi khi load từ .h5:", e)

    print("⚠️ Không tìm thấy model, tự động tải...")
    download_h5(H5_NAME)
    keras_model = tf.keras.models.load_model(H5_NAME)
    keras_model.save(SAVED_DIR)
    NSFW_MODEL = predict.load_model(SAVED_DIR)
    print("✅ Đã tải và load NSFW model thành công.")


ensure_nsfw_model()


# --- Hàm tính tỷ lệ vùng da người (giúp tránh loại nhầm ảnh sáng) ---
def skin_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return np.sum(mask > 0) / mask.size


# --- Hàm kiểm tra ảnh nhạy cảm ---
def is_sensitive_image(path):
    try:
        preds = predict.classify(NSFW_MODEL, path)
        result = preds[path]
        img = cv2.imread(path)
        skin = skin_ratio(img)

        # chỉ loại nếu model nghi ngờ và vùng da > 40%
        if (result["porn"] > 0.7 or result["sexy"] > 0.75) and skin > 0.4:
            print(f"🚫 Ảnh nhạy cảm bị loại: {os.path.basename(path)} ({result})")
            return True

    except Exception as e:
        print(f"⚠️ Lỗi khi phân loại ảnh {path}: {e}")
    return False


# --- Hàm tính độ sắc nét ---
def sharpness_score(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except:
        return 0


# --- Hàm đánh giá độ tập trung vào vật thể (dùng YOLO) ---
def focus_ratio(img):
    try:
        model = YOLO("yolov8n.pt")
        results = model(img, verbose=False)
        h, w = img.shape[:2]
        total_area = w * h
        object_area = 0
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box[:4]
            object_area += max(0, (x2 - x1)) * max(0, (y2 - y1))
        return min(object_area / total_area, 1.0)
    except Exception as e:
        print("⚠️ Lỗi YOLO:", e)
        return 0


# ================== MODEL PHÁT HIỆN CHỮ (EAST) ==================
EAST_MODEL = "frozen_east_text_detection.pb"
if not os.path.exists(EAST_MODEL):
    raise FileNotFoundError("❌ Không tìm thấy file 'frozen_east_text_detection.pb'!")

net = cv2.dnn.readNet(EAST_MODEL)


def text_ratio(img, conf_threshold=0.5):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (320, 320)),
        1.0,
        (320, 320),
        (123.68, 116.78, 103.94),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    (scores, geometry) = net.forward(
        ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    )
    (numRows, numCols) = scores.shape[2:4]

    boxes, confidences = [], []
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData = [geometry[0, i, y] for i in range(5)]
        for x in range(numCols):
            if scoresData[x] < conf_threshold:
                continue
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = xData[4][x]
            cos, sin = np.cos(angle), np.sin(angle)
            h_box = xData[0][x] + xData[2][x]
            w_box = xData[1][x] + xData[3][x]
            endX = int(offsetX + (cos * xData[1][x]) + (sin * xData[2][x]))
            endY = int(offsetY - (sin * xData[1][x]) + (cos * xData[2][x]))
            startX = int(endX - w_box)
            startY = int(endY - h_box)
            boxes.append([startX, startY, int(w_box), int(h_box)])
            confidences.append(float(scoresData[x]))

    if len(boxes) == 0:
        return 0.0

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

    total_area = w * h
    text_area = 0
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, bw, bh) = boxes[i]
            text_area += max(0, bw) * max(0, bh)

    return text_area / total_area


# ================== GIAO DIỆN CHÍNH ==================
class ObjectFocusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chọn ảnh ít chữ nhất & loại ảnh nhạy cảm")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f5f5f5")

        self.image_paths = []
        self.thumbnails = []
        self.best_path = None
        self.lowest_text_ratio = 1.0  # càng nhỏ càng tốt

        btn_frame = tk.Frame(root, bg="#f5f5f5")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text=" Chọn ảnh", command=self.open_images,
                  bg="#FF9800", fg="white", width=15).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Tìm ảnh ít chữ nhất", command=self.predict_best,
                  bg="#4CAF50", fg="white", width=20).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Xóa tất cả", command=self.clear_all,
                  bg="#F44336", fg="white", width=15).grid(row=0, column=2, padx=5)

        self.result_label = tk.Label(root, text="Chưa chọn ảnh.", font=("Arial", 12), bg="#f5f5f5")
        self.result_label.pack(pady=5)

        self.image_panel = tk.Label(root, bg="#ddd")
        self.image_panel.pack(pady=15)

        tk.Label(root, text="Ảnh đã chọn:", font=("Arial", 11, "bold"), bg="#f5f5f5").pack(pady=5)
        frame = tk.Frame(root, bg="#f5f5f5")
        frame.pack(fill="both", expand=False)

        self.canvas = Canvas(frame, bg="#f5f5f5", height=120)
        self.scrollbar = Scrollbar(frame, orient="horizontal", command=self.canvas.xview)
        self.thumb_frame = tk.Frame(self.canvas, bg="#f5f5f5")

        self.thumb_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.thumb_frame, anchor="nw")
        self.canvas.configure(xscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="top", fill="x", expand=True)
        self.scrollbar.pack(side="bottom", fill="x")

    # === Mở ảnh ===
    def open_images(self):
        paths = filedialog.askopenfilenames(title="Chọn ảnh", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if paths:
            self.image_paths = list(paths)
            self.display_thumbnails()
            self.result_label.config(text=f"Đã chọn {len(self.image_paths)} ảnh.")

    # === Hiển thị thumbnail ===
    def display_thumbnails(self):
        for widget in self.thumb_frame.winfo_children():
            widget.destroy()
        self.thumbnails.clear()
        for path in self.image_paths:
            try:
                img = Image.open(path)
                img.thumbnail((100, 100))
                img_tk = ImageTk.PhotoImage(img)
                lbl = tk.Label(self.thumb_frame, image=img_tk, bg="#fff", relief="solid", borderwidth=1)
                lbl.image = img_tk
                lbl.pack(side="left", padx=5, pady=5)
                self.thumbnails.append(lbl)
            except Exception:
                continue

    # === Chọn ảnh có chữ ít nhất (và dừng nếu nhạy cảm) ===
    def predict_best(self):
        if not self.image_paths:
            messagebox.showwarning("Thông báo", "Vui lòng chọn ảnh trước!")
            return

        self.result_label.config(text="Đang xử lý...", fg="black")
        self.root.update_idletasks()

        self.best_path = None
        self.lowest_text_ratio = 1.0

        for path in self.image_paths:
            try:
                # 1️⃣ Kiểm tra ảnh nhạy cảm
                if is_sensitive_image(path):
                    filename = os.path.basename(path)
                    self.result_label.config(text=f"🚫 Ảnh nhạy cảm bị loại: {filename}", fg="red")
                    messagebox.showwarning("Ảnh nhạy cảm", f"Ảnh '{filename}' bị loại do nhạy cảm!")
                    return  # ❌ Dừng luôn, không xét tiếp

                # 2️⃣ Đọc ảnh
                img = cv2.imread(path)
                if img is None:
                    continue

                # 3️⃣ Tính tỷ lệ chữ
                t_ratio = text_ratio(img)
                print(f"📄 Ảnh {os.path.basename(path)} => text_ratio = {t_ratio:.4f}")

                # 4️⃣ Chọn ảnh có chữ ít nhất
                if t_ratio < self.lowest_text_ratio:
                    self.lowest_text_ratio = t_ratio
                    self.best_path = path

            except Exception as e:
                print("❌ Lỗi xử lý:", path, e)

        # 5️⃣ Hiển thị kết quả
        if self.best_path:
            self.display_image(self.best_path)
            filename = os.path.basename(self.best_path)
            self.result_label.config(
                text=f"✅ Ảnh ít chữ nhất: {filename}\nTỷ lệ chữ: {self.lowest_text_ratio*100:.2f}%",
                fg="green"
            )
        else:
            self.result_label.config(text="❌ Không tìm được ảnh phù hợp!", fg="red")

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((500, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.image_panel.config(image=img_tk)
        self.image_panel.image = img_tk

    def clear_all(self):
        self.image_paths.clear()
        for widget in self.thumb_frame.winfo_children():
            widget.destroy()
        self.image_panel.config(image='')
        self.result_label.config(text="Đã xóa tất cả ảnh.")
        self.best_path = None


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectFocusApp(root)
    root.mainloop()

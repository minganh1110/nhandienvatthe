import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, Canvas
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os
import numpy as np

# --- Tải model phát hiện chữ EAST ---
EAST_MODEL = "frozen_east_text_detection.pb"
if not os.path.exists(EAST_MODEL):
    raise FileNotFoundError("❌ Không tìm thấy file 'frozen_east_text_detection.pb' trong thư mục hiện tại!")

net = cv2.dnn.readNet(EAST_MODEL)

# --- Hàm tính độ sắc nét ---
def sharpness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# --- Hàm kiểm tra ảnh có chứa chữ bằng EAST ---
def contains_text(img, conf_threshold=0.5):
    h, w = img.shape[:2]
    new_w, new_h = (320, 320)
    rW, rH = w / float(new_w), h / float(new_h)

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (new_w, new_h)),
        1.0,
        (new_w, new_h),
        (123.68, 116.78, 103.94),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < conf_threshold:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))

    boxes = cv2.dnn.NMSBoxes(
        [cv2.boundingRect(np.array([[x1, y1], [x2, y2]])) for (x1, y1, x2, y2) in rects],
        confidences, conf_threshold, 0.4
    )

    # Nếu có bất kỳ box nào phát hiện chữ
    return len(boxes) > 0

# --- Hàm đánh giá độ tập trung vào vật thể ---
def focus_ratio(result, img_shape):
    h, w, _ = img_shape
    total_area = w * h
    object_areas = []

    for box in result.boxes:
        conf = float(box.conf[0])
        if conf >= 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            object_areas.append(area)

    if not object_areas:
        return 0

    main_area = max(object_areas)
    main_ratio = main_area / total_area
    object_count = len(object_areas)

    ideal_focus = 0.4
    focus_score = max(0, 1 - abs(main_ratio - ideal_focus))
    clutter_penalty = 1 / (1 + (object_count - 1) * 2)
    return focus_score * clutter_penalty

# --- Giao diện chính ---
class ObjectFocusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chọn ảnh tập trung vào vật thể nhất")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f5f5f5")
        self.model = YOLO("yolov8n.pt")

        self.image_paths = []
        self.thumbnails = []

        btn_frame = tk.Frame(root, bg="#f5f5f5")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text=" Chọn ảnh", command=self.open_images, bg="#FF9800", fg="white", width=15).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Tìm ảnh vật thể rõ nhất", command=self.predict_best, bg="#4CAF50", fg="white", width=20).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Xóa tất cả", command=self.clear_all, bg="#F44336", fg="white", width=15).grid(row=0, column=2, padx=5)

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

    def open_images(self):
        paths = filedialog.askopenfilenames(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if paths:
            self.image_paths = list(paths)
            self.display_thumbnails()
            self.result_label.config(text=f"Đã chọn {len(self.image_paths)} ảnh.")

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

    def predict_best(self):
        if not self.image_paths:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước.")
            return

        results = []
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                continue

            # Bỏ qua ảnh có chữ (EAST)
            if contains_text(img):
                print(f"🟡 Bỏ qua ảnh có chữ: {path}")
                continue

            result = self.model(path, verbose=False)[0]
            ratio = focus_ratio(result, img.shape)
            sharp = sharpness_score(img)
            score = (ratio * 1000) + (sharp * 0.05)
            results.append((path, score))

        if not results:
            messagebox.showinfo("Kết quả", "Không tìm thấy ảnh hợp lệ (toàn ảnh có chữ).")
            return

        best_image = max(results, key=lambda x: x[1])[0]
        self.display_image(best_image)
        self.result_label.config(text=f"Ảnh tập trung vào vật thể nhất: {os.path.basename(best_image)}")

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((500, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.image_panel.config(image=img_tk)
        self.image_panel.image = img_tk

    def clear_all(self):
        self.image_paths = []
        self.thumbnails.clear()
        for widget in self.thumb_frame.winfo_children():
            widget.destroy()
        self.image_panel.config(image='')
        self.result_label.config(text="Đã xóa tất cả ảnh.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectFocusApp(root)
    root.mainloop()

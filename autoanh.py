# -*- coding: utf-8 -*-
import os, time, glob, subprocess, threading, queue, json, sys
from pathlib import Path

# ---- GUI / CV / IO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import pandas as pd
from ultralytics import YOLO
from openpyxl import Workbook, load_workbook

# ================== DEFAULT CONFIG (có thể bị override bởi settings.json) ==================
EXCEL_FILE_DEFAULT   = r"C:\toollinkanh\nhandienvatthe\link.xlsx"
DOWNLOAD_DIR_DEFAULT = r"C:\toollinkanh\nhandienvatthe\imgdownload\fileanhtam"
IMG_DONE_DIR_DEFAULT = r"C:\toollinkanh\nhandienvatthe\imgdone"
RESULT_XLSX_DEFAULT  = str(Path(IMG_DONE_DIR_DEFAULT) / "result_links.xlsx")

CHROME_EXE           = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
USER_DATA_DIR        = r"C:\Users\dell\AppData\Local\Google\Chrome\User Data"
PROFILE_DIR_DEFAULT  = "Profile 47"

YOLO_WEIGHTS = "yolov8n.pt"
CONF_THRES   = 0.5
IDEAL_RATIO  = 0.4
MIN_SCORE    = 25.0
OPEN_LINK_DELAY = 0.8
MAX_WAIT_DL  = 90
IMG_EXTS     = (".jpg", ".jpeg", ".png", ".webp")
CLOSE_TAB_EACH_ROUND = True

HEADERS = ["stt", "link sản phẩm", "stt ảnh", "output_path", "score", "focus", "sharp"]
SETTINGS_PATH = Path(__file__).with_name("settings.json")

# ================== UTIL: load/save settings ==================
def load_settings():
    s = {
        "excel_file": EXCEL_FILE_DEFAULT,
        "download_dir": DOWNLOAD_DIR_DEFAULT,
        "img_done_dir": IMG_DONE_DIR_DEFAULT,
        "result_xlsx": RESULT_XLSX_DEFAULT,
        "profile_dir_name": PROFILE_DIR_DEFAULT
    }
    if SETTINGS_PATH.exists():
        try:
            s.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    # đảm bảo result_xlsx nằm trong img_done_dir
    if not s.get("result_xlsx"):
        s["result_xlsx"] = str(Path(s["img_done_dir"]) / "result_links.xlsx")
    return s

def save_settings(d):
    # chỉ lưu các field quan trọng
    keep = {k: d[k] for k in ["excel_file","download_dir","img_done_dir","result_xlsx","profile_dir_name"] if k in d}
    SETTINGS_PATH.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")

# ================== FILE / EXCEL helpers ==================
def ensure_dirs(download_dir, img_done_dir):
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(img_done_dir, exist_ok=True)

def clear_download_dir(download_dir):
    for pattern in [*[f"*{ext}" for ext in IMG_EXTS], "*.crdownload"]:
        for p in glob.glob(os.path.join(download_dir, pattern)):
            try: os.remove(p)
            except: pass

def excel_init(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.append(HEADERS)
    wb.save(path); wb.close()

def excel_ensure(path):
    if not os.path.exists(path):
        excel_init(path)

def excel_append_row(path, row):
    excel_ensure(path)
    wb = load_workbook(path); ws = wb.active
    ws.append(row); wb.save(path); wb.close()

def excel_clear_file(path):
    excel_init(path)

def excel_next_stt(path):
    excel_ensure(path)
    wb = load_workbook(path); ws = wb.active
    r = ws.max_row; wb.close()
    return r  # header = 1, dòng tiếp theo = max_row + 1 => r đang chính là STT

def read_links(excel_path):
    if not os.path.exists(excel_path):
        raise FileNotFoundError(excel_path)
    df = pd.read_excel(excel_path, engine="openpyxl")
    col = "link" if "link" in df.columns else df.columns[0]
    return [str(x).strip() for x in df[col].tolist() if str(x).strip()]

# ================== DOWNLOAD WATCHER ==================
def list_images(dirpath):
    res = []
    for ext in IMG_EXTS:
        res += glob.glob(os.path.join(dirpath, f"*{ext}"))
    return res

def has_crdownload(dirpath):
    return bool(glob.glob(os.path.join(dirpath, "*.crdownload")))

def wait_download_batch(download_dir, timeout=90, quiet_seconds=5.0):
    t0 = time.time()
    last_sig, last_change = None, time.time()

    def snapshot():
        fs = list_images(download_dir)
        sig = tuple(sorted((p, os.path.getsize(p)) for p in fs))
        return fs, sig

    while time.time() - t0 < timeout:
        if has_crdownload(download_dir):
            last_sig = None; last_change = time.time()
            time.sleep(0.3); continue
        files, sig = snapshot()
        if sig != last_sig:
            last_sig, last_change = sig, time.time()
        else:
            if time.time() - last_change >= quiet_seconds:
                return files
        time.sleep(0.3)
    return list_images(download_dir)

# ================== SCORING ==================
def sharpness_score(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def focus_ratio(res, shape):
    h, w = shape[:2]
    tot = max(1, w * h)
    areas = []
    for box in res.boxes:
        if float(box.conf[0]) >= CONF_THRES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            areas.append(max(0, (x2 - x1) * (y2 - y1)))
    if not areas: return 0.0
    main = max(areas) / tot
    clutter = 1 / (1 + (len(areas) - 1) * 2)
    return max(0, 1 - abs(main - IDEAL_RATIO)) * clutter

def composite_score(f, s):
    return f * 1000 + s * 0.05

def score_and_pick(model, files):
    best = ("", 0, 0, 0)
    for p in files:
        try:
            img = cv2.imread(p)
            if img is None: continue
            res = model(p, verbose=False)[0]
            f = focus_ratio(res, img.shape)
            s = sharpness_score(img)
            sc = composite_score(f, s)
            if sc > best[1]:
                best = (p, sc, f, s)
        except:
            continue
    return best if best[0] else (None, 0, 0, 0)

# ================== OPEN LINK ==================
def open_in_chrome(url, profile_dir_name):
    subprocess.Popen(
        [
            CHROME_EXE,
            f"--user-data-dir={USER_DATA_DIR}",
            f"--profile-directory={profile_dir_name}",
            "--new-tab",
            url,
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

# ================== THREAD RUNNER ==================
class RunnerThread(threading.Thread):
    def __init__(self, ui):
        super().__init__(daemon=True)
        self.ui = ui
        self._stop = threading.Event()

    def stop(self): self._stop.set()

    def run(self):
        s = self.ui.settings
        ensure_dirs(s["download_dir"], s["img_done_dir"])
        excel_ensure(s["result_xlsx"])
        model = YOLO(YOLO_WEIGHTS)

        try:
            links = read_links(self.ui.var_excel.get())
        except Exception as e:
            self.ui.log(f"❌ Không đọc được Excel: {e}")
            self.ui.after(0, self.ui.on_thread_done); return

        self.ui.log(f"Tổng {len(links)} link.")
        for link in links:
            if self._stop.is_set(): break

            clear_download_dir(s["download_dir"])
            stt = excel_next_stt(s["result_xlsx"])
            self.ui.log(f"\n[{stt}] Mở link: {link}")
            open_in_chrome(link, self.ui.var_profile.get())
            time.sleep(OPEN_LINK_DELAY)

            self.ui.log("… đợi ảnh ổn định trong thư mục download…")
            files = wait_download_batch(s["download_dir"], MAX_WAIT_DL, 5)
            if not files:
                self.ui.log("  ❌ Không thấy ảnh.")
                excel_append_row(s["result_xlsx"], [stt, link, 0, "", "", "", ""])
                continue

            files = sorted(files, key=os.path.getmtime)
            best, sc, f, sh = score_and_pick(model, files)
            if not best or sc < MIN_SCORE:
                self.ui.log("  ❌ Không có ảnh đạt.")
                excel_append_row(s["result_xlsx"], [stt, link, 0, "", "", "", len(files)])
            else:
                idx = files.index(best) + 1
                out_name = f"{stt:03d}.jpg"
                out_path = str(Path(s["img_done_dir"]) / out_name)
                # Lưu về imgdone (convert jpg cho đồng nhất)
                try:
                    im = Image.open(best).convert("RGB")
                    im.save(out_path, "JPEG", quality=92, subsampling=1)
                except Exception:
                    import shutil; shutil.copy2(best, out_path)

                self.ui.log(f"  ✔ Ảnh #{idx} score={sc:.1f} → {out_name}")
                excel_append_row(s["result_xlsx"], [stt, link, idx, out_path, f"{sc:.2f}", f"{f:.4f}", f"{sh:.2f}"])
                self.ui.show_preview(out_path)

            # dọn download cho vòng sau
            clear_download_dir(s["download_dir"])

            # đóng tab
            if CLOSE_TAB_EACH_ROUND:
                self.ui.try_close_tab()

        self.ui.log("\nHoàn tất!")
        self.ui.after(0, self.ui.on_thread_done)

# ================== APP (UI) ==================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # DPI aware (Windows) để tỉ lệ chữ/khung ổn
        try:
            if sys.platform.startswith("win"):
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass

        self.title("Shopee Auto Pick — Excel only")
        self.geometry("1000x640")
        self.minsize(900, 560)

        self.settings = load_settings()

        # ---------- Layout: trái (config/log) – phải (preview co giãn) ----------
        root = ttk.Frame(self); root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0)   # left fixed
        root.columnconfigure(1, weight=1)   # right grow
        root.rowconfigure(0, weight=1)

        left = ttk.Frame(root, width=340)
        left.grid(row=0, column=0, sticky="nsw")
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ============ LEFT: config ============
        ttk.Label(left, text="Cấu hình", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(10,6), columnspan=3)

        # Excel link
        self.var_excel = tk.StringVar(value=self.settings["excel_file"])
        row1 = ttk.Frame(left); row1.grid(row=1, column=0, sticky="ew", padx=10)
        ttk.Label(row1, text="Excel link:").pack(side="left")
        ttk.Entry(row1, textvariable=self.var_excel, width=28).pack(side="left", padx=6)
        ttk.Button(row1, text="Chọn…", command=self.pick_excel).pack(side="left")

        # Profile input + Save
        self.var_profile = tk.StringVar(value=self.settings["profile_dir_name"])
        row2 = ttk.Frame(left); row2.grid(row=2, column=0, sticky="ew", padx=10, pady=(6,0))
        ttk.Label(row2, text="Chrome profile:").pack(side="left")
        ttk.Entry(row2, textvariable=self.var_profile, width=20).pack(side="left", padx=6)
        ttk.Button(row2, text="Lưu", command=self.save_profile).pack(side="left")

        # Path hiển thị (download & imgdone)
        ttk.Label(left, text=f"DL: {self.settings['download_dir']}", foreground="#666").grid(row=3, column=0, sticky="w", padx=10, pady=(4,0))
        ttk.Label(left, text=f"OUT: {self.settings['img_done_dir']}", foreground="#666").grid(row=4, column=0, sticky="w", padx=10)

        # Buttons
        btns = ttk.Frame(left); btns.grid(row=5, column=0, sticky="w", padx=10, pady=(6,4))
        self.btn_start = ttk.Button(btns, text="▶ Start", command=self.start_run); self.btn_start.pack(side="left", padx=3)
        self.btn_stop  = ttk.Button(btns, text="⏹ Stop",  command=self.stop_run, state="disabled"); self.btn_stop.pack(side="left", padx=3)
        ttk.Button(btns, text="🧹 Clear file", command=self.clear_all).pack(side="left", padx=3)
        ttk.Button(btns, text="↺ Reset",      command=self.reset_tool).pack(side="left", padx=3)

        # Log (Text + scrollbar)  <<< CHỈNH MÀU Ở ĐÂY >>>
        ttk.Label(left, text="Log:").grid(row=6, column=0, sticky="w", padx=10)
        logwrap = ttk.Frame(left); logwrap.grid(row=7, column=0, sticky="nsew", padx=10, pady=(0,10))
        left.rowconfigure(7, weight=1)
        self.txt = tk.Text(
            logwrap, height=26, width=44,
            bg="#1e293b",            # slate-800 (không đen, dịu mắt)
            fg="#e2e8f0",            # slate-200
            insertbackground="#e2e8f0",
            selectbackground="#334155",  # slate-700 khi bôi chọn
            selectforeground="#f8fafc",
            wrap="word",
            borderwidth=0,
            highlightthickness=0
        )
        vsb = ttk.Scrollbar(logwrap, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=vsb.set)
        self.txt.pack(side="left", fill="both", expand=True); vsb.pack(side="right", fill="y")

        # ============ RIGHT: preview (auto fit) ============
        ttk.Label(right, text="Preview ảnh chọn", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(10,6))
        self.preview_canvas = tk.Canvas(right, bg="#202020", highlightthickness=0)
        self.preview_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.preview_path = None
        self.preview_imgtk = None
        self.preview_canvas.bind("<Configure>", self._redraw_preview)

        # queue & thread
        self.queue = queue.Queue()
        self.runner = None
        self.after(80, self.drain_queue)

    # -------- preview helpers --------
    def _redraw_preview(self, _evt=None):
        self.preview_canvas.delete("all")
        if not self.preview_path or not os.path.exists(self.preview_path): 
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width()/2,
                self.preview_canvas.winfo_height()/2,
                text="(Chưa có ảnh chọn)", fill="#888"
            ); return
        try:
            im = Image.open(self.preview_path).convert("RGB")
            cw, ch = self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height()
            im.thumbnail((max(1, cw-4), max(1, ch-4)))
            self.preview_imgtk = ImageTk.PhotoImage(im)
            self.preview_canvas.create_image(cw//2, ch//2, image=self.preview_imgtk)
        except Exception as e:
            self.preview_canvas.create_text(10,10, anchor="nw", text=f"Lỗi ảnh: {e}", fill="#f66")

    def show_preview(self, path):
        self.queue.put(f"__PREVIEW__|{path}")

    # -------- queue & log --------
    def log(self, msg): self.queue.put(msg)

    def drain_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg.startswith("__PREVIEW__|"):
                    _, p = msg.split("|", 1)
                    self.preview_path = p
                    self._redraw_preview()
                else:
                    self.txt.insert("end", msg + "\n"); self.txt.see("end")
        except queue.Empty:
            pass
        self.after(80, self.drain_queue)

    # -------- buttons / actions --------
    def pick_excel(self):
        f = filedialog.askopenfilename(title="Chọn file Excel", filetypes=[("Excel", "*.xlsx")])
        if f:
            self.var_excel.set(f)
            self.settings["excel_file"] = f
            save_settings(self.settings)

    def save_profile(self):
        self.settings["profile_dir_name"] = self.var_profile.get().strip() or PROFILE_DIR_DEFAULT
        save_settings(self.settings)
        self.log(f"Đã lưu profile: {self.settings['profile_dir_name']}")

    def clear_all(self):
        if messagebox.askyesno("Xoá", "Xoá toàn bộ dữ liệu trong file kết quả?"):
            excel_clear_file(self.settings["result_xlsx"])
            self.log("Đã tạo lại header.")

    def reset_tool(self):
        if self.runner and self.runner.is_alive():
            r = self.runner; r.stop()
            try: r.join(timeout=1.0)
            except: pass
            self.runner = None
        clear_download_dir(self.settings["download_dir"])
        self.txt.delete("1.0", "end")
        self.preview_path = None; self._redraw_preview()
        self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
        self.log("Reset xong.")

    def start_run(self):
        if self.runner and self.runner.is_alive(): return
        # cập nhật settings mới nhất từ UI
        self.settings["excel_file"]  = self.var_excel.get().strip()
        self.settings["result_xlsx"] = str(Path(self.settings["img_done_dir"]) / "result_links.xlsx")
        save_settings(self.settings)
        self.txt.delete("1.0", "end")
        self.runner = RunnerThread(self); self.runner.start()
        self.btn_start.config(state="disabled"); self.btn_stop.config(state="normal")

    def stop_run(self):
        if self.runner and self.runner.is_alive():
            r = self.runner; r.stop()
            try: r.join(timeout=1.0)
            except: pass
            self.runner = None
        self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
        self.log("Đã dừng tool.")

    def on_thread_done(self):
        self.runner = None
        self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")

    def try_close_tab(self):
        try:
            import pyautogui
            pyautogui.hotkey("ctrl", "w")
        except:
            pass

# ================== MAIN ==================
if __name__ == "__main__":
    app = App()
    # lần đầu: đảm bảo thư mục tồn tại
    ensure_dirs(app.settings["download_dir"], app.settings["img_done_dir"])
    app.mainloop()

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
from opencv_engine import TrafficProcessor
from ultralytics import solutions
import cv2
import os
import json
import time

PANELS = ["north", "east", "west", "south"]
ROI_FILE = "roi.json"


class VideoPanel:
    def __init__(self, parent, name):
        self.name = name

        #------ START PROCESSING --------
        self.processor = TrafficProcessor("python/dataset/yolo11n.pt")
        self.processing = False
        self.worker_thread = None
        self.processed_frame = None
        self.stats = {}

        # ----- STATE -----
        self.cap = None
        self.imgtk = None
        self.last_frame = None  # last BGR frame (OpenCV)

        # ROI state (normalized points)
        self.editing_roi = False
        self.roi_points_norm = []
        self.roi_closed = False

        # This frame will be managed by GRID (by the parent)
        self.frame = tk.LabelFrame(parent, text=name.upper(), padx=5, pady=5)

        # Canvas (create FIRST)
        self.canvas = tk.Canvas(self.frame, bg="black")
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # Bind events AFTER canvas exists
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)  # redraw ROI on resize

        # Buttons
        self.btn_frame = tk.Frame(self.frame)
        self.btn_frame.pack(fill=tk.X)

        self.load_btn = tk.Button(self.btn_frame, text="Load Video", command=self.load_video)
        self.load_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.edit_btn = tk.Button(self.btn_frame, text="Edit Area", command=self.edit_roi)
        self.edit_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.save_btn = tk.Button(self.btn_frame, text="Save Area", command=self.save_roi_temp)
        self.save_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Load ROI from disk (if exists)
        self.load_roi_temp()
        self.redraw_overlay()

    # ---------- Video Processing ----------
    def start_processing(self):
        if self.processing or not self.cap:
            return

        self.processing = True
        self.worker_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.worker_thread.start()
    #------ PROCESSING LOOP --------
    def _processing_loop(self):
        while self.processing:
            if self.latest_frame is None:
                # wait until UI thread provides a frame
                time.sleep(0.01)
                continue

            frame = self.latest_frame.copy()  # copy safe frame

            # apply ROI only if exists
            roi_px = self.get_roi_polygon_px(frame)
            if roi_px is not None:
                self.processor.set_roi(roi_px.tolist())

            processed, count, green, fin, fout = self.processor.process_frame(frame)

            self.processed_frame = processed
            self.stats = {
                "count": count,
                "green": green,
                "fuzzy_in": fin,
                "fuzzy_out": fout
            }


    # ---------- Video ----------
    def load_video(self, path=None):
        if not path:
            path = filedialog.askopenfilename(
                filetypes=[("Video Files", "*.mp4 *.avi *.mkv")]
            )
        if not path:
            return

        # release old capture
        if self.cap:
            self.cap.release()
            self.cap = None

        cap = cv2.VideoCapture(path)

        # Fail early
        if not cap.isOpened():
            messagebox.showerror("Load Video Failed", f"Cannot open video:\n{path}")
            return

        # Try reading first frame to confirm it's actually readable
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            messagebox.showerror("Load Video Failed", f"Video opened but cannot read frames:\n{path}")
            return

        # If OK, keep it
        self.cap = cap
        self.last_frame = frame

        # Show info
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        h, w = frame.shape[:2]
        messagebox.showinfo(
            "Video Loaded",
            f"[{self.name.upper()}] Loaded successfully!\n\n"
            f"File: {os.path.basename(path)}\n"
            f"Resolution: {w} x {h}\n"
            f"FPS: {fps:.2f}"
        )

        # Display first frame immediately, then continue playing
        self.show_frame(frame)
        self.frame.after(30, self.update_frame)

        if not self.processing:
            self.processing = True
            threading.Thread(target=self.processing_loop, daemon=True).start()



    def update_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()

        if not ret:
            # restart from beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame.after(30, self.update_frame)
            return

        self.last_frame = frame
        self.latest_frame = frame  # feed worker with new frame

        if self.processed_frame is not None:
            # show processed frame from worker
            self.show_frame(self.processed_frame)
        else:
            # show raw frame if processing not ready
            self.show_frame(frame)

        self.frame.after(30, self.update_frame)


    def show_frame(self, frame_bgr):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((canvas_w, canvas_h), Image.BILINEAR)
        self.imgtk = ImageTk.PhotoImage(img)

        self.canvas.delete("bg")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk, tags="bg")

        self.redraw_overlay()

    def on_canvas_resize(self, event):
        if self.last_frame is not None:
            self.show_frame(self.last_frame)


    # ---------- ROI Editing ----------
    def edit_roi(self):
        self.editing_roi = not self.editing_roi
        if self.editing_roi:
            # start new ROI
            self.roi_points_norm = []
            self.roi_closed = False
            self.edit_btn.config(relief=tk.SUNKEN)
            print(f"[{self.name}] ROI editing started")
        else:
            self.edit_btn.config(relief=tk.RAISED)
            print(f"[{self.name}] ROI editing stopped")
        self.redraw_overlay()

    def on_canvas_click(self, event):
        if not self.editing_roi or self.roi_closed:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        x_norm = event.x / cw
        y_norm = event.y / ch

        # clamp 0..1
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))

        self.roi_points_norm.append({"x": x_norm, "y": y_norm})
        self.redraw_overlay()

    def on_canvas_double_click(self, event):
        if not self.editing_roi:
            return

        if len(self.roi_points_norm) >= 3:
            self.roi_closed = True
            self.editing_roi = False
            self.edit_btn.config(relief=tk.RAISED)
            self.save_roi_temp()
            self.redraw_overlay()
            print(f"[{self.name}] ROI closed & saved")

    def redraw_overlay(self):
        # Only delete ROI drawings, keep bg image
        self.canvas.delete("roi")

        if not self.roi_points_norm:
            return

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return

        pts = [(p["x"] * w, p["y"] * h) for p in self.roi_points_norm]

        # points
        for x, y in pts:
            self.canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4,
                fill="lime", outline="", tags="roi"
            )

        # polyline
        if len(pts) >= 2:
            flat = []
            for x, y in pts:
                flat.extend([x, y])
            self.canvas.create_line(*flat, fill="lime", width=2, tags="roi")

        # close polygon
        if self.roi_closed and len(pts) >= 3:
            self.canvas.create_line(
                pts[-1][0], pts[-1][1],
                pts[0][0], pts[0][1],
                fill="lime", width=2, tags="roi"
            )

        # hint text
        if self.editing_roi:
            self.canvas.create_text(
                10, 10, anchor="nw",
                text="ROI edit: click points, double-click to close",
                fill="white", tags="roi"
            )

    # ---------- ROI Persistence (JSON) ----------
    def save_roi_temp(self):
        data = {}
        if os.path.exists(ROI_FILE):
            try:
                with open(ROI_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data[self.name] = {
            "points": self.roi_points_norm,
            "closed": self.roi_closed
        }

        with open(ROI_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"[{self.name}] ROI saved to {ROI_FILE}")
        messagebox.showinfo(
                "Area Saved",
                f"Counted Area on the '{self.name.upper()}' has been saved successfully!"
        )

    def load_roi_temp(self):
        if not os.path.exists(ROI_FILE):
            return

        try:
            with open(ROI_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        if self.name in data:
            item = data[self.name]
            self.roi_points_norm = item.get("points", []) or []
            self.roi_closed = bool(item.get("closed", False))
            print(f"[{self.name}] ROI loaded from {ROI_FILE}")

    # ---------- OpenCV bridge ----------
    def get_roi_polygon_px(self, frame_bgr):
        """
        Returns Nx2 int32 polygon points in pixel space (OpenCV-friendly),
        or None if ROI not valid.
        """
        if not self.roi_closed or len(self.roi_points_norm) < 3:
            return None

        h, w = frame_bgr.shape[:2]
        pts = np.array(
            [[int(p["x"] * w), int(p["y"] * h)] for p in self.roi_points_norm],
            dtype=np.int32
        )
        return pts

    def get_roi_mask(self, frame_bgr):
        """
        Returns a uint8 mask (same w,h) with ROI filled, or None.
        """
        pts = self.get_roi_polygon_px(frame_bgr)
        if pts is None:
            return None
        h, w = frame_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def draw_traffic_overlay(self):
        s = self.stats
        y = 10

        def line(text, color="white"):
            nonlocal y
            self.canvas.create_text(
                10, y,
                anchor="nw",
                text=text,
                fill=color,
                font=("Segoe UI", 10, "bold"),
                tags="overlay"
            )
            y += 18

        self.canvas.delete("overlay")

        line("SMART TRAFFIC LIGHT", "cyan")
        line(f"Vehicles: {s.get('vehicles', 0)}", "yellow")
        line(f"Green Time: {s.get('green_time', 0)} s", "lime")

        fin = s.get("fuzzy_in", {})
        fout = s.get("fuzzy_out", {})

        if fin:
            line(f"Fuzzy In  L:{fin['low']:.2f} M:{fin['medium']:.2f} H:{fin['high']:.2f}")

        if fout:
            line(f"Fuzzy Out S:{fout['short']:.2f} M:{fout['medium']:.2f} L:{fout['long']:.2f}")

class TrafficApp:
    def __init__(self, root):
        self.root = root
        root.title("Smart Traffic Control System")
        root.geometry("1400x800")

        # ================= TOP TOOLBAR =================
        toolbar = tk.Frame(root)
        toolbar.pack(fill=tk.X)

        tk.Button(
            toolbar,
            text="Load Videos (Auto Assign)",
            command=self.load_videos_auto
        ).pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(
            toolbar,
            text="Reload ROI",
            command=self.reload_all_roi
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # ================= MAIN LAYOUT =================
        main = tk.Frame(root)
        main.pack(expand=True, fill=tk.BOTH)

        main.columnconfigure(0, weight=4)  # video area
        main.columnconfigure(1, weight=1)  # info panel
        main.rowconfigure(0, weight=1)

        # ================= LEFT: VIDEO GRID =================
        video_grid = tk.Frame(main)
        video_grid.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        video_grid.rowconfigure((0, 1), weight=1)
        video_grid.columnconfigure((0, 1), weight=1)

        self.panels = {}

        positions = {
            "north": (0, 0),
            "east":  (0, 1),
            "west":  (1, 0),
            "south": (1, 1)
        }

        for name, (r, c) in positions.items():
            panel = VideoPanel(video_grid, name)
            panel.frame.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            self.panels[name] = panel

        # ================= RIGHT: INFO PANEL =================
        self.info_panel = tk.LabelFrame(
            main,
            text="Traffic Information",
            padx=10,
            pady=10
        )
        self.info_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.info_labels = {}

        for name in ["north", "east", "west", "south"]:
            lbl = tk.Label(
                self.info_panel,
                text=self._default_info_text(name),
                justify="left",
                anchor="w",
                font=("Segoe UI", 10),
                padx=6,
                pady=6,
                relief=tk.GROOVE
            )
            lbl.pack(fill="x", pady=5)
            self.info_labels[name] = lbl

        # start periodic update
        self.update_info_panel()

    # ================= INFO PANEL =================
    def _default_info_text(self, name):
        return (
            f"{name.upper()}\n"
            f"Vehicles: 0\n"
            f"Green Light: 0 s"
        )

    def update_info_panel(self):
        for name, panel in self.panels.items():
            stats = getattr(panel, "stats", None)

            if not stats:
                self.info_labels[name].config(
                    text=self._default_info_text(name)
                )
                continue

            text = (
                f"{name.upper()}\n"
                f"Vehicles: {stats.get('count', 0)}\n"
                f"Green Light: {stats.get('green', 0)} s"
            )

            self.info_labels[name].config(text=text)

        # update every 500 ms
        self.root.after(500, self.update_info_panel)

    # ================= ACTIONS =================
    def reload_all_roi(self):
        for p in self.panels.values():
            p.load_roi_temp()
            p.redraw_overlay()

    def load_videos_auto(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv")]
        )

        for path in files:
            base = os.path.basename(path).lower()
            for panel_name in PANELS:
                if panel_name in base:
                    self.panels[panel_name].load_video(path)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficApp(root)
    root.mainloop()

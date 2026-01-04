import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import threading
from opencv_engine import TrafficProcessor
from db import TrafficDatabase
from traffic_controller import TrafficLightController
import cv2
import os
import json
import time
import math

PANELS = ["north", "east", "west", "south"]
ROI_FILE = "roi.json"


class VideoPanel:
    def __init__(self, parent, name, light_controller, database):
        self.name = name
        self.light_controller = light_controller
        self.database = database

        # ------ PROCESSING --------
        self.processor = TrafficProcessor("python/dataset/yolo11n.pt")
        self.processing = False
        self.worker_thread = None
        self.processed_frame = None
        self.latest_frame = None
        self.stats = {}

        # Logging timer (every 5 seconds)
        self.last_log_time = time.time()
        self.log_interval = 5  # seconds

        # ----- STATE -----
        self.cap = None
        self.imgtk = None
        self.last_frame = None

        # ROI state
        self.editing_roi = False
        self.roi_points_norm = []
        self.roi_closed = False

        # Frame
        self.frame = tk.LabelFrame(parent, text=name.upper(), padx=5, pady=5)

        # Canvas
        self.canvas = tk.Canvas(self.frame, bg="black")
        self.canvas.pack(expand=True, fill=tk.BOTH)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Buttons
        self.btn_frame = tk.Frame(self.frame)
        self.btn_frame.pack(fill=tk.X)

        self.load_btn = tk.Button(self.btn_frame, text="Load Video", command=self.load_video)
        self.load_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.edit_btn = tk.Button(self.btn_frame, text="Edit Area", command=self.edit_roi)
        self.edit_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.save_btn = tk.Button(self.btn_frame, text="Save Area", command=self.save_roi_temp)
        self.save_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Traffic light indicator
        self.light_frame = tk.Frame(self.frame, height=40, bg="black")
        self.light_frame.pack(fill=tk.X)
        self.light_frame.pack_propagate(False)

        self.light_canvas = tk.Canvas(self.light_frame, bg="black", height=40, highlightthickness=0)
        self.light_canvas.pack(fill=tk.BOTH, expand=True)

        # Load ROI
        self.load_roi_temp()
        self.redraw_overlay()

        # Start light update
        self.update_light_display()

    def update_light_display(self):
        """Update traffic light visual indicator"""
        state = self.light_controller.get_state(self.name)
        light = state["light"]
        timer = state["timer"]

        # Clear canvas
        self.light_canvas.delete("all")

        # Draw traffic light
        x_start = 10
        circle_r = 15

        # RED light
        red_color = "#FF0000" if light == "RED" else "#440000"
        self.light_canvas.create_oval(
            x_start, 10, x_start + circle_r * 2, 10 + circle_r * 2,
            fill=red_color, outline="white", width=2
        )

        # YELLOW light
        yellow_color = "#FFFF00" if light == "YELLOW" else "#444400"
        self.light_canvas.create_oval(
            x_start + 40, 10, x_start + 40 + circle_r * 2, 10 + circle_r * 2,
            fill=yellow_color, outline="white", width=2
        )

        # GREEN light
        green_color = "#00FF00" if light == "GREEN" else "#004400"
        self.light_canvas.create_oval(
            x_start + 80, 10, x_start + 80 + circle_r * 2, 10 + circle_r * 2,
            fill=green_color, outline="white", width=2
        )

        # Timer text
        if light in ["GREEN", "YELLOW"]:
            self.light_canvas.create_text(
                x_start + 120, 20,
                text=f"Timer: {math.ceil(timer)}s",
                fill="white",
                font=("Segoe UI", 12, "bold"),
                anchor="w"
            )

        # Schedule next update
        self.frame.after(500, self.update_light_display)

    def start_processing(self):
        if self.processing or not self.cap:
            return

        self.processing = True
        self.worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.worker_thread.start()
        print(f"[{self.name}] Processing thread started")

    def _processing_loop(self):
        while self.processing:
            if self.latest_frame is None:
                time.sleep(0.01)
                continue

            frame = self.latest_frame.copy()

            roi_px = self.get_roi_polygon_px(frame)
            if roi_px is not None:
                self.processor.set_roi(roi_px.tolist())
            else:
                self.processor.set_roi(None)

            try:
                processed, count, green, fuzzy_in, fuzzy_out = self.processor.process_frame(frame)

                self.processed_frame = processed
                self.stats = {
                    "count": count,
                    "green": green,
                    "fuzzy_in": fuzzy_in,
                    "fuzzy_out": fuzzy_out
                }

                # Update green duration in light controller
                self.light_controller.update_green_duration(self.name, green)

                # Log to database every 5 seconds
                current_time = time.time()
                if current_time - self.last_log_time >= self.log_interval:
                    light_state = self.light_controller.get_state(self.name)["light"]
                    self.database.log_traffic(
                        self.name, count, green, light_state, fuzzy_in
                    )
                    self.last_log_time = current_time

            except Exception as e:
                print(f"[{self.name}] Processing error: {e}")
                time.sleep(0.1)

    def load_video(self, path=None):
        if not path:
            path = filedialog.askopenfilename(
                filetypes=[("Video Files", "*.mp4 *.avi *.mkv")]
            )
        if not path:
            return

        if self.cap:
            self.cap.release()
            self.cap = None

        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            messagebox.showerror("Load Video Failed", f"Cannot open video:\n{path}")
            return

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            messagebox.showerror("Load Video Failed", f"Video opened but cannot read frames:\n{path}")
            return

        self.cap = cap
        self.last_frame = frame
        self.latest_frame = frame

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        h, w = frame.shape[:2]
        messagebox.showinfo(
            "Video Loaded",
            f"[{self.name.upper()}] Loaded successfully!\n\n"
            f"File: {os.path.basename(path)}\n"
            f"Resolution: {w} x {h}\n"
            f"FPS: {fps:.2f}"
        )

        self.show_frame(frame)
        self.frame.after(30, self.update_frame)

        if not self.processing:
            self.start_processing()

    def update_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame.after(30, self.update_frame)
            return

        self.last_frame = frame
        self.latest_frame = frame

        if self.processed_frame is not None:
            self.show_frame(self.processed_frame)
        else:
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

    def edit_roi(self):
        self.editing_roi = not self.editing_roi
        if self.editing_roi:
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
        self.canvas.delete("roi")

        if not self.roi_points_norm:
            return

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return

        pts = [(p["x"] * w, p["y"] * h) for p in self.roi_points_norm]

        for x, y in pts:
            self.canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4,
                fill="lime", outline="", tags="roi"
            )

        if len(pts) >= 2:
            flat = []
            for x, y in pts:
                flat.extend([x, y])
            self.canvas.create_line(*flat, fill="lime", width=2, tags="roi")

        if self.roi_closed and len(pts) >= 3:
            self.canvas.create_line(
                pts[-1][0], pts[-1][1],
                pts[0][0], pts[0][1],
                fill="lime", width=2, tags="roi"
            )

        if self.editing_roi:
            self.canvas.create_text(
                10, 10, anchor="nw",
                text="ROI edit: click points, double-click to close",
                fill="white", tags="roi"
            )

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

    def get_roi_polygon_px(self, frame_bgr):
        if not self.roi_closed or len(self.roi_points_norm) < 3:
            return None

        h, w = frame_bgr.shape[:2]
        pts = np.array(
            [[int(p["x"] * w), int(p["y"] * h)] for p in self.roi_points_norm],
            dtype=np.int32
        )
        return pts


class TrafficApp:
    def __init__(self, root):
        self.root = root
        root.title("Smart Traffic Control System")
        root.geometry("1600x900")

        # Initialize database and light controller
        self.database = TrafficDatabase()
        self.light_controller = TrafficLightController(PANELS)
        
        # AUTO START traffic lights immediately
        self.light_controller.start()
        print("🚦 Traffic lights AUTO-STARTED: NORTH is GREEN")

        # Top toolbar
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

        tk.Button(
            toolbar,
            text="View Stats",
            command=self.show_stats_window,
            bg="cyan"
        ).pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(
            toolbar,
            text="Export CSV",
            command=self.export_data
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Main layout
        main = tk.Frame(root)
        main.pack(expand=True, fill=tk.BOTH)

        main.columnconfigure(0, weight=4)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Video grid
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
            panel = VideoPanel(video_grid, name, self.light_controller, self.database)
            panel.frame.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            self.panels[name] = panel

        # Info panel
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

        self.update_info_panel()

    def _default_info_text(self, name):
        return (
            f"{name.upper()}\n"
            f"Vehicles: 0\n"
            f"Green Light: 0 s\n"
            f"Status: WAITING"
        )

    def update_info_panel(self):
        for name, panel in self.panels.items():
            stats = getattr(panel, "stats", None)
            light_state = self.light_controller.get_state(name)

            if not stats:
                text = (
                    f"{name.upper()}\n"
                    f"Vehicles: 0\n"
                    f"Green Light: 0 s\n"
                    f"Light: {light_state['light']}"
                )
            else:
                text = (
                    f"{name.upper()}\n"
                    f"Vehicles: {stats.get('count', 0)}\n"
                    f"Green Light: {stats.get('green', 0)} s\n"
                    f"Light: {light_state['light']}"
                )

            self.info_labels[name].config(text=text)

        self.root.after(500, self.update_info_panel)

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

    def show_stats_window(self):
        """Show statistics window"""
        stats_win = tk.Toplevel(self.root)
        stats_win.title("Traffic Statistics")
        stats_win.geometry("800x600")

        # Notebook (tabs)
        notebook = ttk.Notebook(stats_win)
        notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Recent logs tab
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Recent Logs")

        # Create treeview
        tree = ttk.Treeview(
            logs_frame,
            columns=("Time", "Panel", "Vehicles", "Green", "Light"),
            show="headings"
        )

        tree.heading("Time", text="Timestamp")
        tree.heading("Panel", text="Panel")
        tree.heading("Vehicles", text="Vehicles")
        tree.heading("Green", text="Green Time")
        tree.heading("Light", text="Light State")

        tree.column("Time", width=150)
        tree.column("Panel", width=100)
        tree.column("Vehicles", width=100)
        tree.column("Green", width=100)
        tree.column("Light", width=100)

        tree.pack(expand=True, fill=tk.BOTH)

        # Load data
        logs = self.database.get_recent_logs(100)
        for log in logs:
            tree.insert("", "end", values=log)

        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")

        stats_text = tk.Text(stats_frame, wrap=tk.WORD, font=("Courier", 10))
        stats_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Get statistics for each panel
        stats_text.insert(tk.END, "="*60 + "\n")
        stats_text.insert(tk.END, "TRAFFIC STATISTICS\n")
        stats_text.insert(tk.END, "="*60 + "\n\n")

        for panel in PANELS:
            stats = self.database.get_statistics(panel)
            if stats:
                total, avg, min_v, max_v, avg_green = stats
                stats_text.insert(tk.END, f"{panel.upper()}\n")
                stats_text.insert(tk.END, f"  Total Logs: {total}\n")
                stats_text.insert(tk.END, f"  Avg Vehicles: {avg:.2f}\n")
                stats_text.insert(tk.END, f"  Min Vehicles: {min_v}\n")
                stats_text.insert(tk.END, f"  Max Vehicles: {max_v}\n")
                stats_text.insert(tk.END, f"  Avg Green Time: {avg_green:.2f}s\n\n")

        stats_text.config(state=tk.DISABLED)

    def export_data(self):
        """Export data to CSV"""
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )

        if path:
            if self.database.export_to_csv(path):
                messagebox.showinfo(
                    "Export Success",
                    f"Data exported successfully to:\n{path}"
                )


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficApp(root)
    root.mainloop()
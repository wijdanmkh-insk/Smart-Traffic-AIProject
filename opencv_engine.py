import cv2
import numpy as np
from ultralytics import solutions
from fuzzy_logic import TrafficFuzzyController

class TrafficProcessor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.fuzzy = TrafficFuzzyController()
        self.counter = None
        self.last_roi = None

    def set_roi(self, roi_points_px):
        """
        roi_points_px: list of (x, y) in pixel coordinates
        """
        if roi_points_px is None or len(roi_points_px) < 3:
            self.counter = None
            self.last_roi = None
            return

        # Rebuild ObjectCounter when ROI changes
        if roi_points_px != self.last_roi:
            self.counter = solutions.ObjectCounter(
                show=False,
                region=roi_points_px,
                model=self.model_path
            )
            self.last_roi = roi_points_px

    def process_frame(self, frame):
        """
        Process ONE frame.
        Returns:
          processed_frame, vehicle_count, green_time, fuzzy_in, fuzzy_out
        """
        if self.counter is None:
            # No ROI yet → just return original frame
            return frame, 0, 15, {}, {}

        results = self.counter(frame)

        processed_frame = (
            results.plot_im if hasattr(results, "plot_im") else frame
        )

        if hasattr(self.counter, "in_count"):
            vehicle_count = self.counter.in_count
        else:
            vehicle_count = len(results.boxes) if hasattr(results, "boxes") else 0

        green_time, fuzzy_in, fuzzy_out = (
            self.fuzzy.calculate_green_time(vehicle_count)
        )

        return processed_frame, vehicle_count, green_time, fuzzy_in, fuzzy_out
    def draw_traffic_overlay(frame, vehicle_count, green_time, fuzzy_in, fuzzy_out):
        """
        Draw traffic info directly on the frame (OpenCV overlay).
        """
        y = 25
        line_h = 22

        def text(msg, color=(255, 255, 255)):
            nonlocal y
            cv2.putText(
                frame, msg,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )
            y += line_h

        # Background box (for readability)
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (420, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        text("SMART TRAFFIC LIGHT", (0, 255, 255))
        text(f"Vehicles: {vehicle_count}", (0, 255, 255))
        text(f"Green Time: {green_time:.1f} s", (0, 255, 0))

        if fuzzy_in:
            text(
                f"Fuzzy In  L:{fuzzy_in['low']:.2f} "
                f"M:{fuzzy_in['medium']:.2f} "
                f"H:{fuzzy_in['high']:.2f}"
            )

        if fuzzy_out:
            text(
                f"Fuzzy Out S:{fuzzy_out['short']:.2f} "
                f"M:{fuzzy_out['medium']:.2f} "
                f"L:{fuzzy_out['long']:.2f}"
            )

        return frame

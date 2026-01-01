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
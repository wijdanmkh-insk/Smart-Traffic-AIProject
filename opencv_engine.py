import cv2
import numpy as np
from ultralytics import YOLO
import time

# ==================== FUZZY LOGIC SYSTEM ====================
class TrafficFuzzyController:
    def __init__(self):
        # Membership functions parameters
        self.vehicle_low = (0, 0, 5, 10)      # Trapezoid: (a, b, c, d)
        self.vehicle_medium = (5, 10, 15, 20)
        self.vehicle_high = (15, 25, 40, 40)
        
        self.time_short = (5, 5, 15, 25)      # Green light: 5-25 seconds
        self.time_medium = (20, 30, 40, 50)   # Green light: 20-50 seconds
        self.time_long = (45, 60, 90, 90)     # Green light: 45-90 seconds
    
    def trapezoid_membership(self, x, a, b, c, d):
        """Calculate trapezoid membership function"""
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return 1.0
        elif c < x < d:
            return (d - x) / (d - c)
    
    def fuzzify_vehicles(self, count):
        """Fuzzify vehicle count into linguistic variables"""
        low = self.trapezoid_membership(count, *self.vehicle_low)
        medium = self.trapezoid_membership(count, *self.vehicle_medium)
        high = self.trapezoid_membership(count, *self.vehicle_high)
        return {'low': low, 'medium': medium, 'high': high}
    
    def apply_rules(self, fuzzy_input):
        """Apply fuzzy rules"""
        # Rule 1: IF vehicles is LOW THEN time is SHORT
        rule1 = fuzzy_input['low']
        
        # Rule 2: IF vehicles is MEDIUM THEN time is MEDIUM
        rule2 = fuzzy_input['medium']
        
        # Rule 3: IF vehicles is HIGH THEN time is LONG
        rule3 = fuzzy_input['high']
        
        return {'short': rule1, 'medium': rule2, 'long': rule3}
    
    def defuzzify(self, fuzzy_output):
        """Defuzzify using Center of Gravity (COG) method"""
        # Create sample points for each output membership function
        x = np.linspace(5, 90, 850)
        
        # Calculate membership values for each x
        mu_short = np.array([self.trapezoid_membership(xi, *self.time_short) for xi in x])
        mu_medium = np.array([self.trapezoid_membership(xi, *self.time_medium) for xi in x])
        mu_long = np.array([self.trapezoid_membership(xi, *self.time_long) for xi in x])
        
        # Apply fuzzy output strengths (min with rule strength)
        mu_short = np.minimum(mu_short, fuzzy_output['short'])
        mu_medium = np.minimum(mu_medium, fuzzy_output['medium'])
        mu_long = np.minimum(mu_long, fuzzy_output['long'])
        
        # Aggregate using max
        aggregated = np.maximum(np.maximum(mu_short, mu_medium), mu_long)
        
        # Calculate centroid
        if np.sum(aggregated) == 0:
            return 15  # Default minimal time
        
        centroid = np.sum(x * aggregated) / np.sum(aggregated)
        return round(centroid, 1)
    
    def calculate_green_time(self, vehicle_count):
        """Main function to calculate green light duration"""
        fuzzy_input = self.fuzzify_vehicles(vehicle_count)
        fuzzy_output = self.apply_rules(fuzzy_input)
        green_duration = self.defuzzify(fuzzy_output)
        
        return green_duration, fuzzy_input, fuzzy_output


# ==================== TRAFFIC PROCESSOR ====================
class TrafficProcessor:
    def __init__(self, model_path):
        """Initialize traffic processor with YOLO model"""
        self.model = YOLO(model_path)
        self.fuzzy = TrafficFuzzyController()
        self.roi_polygon = None
        
        # Tracking
        self.vehicle_count = 0
        self.tracked_ids = set()
        
    def set_roi(self, polygon_points):
        """Set ROI polygon points (list of [x, y] coordinates)"""
        if polygon_points and len(polygon_points) >= 3:
            self.roi_polygon = np.array(polygon_points, dtype=np.int32)
        else:
            self.roi_polygon = None
    
    def is_in_roi(self, bbox_center):
        """Check if point is inside ROI polygon"""
        if self.roi_polygon is None:
            return True  # No ROI = count everything
        
        result = cv2.pointPolygonTest(
            self.roi_polygon, 
            (float(bbox_center[0]), float(bbox_center[1])), 
            False
        )
        return result >= 0
    
    def process_frame(self, frame):
        """
        Process a single frame with YOLO detection + fuzzy logic
        Returns: (processed_frame, vehicle_count, green_time, fuzzy_in, fuzzy_out)
        """
        # Run YOLO detection
        results = self.model.track(frame, persist=True, verbose=False)
        
        # Create copy for drawing
        annotated = frame.copy()
        
        # Draw ROI if exists
        if self.roi_polygon is not None:
            cv2.polylines(
                annotated, 
                [self.roi_polygon], 
                isClosed=True, 
                color=(0, 255, 0), 
                thickness=2
            )
        
        # Count vehicles in ROI
        current_ids = set()
        vehicles_in_roi = 0
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Check if in ROI
                if self.is_in_roi((center_x, center_y)):
                    vehicles_in_roi += 1
                    current_ids.add(track_id)
                    
                    # Draw bounding box (green if in ROI)
                    cv2.rectangle(
                        annotated, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (0, 255, 0), 
                        2
                    )
                    
                    # Draw ID
                    cv2.putText(
                        annotated,
                        f"ID:{track_id}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                else:
                    # Draw grey box if outside ROI
                    cv2.rectangle(
                        annotated, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (128, 128, 128), 
                        1
                    )
        
        # Update tracked vehicles (cumulative count)
        self.tracked_ids.update(current_ids)
        self.vehicle_count = len(self.tracked_ids)
        
        # Calculate fuzzy logic
        green_time, fuzzy_in, fuzzy_out = self.fuzzy.calculate_green_time(vehicles_in_roi)
        
        return annotated, vehicles_in_roi, green_time, fuzzy_in, fuzzy_out
    
    def reset_count(self):
        """Reset vehicle counter"""
        self.vehicle_count = 0
        self.tracked_ids.clear()
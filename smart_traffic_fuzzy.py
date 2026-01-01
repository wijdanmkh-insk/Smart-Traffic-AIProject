import cv2
import numpy as np
from ultralytics import solutions
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

# ==================== MAIN PROGRAM ====================
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        points.append((x, y))

def draw_traffic_info(frame, vehicle_count, green_time, fuzzy_in, fuzzy_out):
    """Draw traffic light information on frame"""
    h, w = frame.shape[:2]
    
    # Create info panel
    panel_height = 200
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    
    # Title
    cv2.putText(panel, "SMART TRAFFIC LIGHT - FUZZY LOGIC", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Vehicle count
    cv2.putText(panel, f"Vehicles Detected: {vehicle_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Green light duration
    cv2.putText(panel, f"Green Light Duration: {green_time} seconds", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Fuzzy input
    cv2.putText(panel, f"Fuzzy Input - Low: {fuzzy_in['low']:.2f} | Med: {fuzzy_in['medium']:.2f} | High: {fuzzy_in['high']:.2f}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Fuzzy output
    cv2.putText(panel, f"Fuzzy Output - Short: {fuzzy_out['short']:.2f} | Med: {fuzzy_out['medium']:.2f} | Long: {fuzzy_out['long']:.2f}",
                (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Instructions
    cv2.putText(panel, "Press 'q' to quit | Press 's' to save info", (10, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Combine frame and panel
    combined = np.vstack([frame, panel])
    return combined

def count_objects_in_region(video_path, output_video_path, model_path):
    """Count objects in a specific region within a video with fuzzy logic controller."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
    # Create video writer with increased height for info panel
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h + 200))

    region_points = [(318, 30), (412, 123), (529, 106), (425, 11)]
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)
    
    # Initialize fuzzy controller
    fuzzy_controller = TrafficFuzzyController()
    
    # Variables for tracking
    vehicle_count = 0
    green_time = 15
    fuzzy_in = {'low': 0, 'medium': 0, 'high': 0}
    fuzzy_out = {'short': 0, 'medium': 0, 'long': 0}
    
    print("\n" + "="*60)
    print("SMART TRAFFIC LIGHT SIMULATION WITH FUZZY LOGIC")
    print("="*60)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("\nVideo frame is empty or processing is complete.")
            break
    
        # Mark clicked points
        for p in points:
            cv2.circle(im0, p, 5, (0, 0, 255), -1)

        # Process frame with YOLO
        results = counter(im0)
        processed_frame = results.plot_im
        
        # Get vehicle count from counter
        current_count = len(results.boxes) if hasattr(results, 'boxes') else 0
        
        # Update vehicle count (use counter's internal count if available)
        if hasattr(counter, 'in_count'):
            vehicle_count = counter.in_count
        else:
            vehicle_count = current_count
        
        # Calculate green light duration using fuzzy logic
        green_time, fuzzy_in, fuzzy_out = fuzzy_controller.calculate_green_time(vehicle_count)
        
        # Draw traffic information
        final_frame = draw_traffic_info(processed_frame, vehicle_count, green_time, fuzzy_in, fuzzy_out)
        
        # Show frame
        cv2.imshow("Smart Traffic Light - Fuzzy Logic", final_frame)
        cv2.setMouseCallback("Smart Traffic Light - Fuzzy Logic", mouse_callback)
        
        # Write frame
        video_writer.write(final_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nInterrupted by user (q pressed)")
            break
        elif key == ord('s'):
            print(f"\n[SAVED INFO] Vehicles: {vehicle_count} | Green Time: {green_time}s")
    
    # Cleanup
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Last Vehicle Count: {vehicle_count}")
    print(f"Calculated Green Light Duration: {green_time} seconds")
    print(f"Fuzzy Membership - Low: {fuzzy_in['low']:.2f} | Medium: {fuzzy_in['medium']:.2f} | High: {fuzzy_in['high']:.2f}")
    print("="*60)

# Run the program
count_objects_in_region("north.mp4", "/output_video.avi", "python/dataset/yolo11n.pt")
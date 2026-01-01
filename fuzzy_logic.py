import numpy as np
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
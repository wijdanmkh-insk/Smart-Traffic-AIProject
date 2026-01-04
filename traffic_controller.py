"""
Traffic Light Cycle Controller
Manages light states and timers for all panels
Sequence: North → East → West → South (repeat)
"""

import time
import threading


class TrafficLightController:
    def __init__(self, panels):
        """
        Initialize traffic light controller
        panels: list of panel names (e.g., ['north', 'east', 'west', 'south'])
        """
        self.panels = panels
        self.current_index = 0
        self.cycle_order = ["north", "east", "west", "south"]
        
        # Light states for each panel
        self.light_states = {panel: "RED" for panel in panels}
        self.timers = {panel: 0 for panel in panels}
        self.green_durations = {panel: 15 for panel in panels}  # Default 15s
        
        # Yellow light duration (transition)
        self.yellow_duration = 3  # 3 seconds
        
        # Current active panel
        self.active_panel = None
        self.running = False
        self.thread = None
        
        # Transition state
        self.in_yellow = False
        self.yellow_timer = 0
    
    def start(self):
        """Start the traffic light cycle"""
        if self.running:
            return
        
        self.running = True
        self.active_panel = self.cycle_order[0]
        self.light_states[self.active_panel] = "GREEN"
        self.timers[self.active_panel] = self.green_durations[self.active_panel]
        
        # Start control thread
        self.thread = threading.Thread(target=self._cycle_loop, daemon=True)
        self.thread.start()
        
        print("🚦 Traffic light controller started")
        print(f"   First green: {self.active_panel.upper()}")
    
    def stop(self):
        """Stop the traffic light cycle"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        print("🚦 Traffic light controller stopped")
    
    def update_green_duration(self, panel, duration):
        """Update green light duration for a panel"""
        self.green_durations[panel] = max(5, min(90, duration))  # Clamp 5-90s
    
    def get_state(self, panel):
        """Get current state for a panel"""
        return {
            "light": self.light_states.get(panel, "RED"),
            "timer": self.timers.get(panel, 0),
            "is_active": self.active_panel == panel
        }
    
    def _cycle_loop(self):
        """Main cycle loop (runs in separate thread)"""
        last_tick = time.time()
        
        while self.running:
            current_time = time.time()
            elapsed = current_time - last_tick
            
            if elapsed >= 1.0:  # Update every second
                last_tick = current_time
                self._update_timers()
            
            time.sleep(0.1)  # Small sleep to avoid CPU overuse
    
    def _update_timers(self):
        """Update all timers and handle transitions"""
        if not self.active_panel:
            return
        
        # Yellow transition state
        if self.in_yellow:
            self.yellow_timer -= 1
            
            if self.yellow_timer <= 0:
                # Yellow finished, move to next panel
                self._switch_to_next()
                self.in_yellow = False
            
            return
        
        # Update active panel timer
        self.timers[self.active_panel] -= 1
        
        # Check if green time is up
        if self.timers[self.active_panel] <= 0:
            # Start yellow transition
            self.light_states[self.active_panel] = "YELLOW"
            self.in_yellow = True
            self.yellow_timer = self.yellow_duration
            self.timers[self.active_panel] = self.yellow_duration
    
    def _switch_to_next(self):
        """Switch to next panel in cycle"""
        # Set current panel to RED
        if self.active_panel:
            self.light_states[self.active_panel] = "RED"
            self.timers[self.active_panel] = 0
        
        # Move to next panel
        self.current_index = (self.current_index + 1) % len(self.cycle_order)
        self.active_panel = self.cycle_order[self.current_index]
        
        # Set new panel to GREEN
        self.light_states[self.active_panel] = "GREEN"
        self.timers[self.active_panel] = self.green_durations[self.active_panel]
        
        print(f"🚦 Light changed: {self.active_panel.upper()} is now GREEN")
    
    def get_all_states(self):
        """Get states for all panels"""
        return {
            panel: self.get_state(panel)
            for panel in self.panels
        }

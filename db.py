"""
SQLite Database Handler for Traffic Data Logging
Logs vehicle count every 5 seconds per panel
"""

import sqlite3
import os
from datetime import datetime
import csv

DB_FILE = "traffic_data.db"


class TrafficDatabase:
    def __init__(self, db_path=DB_FILE):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.init_database()
    
    def init_database(self):
        """Initialize database and create tables if not exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Create table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS traffic_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                panel TEXT NOT NULL,
                vehicle_count INTEGER NOT NULL,
                green_time REAL NOT NULL,
                light_state TEXT NOT NULL,
                fuzzy_low REAL,
                fuzzy_medium REAL,
                fuzzy_high REAL
            )
        """)
        
        # Create index for faster queries
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON traffic_logs(timestamp)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_panel 
            ON traffic_logs(panel)
        """)
        
        self.conn.commit()
        print(f"✅ Database initialized: {self.db_path}")
    
    def log_traffic(self, panel, vehicle_count, green_time, light_state, fuzzy_in=None):
        """Log traffic data for a panel"""
        try:
            fuzzy_low = fuzzy_in.get('low', 0) if fuzzy_in else 0
            fuzzy_medium = fuzzy_in.get('medium', 0) if fuzzy_in else 0
            fuzzy_high = fuzzy_in.get('high', 0) if fuzzy_in else 0
            
            self.cursor.execute("""
                INSERT INTO traffic_logs 
                (panel, vehicle_count, green_time, light_state, fuzzy_low, fuzzy_medium, fuzzy_high)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (panel, vehicle_count, green_time, light_state, fuzzy_low, fuzzy_medium, fuzzy_high))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"❌ Database log error: {e}")
            return False
    
    def get_recent_logs(self, limit=100):
        """Get recent logs (all panels)"""
        try:
            self.cursor.execute("""
                SELECT timestamp, panel, vehicle_count, green_time, light_state
                FROM traffic_logs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            return self.cursor.fetchall()
        except Exception as e:
            print(f"❌ Database query error: {e}")
            return []
    
    def get_panel_logs(self, panel, limit=50):
        """Get logs for specific panel"""
        try:
            self.cursor.execute("""
                SELECT timestamp, vehicle_count, green_time, light_state
                FROM traffic_logs
                WHERE panel = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (panel, limit))
            
            return self.cursor.fetchall()
        except Exception as e:
            print(f"❌ Database query error: {e}")
            return []
    
    def get_statistics(self, panel=None):
        """Get statistics (average, min, max vehicle count)"""
        try:
            if panel:
                self.cursor.execute("""
                    SELECT 
                        COUNT(*) as total_logs,
                        AVG(vehicle_count) as avg_vehicles,
                        MIN(vehicle_count) as min_vehicles,
                        MAX(vehicle_count) as max_vehicles,
                        AVG(green_time) as avg_green_time
                    FROM traffic_logs
                    WHERE panel = ?
                """, (panel,))
            else:
                self.cursor.execute("""
                    SELECT 
                        COUNT(*) as total_logs,
                        AVG(vehicle_count) as avg_vehicles,
                        MIN(vehicle_count) as min_vehicles,
                        MAX(vehicle_count) as max_vehicles,
                        AVG(green_time) as avg_green_time
                    FROM traffic_logs
                """)
            
            return self.cursor.fetchone()
        except Exception as e:
            print(f"❌ Statistics query error: {e}")
            return None
    
    def export_to_csv(self, output_path="traffic_export.csv"):
        """Export all data to CSV"""
        try:
            self.cursor.execute("""
                SELECT timestamp, panel, vehicle_count, green_time, light_state,
                       fuzzy_low, fuzzy_medium, fuzzy_high
                FROM traffic_logs
                ORDER BY timestamp DESC
            """)
            
            rows = self.cursor.fetchall()
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Panel', 'Vehicle Count', 'Green Time', 
                               'Light State', 'Fuzzy Low', 'Fuzzy Medium', 'Fuzzy High'])
                writer.writerows(rows)
            
            print(f"✅ Data exported to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Export error: {e}")
            return False
    
    def clear_old_logs(self, days=30):
        """Clear logs older than specified days"""
        try:
            self.cursor.execute("""
                DELETE FROM traffic_logs
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            deleted = self.cursor.rowcount
            self.conn.commit()
            print(f"✅ Deleted {deleted} old logs")
            return deleted
        except Exception as e:
            print(f"❌ Clear logs error: {e}")
            return 0
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")
# src/indexer.py

import sqlite3
import datetime
import os
from typing import List, Dict, Any, Optional
import json # To store list/dict data

# --- Configuration ---
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'security_analysis_od.db') # Use new DB file name

class FrameIndexerSQLiteOD: # Renamed class slightly
    def __init__(self, db_path: str = DB_PATH):
        """Initializes the SQLite frame indexer for Object Detection results."""
        self.db_path = db_path
        print(f"FrameIndexerSQLiteOD: Initializing database at {self.db_path}")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)

    def _create_table(self):
        """Creates the 'detections' table if it doesn't exist."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                # Modified Schema for Object Detection
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP NOT NULL,
                        location TEXT NOT NULL,
                        frame_number INTEGER,
                        detected_objects TEXT NOT NULL -- Store list of detection dicts as JSON string
                        -- Removed vlm_description, potential_actions, raw_description
                    )
                ''')
                conn.commit()
                print("FrameIndexerSQLiteOD: 'detections' table ensured.")
        except sqlite3.Error as e:
            print(f"FrameIndexerSQLiteOD: Database error during table creation: {e}")
            raise

    def add_detections(self,
                       timestamp: datetime.datetime,
                       location: str, # << Needs this
                       detections: List[Dict[str, Any]], # << Needs this
                       frame_number: Optional[int] = None):
        """Adds object detection results for a frame to the SQLite database."""
        # Store the entire list of detection dicts as a JSON string
        detections_json = json.dumps(detections)

        sql = '''
            INSERT INTO detections (timestamp, location, frame_number, detected_objects)
            VALUES (?, ?, ?, ?)
        '''
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                # Note: Check for DeprecationWarning solution if needed for Python 3.12+ adapters
                cursor.execute(sql, (timestamp, location, frame_number, detections_json))
                conn.commit()
        except sqlite3.Error as e:
            print(f"FrameIndexerSQLiteOD: Error adding detections to DB: {e}")


    def query_detections(self,
                         start_time: Optional[datetime.datetime] = None,
                         end_time: Optional[datetime.datetime] = None,
                         location: Optional[str] = None,
                         object_label: Optional[str] = None, # Query based on object label
                         min_confidence: Optional[float] = None, # Query by confidence
                         limit: int = 100
                         ) -> List[Dict[str, Any]]:
        """Queries the indexed detections from the SQLite database."""
        query_parts = ["SELECT id, timestamp, location, frame_number, detected_objects FROM detections"]
        conditions = []
        params = []

        if start_time: conditions.append("timestamp >= ?"); params.append(start_time)
        if end_time: conditions.append("timestamp <= ?"); params.append(end_time)
        if location: conditions.append("lower(location) = lower(?)"); params.append(location)

        # --- Querying JSON content ---
        # Simple LIKE is brittle. Ideally requires JSON1 extension support in SQLite.
        # We'll filter *after* fetching for object_label and min_confidence for simplicity.
        # If performance was critical, we'd use JSON1: json_extract(detected_objects, '$[*].label') = ?
        # For now, fetch all matching time/location and filter in Python.

        if conditions:
            query_parts.append("WHERE " + " AND ".join(conditions))

        query_parts.append("ORDER BY timestamp DESC")
        query_parts.append(f"LIMIT {limit * 5}") # Fetch more initially to allow for post-filtering

        sql = " ".join(query_parts)
        results = []

        try:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # Note: Check for DeprecationWarning solution if needed for Python 3.12+ adapters
                cursor.execute(sql, params)
                rows = cursor.fetchall()

                processed_count = 0
                for row in rows:
                    if processed_count >= limit: break # Apply limit after filtering

                    row_dict = dict(row)
                    try:
                        # Convert JSON string back to list of detection dicts
                        detections_list = json.loads(row_dict['detected_objects'] or '[]')
                        row_dict['detected_objects'] = detections_list # Replace JSON string with list

                        # --- Post-filtering ---
                        passes_filter = True
                        if object_label or min_confidence:
                             found_match = False
                             for detection in detections_list:
                                 label_matches = (not object_label) or (detection.get('label', '').lower() == object_label.lower())
                                 confidence_matches = (not min_confidence) or (detection.get('score', 0) >= min_confidence)
                                 if label_matches and confidence_matches:
                                     found_match = True
                                     break # Found at least one object matching criteria in this frame
                             if not found_match:
                                 passes_filter = False

                        if passes_filter:
                            results.append(row_dict)
                            processed_count += 1

                    except json.JSONDecodeError:
                         print(f"Warning: Could not decode JSON for row id {row_dict.get('id')}")
                         continue # Skip rows with bad JSON

        except sqlite3.Error as e:
             print(f"FrameIndexerSQLiteOD: Database error during query: {e}")

        return results # Returns already filtered list respecting the limit

    def get_index_size(self) -> int:
        """Returns the total number of frames indexed."""
        # ... (same as before, just query COUNT(*) from 'detections' table) ...
        sql = "SELECT COUNT(*) FROM detections"
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                count = cursor.fetchone()[0]
                return count
        except sqlite3.Error as e:
             print(f"FrameIndexerSQLiteOD: Database error getting count: {e}")
             return 0

# --- Basic Test ---
if __name__ == "__main__":
    # Use a different DB file for OD testing
    TEST_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'security_analysis_od_TEST.db')
    if os.path.exists(TEST_DB_PATH): os.remove(TEST_DB_PATH)

    print("Testing SQLite Indexer for Object Detection...")
    indexer = FrameIndexerSQLiteOD(db_path=TEST_DB_PATH)

    # Add sample detection data
    print("\nAdding dummy data...")
    ts1 = datetime.datetime.now() - datetime.timedelta(minutes=5)
    ts2 = datetime.datetime.now() - datetime.timedelta(minutes=2)
    ts3 = datetime.datetime.now()

    detections1 = [{"label": "car", "score": 0.95, "box": [10, 10, 50, 50]}, {"label": "traffic light", "score": 0.88, "box": [60, 5, 70, 30]}]
    detections2 = [{"label": "person", "score": 0.92, "box": [100, 50, 120, 150]}, {"label": "car", "score": 0.75, "box": [200, 80, 280, 150]}] # Lower score car
    detections3 = [{"label": "person", "score": 0.89, "box": [150, 60, 170, 160]}, {"label": "person", "score": 0.91, "box": [180, 65, 200, 165]}]

    indexer.add_detections(ts1, "Main Gate", detections1, frame_number=10)
    indexer.add_detections(ts2, "Perimeter Fence", detections2, frame_number=25)
    indexer.add_detections(ts3, "Main Gate", detections3, frame_number=30)

    print(f"\nTotal frames indexed: {indexer.get_index_size()}")

    print("\n--- Querying Index ---")

    # Query 1: Find frames containing a 'car'
    car_frames = indexer.query_detections(object_label='car')
    print(f"\nQuery: object_label='car' ({len(car_frames)} results)")
    for frame in car_frames: print(f"  - {frame['timestamp']} at {frame['location']} (Detections: {frame['detected_objects']})")

    # Query 2: Find frames containing a 'person'
    person_frames = indexer.query_detections(object_label='person')
    print(f"\nQuery: object_label='person' ({len(person_frames)} results)")
    for frame in person_frames: print(f"  - {frame['timestamp']} at {frame['location']} (Detections: {frame['detected_objects']})")

    # Query 3: Find 'person' with high confidence
    high_conf_person = indexer.query_detections(object_label='person', min_confidence=0.90)
    print(f"\nQuery: object_label='person', min_confidence=0.90 ({len(high_conf_person)} results)")
    for frame in high_conf_person: print(f"  - {frame['timestamp']} at {frame['location']} (Detections: {frame['detected_objects']})")

    # Query 4: Find 'car' with high confidence (should only match frame 1)
    high_conf_car = indexer.query_detections(object_label='car', min_confidence=0.90)
    print(f"\nQuery: object_label='car', min_confidence=0.90 ({len(high_conf_car)} results)")
    for frame in high_conf_car: print(f"  - {frame['timestamp']} at {frame['location']} (Detections: {frame['detected_objects']})")

    # Query 5: Find frames at 'Main Gate'
    gate_frames = indexer.query_detections(location='Main Gate')
    print(f"\nQuery: location='Main Gate' ({len(gate_frames)} results)")
    for frame in gate_frames: print(f"  - {frame['timestamp']} at {frame['location']}") # Don't print full detections here
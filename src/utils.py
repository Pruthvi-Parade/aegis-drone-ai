# src/utils.py

import re
import datetime

def parse_frame_description(timestamp: datetime.datetime, location: str, description: str) -> dict:
    """
    Parses the raw frame description string to extract structured information.

    Args:
        timestamp: The timestamp of the frame.
        location: The location context for the frame.
        description: The raw text description from the simulator.

    Returns:
        A dictionary containing structured data, e.g.,
        {
            'timestamp': datetime_object,
            'location': 'Gate A',
            'raw_description': 'Frame 101: person loitering at Main Gate',
            'frame_number': 101,
            'detected_objects': ['person'],
            'potential_actions': ['loitering'],
            'vehicle_details': {} # {'type': 'sedan', 'color': 'blue'} later?
        }
        Returns None if parsing fails fundamentally.
    """
    parsed_data = {
        'timestamp': timestamp,
        'location': location,
        'raw_description': description,
        'frame_number': None,
        'detected_objects': [],
        'potential_actions': [],
        'vehicle_details': {} # Placeholder for more detailed analysis
    }

    # 1. Extract Frame Number (using regex)
    match = re.search(r"Frame (\d+):", description)
    if match:
        parsed_data['frame_number'] = int(match.group(1))

    # 2. Identify Known Objects (case-insensitive)
    known_objects = ['person', 'truck', 'suv', 'sedan', 'vehicle', 'dog', 'ford f150'] # Expand this list
    description_lower = description.lower()
    for obj in known_objects:
        if obj in description_lower:
            # Handle specifics like 'Ford F150' -> 'truck' and details
            if obj == 'ford f150':
                if 'truck' not in parsed_data['detected_objects']:
                     parsed_data['detected_objects'].append('truck') # General category
                parsed_data['vehicle_details']['make_model'] = 'Ford F150'
                # Try to find color mentioned nearby
                color_match = re.search(r"(blue|red|black|white|silver|gray)\s+ford f150", description_lower)
                if color_match:
                     parsed_data['vehicle_details']['color'] = color_match.group(1)
            elif obj in ['truck', 'suv', 'sedan']:
                 if 'vehicle' not in parsed_data['detected_objects']:
                     parsed_data['detected_objects'].append('vehicle') # General category
                 parsed_data['detected_objects'].append(obj) # Specific type
                 parsed_data['vehicle_details']['type'] = obj
            else:
                 parsed_data['detected_objects'].append(obj)


    # Basic check: If simulator says "unknown vehicle", mark as vehicle
    if "unknown vehicle" in description_lower and "vehicle" not in parsed_data['detected_objects']:
         parsed_data['detected_objects'].append('vehicle')

    # Remove duplicates just in case
    parsed_data['detected_objects'] = list(set(parsed_data['detected_objects']))


    # 3. Identify Potential Actions/Keywords
    known_actions = ['loitering', 'approaching', 'entering', 'exiting', 'parked', 'moving', 'stopped', 'walking', 'idle']
    for action in known_actions:
        if action in description_lower:
            parsed_data['potential_actions'].append(action)

    # Add a check for 'nothing specific' to potentially ignore frames later?
    if "nothing specific" in description_lower:
         parsed_data['potential_actions'].append('idle') # Or maybe a specific 'no_event' flag?

    parsed_data['potential_actions'] = list(set(parsed_data['potential_actions']))


    return parsed_data

# --- Basic Test ---
if __name__ == "__main__":
    test_data = [
        (datetime.datetime.now(), "Main Gate", "Frame 1: Blue Ford F150 approaching at Main Gate"),
        (datetime.datetime.now(), "Perimeter Fence", "Frame 2: person loitering detected at Perimeter Fence"),
        (datetime.datetime.now(), "Garage", "Frame 3: red SUV parked at Garage"),
        (datetime.datetime.now(), "Loading Dock", "Frame 4: unknown vehicle stopped at Loading Dock"),
        (datetime.datetime.now(), "Main Gate", "Frame 5: person walking dog exiting property at Main Gate"),
        (datetime.datetime.now(), "Garage", "Frame 10: blue Ford F150 parked at Garage"), # Test repeat vehicle
        (datetime.datetime.now(), "Garage", "Frame 11: nothing specific idle at Garage"),
    ]

    print("Testing Data Parser...")
    for ts, loc, desc in test_data:
        parsed = parse_frame_description(ts, loc, desc)
        print(f"\nOriginal: ({ts.strftime('%H:%M:%S')}, {loc}, '{desc}')")
        print(f"Parsed: {parsed}")
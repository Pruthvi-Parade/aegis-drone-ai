# src/simulator.py

import cv2 # OpenCV for video processing
import datetime
import time
import os
from typing import Generator, Tuple, Any # For type hinting

# --- Configuration ---
# Adjust this path if you saved the video elsewhere
VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_video.mp4')
FRAME_SKIP = 5  # Process every Nth frame to speed up simulation (adjust as needed)
LOCATION_CONTEXT = "Main Street Intersection" # Simulate a fixed drone location context

def generate_drone_data(video_path: str = VIDEO_PATH,
                         frame_skip: int = FRAME_SKIP,
                         location: str = LOCATION_CONTEXT
                         ) -> Generator[Tuple[datetime.datetime, str, Any], None, None]:
    """
    Simulates drone data feed by reading frames from a video file.

    Args:
        video_path: Path to the video file.
        frame_skip: Process only every Nth frame.
        location: Simulated location context for the drone/camera.

    Yields:
        tuple: (timestamp, location, frame) where frame is a NumPy array (BGR format).
               Returns None if the video cannot be opened or processing fails.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    print(f"Simulator: Reading video '{os.path.basename(video_path)}'...")
    frame_count = 0
    processed_frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Simulator: Reached end of video or cannot read frame.")
            break # Exit the loop if no frame is returned

        # Apply frame skipping
        if frame_count % frame_skip == 0:
            now = datetime.datetime.now() # Use current time for simulation timestamp
            processed_frame_count += 1
            # Frame is usually in BGR format from OpenCV
            yield (now, location, frame)
            # Simulate some processing delay if needed, otherwise VLM will be the bottleneck
            # time.sleep(0.1) # Optional small delay

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Simulator: Finished processing. Read {frame_count} frames, yielded {processed_frame_count} frames.")


# --- Basic Test ---
if __name__ == "__main__":
    print("Testing Video Simulator...")

    # Check if video exists before running
    if not os.path.exists(VIDEO_PATH):
         print("-" * 30)
         print(f"ERROR: Test video not found at '{VIDEO_PATH}'")
         print("Please download the video from the link in the documentation")
         print("and place it in the 'data/' folder.")
         print("-" * 30)
    else:
        simulator_generator = generate_drone_data(frame_skip=15) # Skip more frames for quick test

        if simulator_generator:
            frame_yield_count = 0
            try:
                for i, (ts, loc, frame_data) in enumerate(simulator_generator):
                    frame_yield_count += 1
                    print(f"Yielded Frame {frame_yield_count}: Timestamp={ts.strftime('%H:%M:%S.%f')[:-3]}, Location='{loc}', Frame Shape={frame_data.shape}")

                    # Optionally display the frame for visual feedback
                    if frame_yield_count <= 5: # Show first 5 yielded frames
                         cv2.imshow(f"Simulator Test Frame {frame_yield_count}", frame_data)
                         if cv2.waitKey(500) & 0xFF == ord('q'): # Wait 500ms, press 'q' to quit display
                             break
                    elif frame_yield_count == 6:
                         print("(Stopping frame display after 5 frames)")
                         cv2.destroyAllWindows() # Close windows after showing a few

                cv2.destroyAllWindows() # Ensure all windows closed at the end

            except Exception as e:
                print(f"An error occurred during simulation test: {e}")
            finally:
                 cv2.destroyAllWindows() # Clean up windows in case of error
                 print(f"\nSimulator test finished. Total frames yielded: {frame_yield_count}")
        else:
             print("Simulator generator could not be created.")
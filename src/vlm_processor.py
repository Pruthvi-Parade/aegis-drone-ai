# src/vlm_processor.py

# No top-level torch import needed here anymore based on previous steps

from PIL import Image
# Use AutoProcessor and AutoModelForObjectDetection for flexibility
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import cv2
import os
import time
from typing import List, Dict, Any

# --- Configuration ---
# Define model ID for object detection
OBJECT_DETECTION_MODEL_ID = "facebook/detr-resnet-50"
# Confidence threshold to filter detections
CONFIDENCE_THRESHOLD = 0.85 # Adjust as needed (0.85 is quite high)

class ObjectDetector:
    def __init__(self, model_id: str = OBJECT_DETECTION_MODEL_ID, threshold: float = CONFIDENCE_THRESHOLD):
        """
        Initializes the Object Detector using a Hugging Face model.
        """
        import torch # Import torch here
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the object detection model and processor."""
        import torch # Import torch here
        if not hasattr(self, 'device') or self.device is None:
             self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"ObjectDetector: Loading model '{self.model_id}' to {self.device}...")
        start_time = time.time()
        try:
            # Use Auto* classes to load appropriate processor and model
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForObjectDetection.from_pretrained(self.model_id).to(self.device)
            print(f"ObjectDetector: Model loaded successfully in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"ObjectDetector: Error loading model '{self.model_id}': {e}")
            # Handle error

    def detect_objects(self, frame_bgr: cv2.typing.MatLike) -> List[Dict[str, Any]]:
        """
        Detects objects in a single image frame.

        Args:
            frame_bgr: The image frame as a NumPy array in BGR format.

        Returns:
            A list of dictionaries, where each dict represents a detected object
            containing 'label', 'score', and 'box' (xmin, ymin, xmax, ymax).
            Returns an empty list if detection fails or no objects above threshold.
        """
        try:
            import torch # Import torch here
        except ImportError:
            print("Error: PyTorch not found")
            return []

        if self.model is None or self.processor is None:
            print("Error: Object Detection Model not loaded.")
            return []

        detected_objects = []
        try:
            # Convert BGR to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Process image
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

            # Perform inference
            with torch.no_grad(): # Disable gradient calculation for inference
                 outputs = self.model(**inputs)

            # Post-process results
            # Target sizes need to be tensors in the shape of [N, 2] where N is the number of images.
            # We only have one image, so N=1. The shape is (height, width).
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device) # (height, width)

            # Convert outputs (bounding boxes and class logits) to final predictions
            results = self.processor.post_process_object_detection(
                outputs, threshold=self.threshold, target_sizes=target_sizes
            )[0] # Get results for the first (and only) image

            # Extract relevant info
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box_coords = [round(i, 2) for i in box.tolist()]
                label_text = self.model.config.id2label[label.item()]
                detected_objects.append({
                    "label": label_text,
                    "score": round(score.item(), 3),
                    "box": box_coords # [xmin, ymin, xmax, ymax]
                })

            return detected_objects

        except Exception as e:
            print(f"ObjectDetector: Error detecting objects in frame: {e}")
            # import traceback # Optional: print full traceback for debugging
            # traceback.print_exc()
            return [] # Return empty list on error

# --- Basic Test ---
if __name__ == "__main__":
    print("Testing Object Detector...")
    # --- Test Setup (same as before) ---
    VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_video.mp4')
    test_frame = None
    # ... (code to load test_frame from VIDEO_PATH as in previous vlm_processor test) ...
    if os.path.exists(VIDEO_PATH):
        cap = cv2.VideoCapture(VIDEO_PATH)
        if cap.isOpened():
            for _ in range(90): # Skip ~3 seconds
                 ret, _ = cap.read()
                 if not ret: break
            ret, test_frame = cap.read()
            if ret: print(f"Loaded test frame, shape: {test_frame.shape}")
            else: print("Could not read frame.")
            cap.release()
        else: print(f"Could not open video: {VIDEO_PATH}")
    else: print(f"Test video not found: {VIDEO_PATH}")

    # --- Run Test ---
    if test_frame is not None:
        print("Initializing ObjectDetector...")
        detector = ObjectDetector(threshold=0.8) # Use slightly lower threshold for testing potentially

        if detector.model:
            print("Detecting objects in test frame...")
            start_detect_time = time.time()
            detections = detector.detect_objects(test_frame)
            detect_duration = time.time() - start_detect_time
            print(f"\n--- Detection Results (took {detect_duration:.2f}s) ---")
            if detections:
                print(f"Found {len(detections)} objects:")
                for obj in detections:
                    print(f"  - Label: {obj['label']}, Score: {obj['score']:.3f}, Box: {obj['box']}")
                 # Draw boxes on the frame for visual verification
                frame_with_boxes = test_frame.copy()
                for obj in detections:
                     box = [int(coord) for coord in obj['box']]
                     cv2.rectangle(frame_with_boxes, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                     cv2.putText(frame_with_boxes, f"{obj['label']} ({obj['score']:.2f})",
                                 (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Object Detection Test", frame_with_boxes)
                print("\nDisplaying frame with detected objects. Press 'q' to close.")
                while True:
                    if cv2.waitKey(100) & 0xFF == ord('q'): break
                cv2.destroyAllWindows()
            else:
                 print("No objects detected above the threshold.")
        else:
             print("Skipping detection as model failed to load.")
    else:
        print("No test frame loaded, skipping detection test.")
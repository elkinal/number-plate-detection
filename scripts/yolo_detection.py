from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path):
        # Load the YOLO model
        self.model = YOLO(model_path)

    def detect(self, frame):
        # Perform object detection
        results = self.model(frame)[0]  # Directly access the first result to reduce indexing
        detections = [
            (obj.xyxy[0].tolist(), obj.conf[0].item(), obj.cls[0].item())  # Extract bbox, confidence, class_id
            for obj in results.boxes
        ]
        return detections

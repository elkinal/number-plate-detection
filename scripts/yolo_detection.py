from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path):
        # Load the YOLO model
        self.model = YOLO(model_path)

    def detect(self, frame):
        # Perform object detection
        results = self.model(frame)
        detections = []
        for obj in results[0].boxes:
            bbox = obj.xyxy[0].tolist()
            confidence = obj.conf[0].tolist()
            class_id = obj.cls[0].tolist()
            detections.append((bbox, confidence, class_id))
        return detections
import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Load a video clip or image
cap = cv2.VideoCapture('../data/TownCentreXVID.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Render detection results on the frame
    results.render()

    # Display the frame with detections
    cv2.imshow('YOLO Object Detection', results.ims[0])

    # Press Q to exit the video display
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from yolo_detection import YOLODetector
from tracking import ObjectTracker
from anomaly_detection import detect_loitering, detect_erratic_movement

# Initialize YOLO detector and tracker
yolo = YOLODetector("yolov10n.pt")
tracker = ObjectTracker()

# Open a file for logging detected anomalies
log_file = open("anomaly_logs.txt", "w")

# Load video
cap = cv2.VideoCapture('../data/TownCentreTRIMMED.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1  # Increment the frame number for each frame
    timestamp = frame_number / fps  # Calculate the timestamp in seconds

    # Object detection
    detections = yolo.detect(frame)

    # Object tracking
    tracks = tracker.track(detections, frame)

    # Apply rule-based anomaly detection
    loitering_objects = detect_loitering(tracks, tracker.track_histories)
    erratic_objects = detect_erratic_movement(tracks, tracker.track_histories)

    # Output and log any detections with timestamps
    if loitering_objects:
        for obj_id in loitering_objects:
            log_message = f"Loitering detected for object {obj_id} at {timestamp:.2f} seconds\n"
            print(log_message.strip())  # Print to console
            log_file.write(log_message)  # Write to log file

    if erratic_objects:
        for obj_id in erratic_objects:
            log_message = f"Erratic movement detected for object {obj_id} at {timestamp:.2f} seconds\n"
            print(log_message.strip())  # Print to console
            log_file.write(log_message)  # Write to log file

    # Visualization: Highlight loitering objects in red
    for track in tracks:
        ltrb = track.to_ltrb()
        track_id = track.track_id
        class_id = int(track.detection[4])
        # Note that opencv uses BGR not RGB
        if track.track_id in loitering_objects:
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 0, 255), 2)
        elif track.track_id in erratic_objects:
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)

        cv2.putText(frame, f'ID: {track.track_id}', (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with anomalies highlighted
    cv2.imshow('YOLOv10 with DeepSORT Tracking and Anomaly Detection', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
log_file.close() 
cv2.destroyAllWindows()
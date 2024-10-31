import cv2
from yolo_detection import YOLODetector
from tracking import ObjectTracker
from anomaly_detection import detect_loitering, detect_erratic_movement

# Initialize YOLO detector and tracker
yolo = YOLODetector("yolov10n.pt")
tracker = ObjectTracker()

# Open a file for logging detected anomalies
with open("anomaly_logs.txt", "w") as log_file:

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

        # Object detection and tracking
        detections = yolo.detect(frame)
        tracks = tracker.track(detections, frame)

        # Anomaly detection
        loitering_objects = detect_loitering(tracks, tracker.track_histories)
        erratic_objects = detect_erratic_movement(tracks, tracker.track_histories)

        # Combined logging and drawing in a single loop
        for track in tracks:
            ltrb = track.to_ltrb()
            track_id = track.track_id
            color = (0, 255, 0)  # Default color: green

            # Log loitering and erratic movement once per object
            if track_id in loitering_objects:
                color = (0, 0, 255)  # Loitering: red
                log_message = f"Loitering detected for object {track_id} at {timestamp:.2f} seconds\n"
                print(log_message.strip())
                log_file.write(log_message)
            elif track_id in erratic_objects:
                color = (255, 0, 0)  # Erratic movement: blue
                log_message = f"Erratic movement detected for object {track_id} at {timestamp:.2f} seconds\n"
                print(log_message.strip())
                log_file.write(log_message)

            # Draw rectangles for all tracked objects
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), color, 2)
            cv2.putText(frame, f'ID: {track_id}', (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame
        cv2.imshow('YOLOv10 with DeepSORT Tracking and Anomaly Detection', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

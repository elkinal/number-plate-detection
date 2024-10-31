from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class ObjectTracker:
    def __init__(self, max_history=30):
        self.tracker = DeepSort(max_age=30, n_init=3)
        self.track_histories = {}  # Dictionary to store track ID and its history
        self.max_history = max_history  # Maximum length of position history per track ID

    def track(self, detections, frame):
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Update history for each track
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # Bounding box in left, top, right, bottom format
            center_position = np.array([(ltrb[0] + ltrb[2]) / 2, (ltrb[1] + ltrb[3]) / 2])  # Center as NumPy array

            # Initialize history if track_id is new
            if track_id not in self.track_histories:
                self.track_histories[track_id] = []

            # Update the track history, ensuring max history length
            history = self.track_histories[track_id]
            if len(history) >= self.max_history:
                history.pop(0)  # Remove oldest position if at max history
            history.append(center_position)  # Add current position

        return tracks

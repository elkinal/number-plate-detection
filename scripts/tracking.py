from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30, n_init=3)
        self.track_histories = {}  # Dictionary to store track ID and its history

    def track(self, detections, frame):
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # Update history for each track
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Bounding box in left, top, right, bottom format
            center_position = np.array([(ltrb[0] + ltrb[2]) / 2, (ltrb[1] + ltrb[3]) / 2])  # Store as NumPy array

            # Update the history for the track
            if track_id not in self.track_histories:
                self.track_histories[track_id] = []
            self.track_histories[track_id].append(center_position)

        return tracks
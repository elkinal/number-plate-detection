import numpy as np
from anomaly_detection import detect_loitering, detect_erratic_movement, detect_unattended_objects

#SYNTHETIC DATA
class SyntheticTrack:
    def __init__(self, track_id, class_id='person', confirmed=True):
        self.track_id = track_id
        self.class_id = class_id
        self.confirmed = confirmed

    def is_confirmed(self):
        return self.confirmed

    @property
    def centroid(self):
        return np.array([100, 100])

tracks = [
    SyntheticTrack(track_id=1, class_id='person'),
    SyntheticTrack(track_id=2, class_id='bag'),
    SyntheticTrack(track_id=3, class_id='person'),
    SyntheticTrack(track_id = 4, class_id = 'person'),
    SyntheticTrack(track_id = 5, class_id = 'car') 
]

# CHATGPT Generated erratic movement: sudden jumps, stops, and random changes in direction
# Wanted a concrete erratic mvoement track history to test
erratic_history = [
    np.array([100, 100]),  # Starting point
    np.array([200, 150]),  # Moves to the right and slightly down
    np.array([210, 300]),  # Sudden vertical movement
    np.array([190, 50]),   # Sharp left and up
    np.array([400, 400]),  # Random large jump to the bottom-right
    np.array([100, 100]),  # Back to the start in a sudden move
    np.array([150, 170]),  # Small move
    np.array([50, 250]),   # Another large jump
    np.array([500, 50]),   # Sharp diagonal movement
]

track_histories = {
    1: [np.array([100, 100]), np.array([100, 100]), np.array([100, 100])],  # Stationary
    2: [np.array([50, 50]), np.array([51, 51]), np.array([52, 52])],  # Moving slowly
    3: [np.array([200, 200]), np.array([10000, 10000]), np.array([0, 0])],  # Erratic movement
    4: [np.array([100,100]), np.array([200,200]), np.array([300,300])], # "Normal" movement
    5: erratic_history
}

# People tracks for unattended object detection
people_tracks = [
    SyntheticTrack(track_id=4, class_id='person')  # Simulating a nearby person that isn't near the object
]

# Test loitering detection
loitering_objects = detect_loitering(tracks, track_histories, min_stationary_time=3)
print(f"Loitering objects: {loitering_objects}")

#should be objects 1, 2 . Currently 'objects' includes people.

# Test erratic movement detection
# Should most definiftely include object 5
erratic_objects = detect_erratic_movement(tracks, track_histories, velocity_threshold=10)
print(f"Erratic movement objects: {erratic_objects}")

# Test unattended objects detection
unattended_objects = detect_unattended_objects(tracks, track_histories, people_tracks, stationary_threshold=3)
print(f"Unattended objects: {unattended_objects}")
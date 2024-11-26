import numpy as np
from tracking import ObjectTracker

def test_object_tracker():
    # Initialize the tracker
    tracker = ObjectTracker(max_history=5)

    # Create dummy detections (x, y, width, height, confidence, class_id)
    detections = [
        [50, 50, 100, 100, 0.9, 0],  # Example detection
        [200, 200, 250, 250, 0.8, 1]
    ]
    frame = np.zeros((500, 500, 3), dtype=np.uint8)  # Dummy frame

    # Run tracking
    tracks = tracker.track(detections, frame)

    # Validate tracking
    assert tracks is not None
    assert len(tracks) > 0  # Ensure tracks are created

    # Check track histories
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            assert track_id in tracker.track_histories
            assert len(tracker.track_histories[track_id]) <= tracker.max_history
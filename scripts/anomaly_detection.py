import numpy as np

def detect_loitering(tracks, track_histories, min_stationary_time=100):
    loitering_objects = []
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Get the history of the track
            history = track_histories.get(track_id, [])

            # Check if the object has been stationary for enough frames
            if len(history) >= min_stationary_time:
                # Calculate the total movement
                movement = np.linalg.norm(np.diff(history, axis=0), axis=1).sum()
                if movement < 50:  # movement threshold for detecting loitering
                    loitering_objects.append(track_id)

    return loitering_objects

def detect_erratic_movement(tracks, track_histories, velocity_threshold=10):
    erratic_objects = []
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Get the history of the track
            history = track_histories.get(track_id, [])

            # Check if there is enough history to calculate velocity
            if len(history) >= 5:
                velocities = [np.linalg.norm(history[i] - history[i-1]) for i in range(1, len(history))]
                if np.std(velocities) > velocity_threshold:  # Standard deviation for erratic movement
                    erratic_objects.append(track_id)

    return erratic_objects

def detect_unattended_objects(tracks, track_histories, people_tracks, stationary_threshold=150):
    class_ids_that_are_objects = ['bag']
    unattended_objects = []
    for track in tracks:
        if track.is_confirmed() and track.class_id in class_ids_that_are_objects:  
            # Can change this later to a LLM pipeline such as gemini that determines whether the detected classes can be classified as unattended objects,
            # Or if they are something else, like a person or animal.
            # But this would mean that there has to be essentially a two stage process
            # In which we get all the possible track class_id's from the video clip, and then sort based on LLM which are unattended objects and which are not.
            track_id = track.track_id
            history = track_histories.get(track_id, [])

            # Check if the object is stationary
            if len(history) >= stationary_threshold:
                movement = np.linalg.norm(np.diff(history, axis=0), axis=1).sum()
                if movement < 10:  # movement threshold for stationary objects
                    # Check for nearby people
                    nearby_people = [p for p in people_tracks if np.linalg.norm(p.centroid - track.centroid) < 100]
                    if not nearby_people:
                        unattended_objects.append(track_id)

    return unattended_objects
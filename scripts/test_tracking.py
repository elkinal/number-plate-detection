import unittest
import numpy as np
from unittest.mock import MagicMock, patch

class TestObjectTracker(unittest.TestCase):
    
    def setUp(self):
        self.tracker = ObjectTracker(max_history=5)  
        self.fake_detections = [
            [100, 50, 150, 100],  
            [200, 120, 250, 170]
        ]
        self.fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  

    @patch('your_module.DeepSort.update_tracks')
    def test_track_updates_history(self, mock_update_tracks):
        mock_track = MagicMock()
        mock_track.track_id = 1
        mock_track.is_confirmed.return_value = True
        mock_track.to_ltrb.return_value = [100, 50, 150, 100]
        mock_update_tracks.return_value = [mock_track]

        tracks = self.tracker.track(self.fake_detections, self.fake_frame)

        self.assertIn(1, self.tracker.track_histories)
        history = self.tracker.track_histories[1]
        self.assertEqual(len(history), 1)  
        self.assertTrue(np.allclose(history[0], np.array([125, 75]))) 

    def test_max_history_limit(self):
        self.tracker.tracker.update_tracks = MagicMock(return_value=[
            MagicMock(track_id=1, is_confirmed=MagicMock(return_value=True), to_ltrb=MagicMock(return_value=[100, 50, 150, 100]))
        ])

        for _ in range(10):  
            self.tracker.track(self.fake_detections, self.fake_frame)

        self.assertEqual(len(self.tracker.track_histories[1]), self.tracker.max_history)

    def test_unconfirmed_tracks_not_added(self):
        unconfirmed_track = MagicMock(track_id=2, is_confirmed=MagicMock(return_value=False))
        self.tracker.tracker.update_tracks = MagicMock(return_value=[unconfirmed_track])

        tracks = self.tracker.track(self.fake_detections, self.fake_frame)

        self.assertNotIn(2, self.tracker.track_histories)

if __name__ == '__main__':
    unittest.main()

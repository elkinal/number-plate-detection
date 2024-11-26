import os
import cv2
import numpy as np
from feature_extraction import compute_hog, extract_features_from_images

def test_compute_hog():
    # Create a dummy grayscale image
    image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    # Compute HOG features
    features = compute_hog(image)

    # Check if features are computed and non-empty
    assert features is not None
    assert len(features) > 0

def test_extract_features_from_images(tmpdir):
    # Create temporary images
    test_images = []
    for i in range(3):
        img_path = os.path.join(tmpdir, f"test_img_{i}.png")
        dummy_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(img_path, dummy_image)
        test_images.append(img_path)

    # Extract features from images
    features = extract_features_from_images(test_images)

    # Validate feature extraction
    assert len(features) == 3  # Ensure we have features for all images
    for feature in features:
        assert feature is not None
        assert len(feature) > 0
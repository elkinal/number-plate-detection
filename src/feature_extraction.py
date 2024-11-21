import cv2
import numpy as np

def compute_hog(image, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    """
    Compute HOG features for an image.
    Args:
        image (numpy.ndarray): Input image (grayscale).
        cell_size (tuple): Size of HOG cell.
        block_size (tuple): Size of HOG block (in cells).
        nbins (int): Number of orientation bins.
    Returns:
        numpy.ndarray: HOG feature vector.
    """
    hog = cv2.HOGDescriptor(
        _winSize=(image.shape[1] // cell_size[1] * cell_size[1],
                  image.shape[0] // cell_size[0] * cell_size[0]),
        _blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
        _blockStride=(cell_size[1], cell_size[0]),
        _cellSize=(cell_size[1], cell_size[0]),
        _nbins=nbins
    )
    return hog.compute(image).flatten()

def extract_features_from_images(image_paths, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    """
    Extract HOG features for multiple images.
    Args:
        image_paths (list): List of image paths.
    Returns:
        list: List of HOG feature vectors.
    """
    features = []
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        features.append(compute_hog(image, cell_size, block_size, nbins))
    return features

def save_features(features, output_path):
    """
    Save extracted features to a file.
    Args:
        features (list): List of feature vectors.
        output_path (str): Path to save the features.
    """
    np.save(output_path, np.array(features))

def load_features(feature_file):
    """
    Load saved features from a file.
    Args:
        feature_file (str): Path to the saved feature file.
    Returns:
        numpy.ndarray: Loaded feature vectors.
    """
    return np.load(feature_file)

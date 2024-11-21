import cv2
import numpy as np
import joblib
from feature_extraction import compute_hog
#this file predicts on new images using the trained model
def load_model(model_path):

    return joblib.load(model_path)

def preprocess_image_for_prediction(image_path, target_size, grayscale=True):
    """
    Preprocess a single image for prediction.
    Args: image_path (str): Path to the image file.
    Returns: Preprocessed image.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
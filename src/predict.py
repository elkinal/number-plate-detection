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
    Returns: Preprocessed image.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def predict_image(image_path, model, target_size, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    """
    Predict if a person is present in the image.
    """
    image = preprocess_image_for_prediction(image_path, target_size)
    features = compute_hog(image, cell_size, block_size, nbins)
    prediction = model.predict([features])
    return "Person Detected" if prediction[0] == 1 else "No Person Detected"

def predict_batch(image_paths, model, target_size, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    """
    Predict for a batch of images.
    """
    predictions = []
    for path in image_paths:
        predictions.append(predict_image(path, model, target_size, cell_size, block_size, nbins))
    return predictions
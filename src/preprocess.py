import cv2
import numpy as np
import os
#this file has image preprocessing functions

def load_image(image_path):
    """
    Load an image from the given path.
    """
    return cv2.imread(image_path)

def resize_image(image, target_size):
    """
    Resize an image to the target size.
    """
    return cv2.resize(image, target_size)

def normalize_image(image):
    """
    Normalize image pixel values to the range [0, 1].
    """
    return image / 255.0

def convert_to_grayscale(image):
    """
    Convert an image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def preprocess_image(image_path, target_size, grayscale=True):
    """
    Full preprocessing pipeline for a single image.
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size as (width, height).
        grayscale (bool): Whether to convert the image to grayscale.
    """
    image = load_image(image_path)
    image = resize_image(image, target_size)
    if grayscale:
        image = convert_to_grayscale(image)
    image = normalize_image(image)
    return image

def save_preprocessed_image(image, output_path):
    """
    Save a preprocessed image to disk.
    """
    cv2.imwrite(output_path, (image * 255).astype(np.uint8))

def preprocess_images_in_folder(input_folder, output_folder, target_size, grayscale=True):
    """
    Preprocess all images in a folder.
    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder for preprocessed images.
        target_size (tuple): Target size as (width, height).
        grayscale (bool): Whether to convert images to grayscale.
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        if os.path.isfile(input_path):
            preprocessed_image = preprocess_image(input_path, target_size, grayscale)
            save_preprocessed_image(preprocessed_image, output_path)
def resize_image(image, target_size):
    """
    Resize an image to the target size.
    """
    return cv2.resize(image, target_size)
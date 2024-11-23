import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from feature_extraction import extract_features_from_images
#this file trains the SVM model using extracted HOG features


def load_data(image_folder):
    """
    Load image paths and labels.
    """
    image_paths, labels = [], []
    for label, subfolder in enumerate(os.listdir(image_folder)):
        subfolder_path = os.path.join(image_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for image_name in os.listdir(subfolder_path):
                image_paths.append(os.path.join(subfolder_path, image_name))
                labels.append(label)
    return image_paths, labels

def train_svm(features, labels, kernel="linear", C=1.0):
    """
    Train an SVM model.
    """
    model = SVC(kernel=kernel, C=C)
    model.fit(features, labels)
    return model

def save_model(model, model_path):
    """
    Save the trained model.
    """
    joblib.dump(model, model_path)

def main(image_folder, model_path, target_size=(128, 128)):
    """
    Main training pipeline.
    """

    image_paths, labels = load_data(image_folder)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42)

    train_features = extract_features_from_images(train_paths)
    test_features = extract_features_from_images(test_paths)

    model = train_svm(train_features, train_labels)

    # Evaluate model
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")
    save_model(model, model_path)

if __name__ == "__main__":
    main(image_folder="data/train", model_path="models/svm_model.pkl")

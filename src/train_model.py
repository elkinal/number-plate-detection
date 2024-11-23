import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from feature_extraction import extract_features_from_images
#this file trains the SVM model using extracted HOG features


def load_data_from_yolo(images_path, labels_path):
    """
    Load image paths and YOLO-style labels.
    """
    image_paths, labels = [], []
    for image_file in os.listdir(images_path):
        if image_file.endswith(".png"):
            label_file = os.path.join(labels_path, image_file.replace(".png", ".txt"))
            if os.path.exists(label_file):
                image_paths.append(os.path.join(images_path, image_file))
                with open(label_file, "r") as f:
                    labels.append(f.read())
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

def main(image_folder, model_path, target_size=(128, 128), use_cropped=True):
    """
    Main training pipeline with options for cropped data.
    """
    if use_cropped:
        train_features = extract_features_from_crops(train_paths)
        test_features = extract_features_from_crops(test_paths)
    else:
        train_features = extract_features_from_images(train_paths)
        test_features = extract_features_from_images(test_paths)

    model = train_svm(train_features, train_labels)
    accuracy = accuracy_score(test_labels, model.predict(test_features))
    print(f"Model Accuracy: {accuracy:.2f}")
    save_model(model, model_path)

def extract_features_from_crops(image_paths):
    """
    Extract HOG features from cropped images.
    """
    features = []
    for image_path in image_paths:
        cropped_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        features.append(compute_hog(cropped_image))
    return features


from sklearn.model_selection import GridSearchCV

def tune_svm_hyperparameters(features, labels):
    """
    Tune SVM hyperparameters using grid search.
    """
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid_search = GridSearchCV(SVC(), param_grid, cv=3)
    grid_search.fit(features, labels)
    return grid_search.best_estimator_


def save_predictions(image_paths, predictions, output_dir):
    """
    Save images with predictions for review.
    """
    os.makedirs(output_dir, exist_ok=True)
    for path, pred in zip(image_paths, predictions):
        img = cv2.imread(path)
        label = "Person" if pred == 1 else "No Person"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        output_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(output_path, img)

if __name__ == "__main__":
    main(image_folder="data/train", model_path="models/svm_model.pkl")

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from feature_extraction import extract_features_from_images

#this file evaluates the trained SVM model on a test dataset


def load_model(model_path):
    """
    Load the trained SVM model.
    """
    return joblib.load(model_path)

def evaluate_model(model, test_features, test_labels):
    """
    Evaluate the model on test data.
    """
    predictions = model.predict(test_features)
    return accuracy_score(test_labels, predictions), precision_score(test_labels, predictions), recall_score(test_labels, predictions)


def main(test_image_folder, model_path):
    """
    Main evaluation pipeline.
    """
    test_paths, test_labels = load_data(test_image_folder)
    test_features = extract_features_from_images(test_paths)
    model = load_model(model_path)
    accuracy, precision, recall = evaluate_model(model, test_features, test_labels)
    print(f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}")


if __name__ == "__main__":
    main(test_image_folder="data/test", model_path="models/svm_model.pkl")

import os
import numpy as np
import joblib
from sklearn.svm import SVC
from train_model import train_svm, save_model, load_data_from_yolo

def test_train_svm():
    # Dummy features and labels
    features = np.random.rand(10, 5)  # 10 samples, 5 features each
    labels = np.random.randint(0, 2, 10)  # Binary labels

    # Train the SVM model
    model = train_svm(features, labels)

    # Validate model training
    assert model is not None
    assert isinstance(model, SVC)

def test_save_model(tmpdir):
    # Dummy model
    model = SVC()
    model.fit(np.random.rand(10, 5), np.random.randint(0, 2, 10))  # Train dummy model

    # Save the model
    model_path = os.path.join(tmpdir, "test_model.pkl")
    save_model(model, model_path)

    # Validate saved model file
    assert os.path.exists(model_path)
    loaded_model = joblib.load(model_path)
    assert isinstance(loaded_model, SVC)

def test_load_data_from_yolo(tmpdir):
    # Create dummy image and label files
    images_path = tmpdir.mkdir("images")
    labels_path = tmpdir.mkdir("labels")

    for i in range(3):
        img_path = images_path.join(f"image_{i}.png")
        label_path = labels_path.join(f"image_{i}.txt")
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.25 0.25\n")  # Dummy YOLO label
        img_path.write("dummy image content")  # Placeholder content

    # Load the data
    image_paths, labels = load_data_from_yolo(images_path, labels_path)

    # Validate data loading
    assert len(image_paths) == 3
    assert len(labels) == 3
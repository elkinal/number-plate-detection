import os

# Paths
dataset_path = "data"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")
yolo_labels_path = os.path.join(dataset_path, "yolo_labels")
models_path = "models"
yolo_model = os.path.join(models_path, "yolov10n.pt")

# Train-test split ratio
test_size = 0.2
random_state = 42


os.makedirs(yolo_labels_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)
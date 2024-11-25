from scripts.read_plates import convert_to_yolo
from scripts.feature_extraction import extract_features_from_images
from scripts.train_model import main as train_model
from scripts.evaluate_model import main as evaluate_model
from config import images_path, labels_path, yolo_labels_path, test_size, random_state
import os
from sklearn.model_selection import train_test_split
import shutil

def split_dataset():
    """
    Split dataset into training and validation sets. prompt: how do i Split dataset
    """
    # List all images and labels
    images = [f for f in os.listdir(images_path) if f.endswith(".png")]
    labels = [f.replace(".png", ".txt") for f in images]

    # Split into train and validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=test_size, random_state=random_state
    )

    # Organize directories
    for subset, img_list, lbl_list in zip(
        ["train", "val"], [train_images, val_images], [train_labels, val_labels]
    ):
        img_path = os.path.join(images_path, subset)
        lbl_path = os.path.join(labels_path, subset)
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(lbl_path, exist_ok=True)

        # Move images and labels
        for img, lbl in zip(img_list, lbl_list):
            shutil.copy(os.path.join(images_path, img), os.path.join(img_path, img))
            shutil.copy(os.path.join(yolo_labels_path, lbl), os.path.join(lbl_path, lbl))

    return len(train_images), len(val_images)

def log_summary(train_count, val_count):
    """
    Print a summary of the dataset and pipeline results.
    """
    print("\nPipeline Summary:")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print("Pipeline execution completed successfully!")

def run_pipeline():
    # Step 1: Convert XML labels to YOLO format
    print("Converting XML labels to YOLO format...")
    for xml_file in os.listdir(labels_path):
        if xml_file.endswith(".xml"):
            image_file = os.path.join(images_path, xml_file.replace(".xml", ".png"))
            if os.path.exists(image_file):
                convert_to_yolo(
                    os.path.join(labels_path, xml_file), yolo_labels_path, image_file
                )

    # Step 2: Split the dataset
    print("Splitting dataset into train and validation sets...")
    train_count, val_count = split_dataset()

    # Step 3: Train the model
    print("Training the model...")
    train_model(images_path, labels_path)

    # Step 4: Evaluate the model
    print("Evaluating the model...")
    evaluate_model(images_path, labels_path)

    # Log summary
    log_summary(train_count, val_count)

if __name__ == "__main__":
    run_pipeline()
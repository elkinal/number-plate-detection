from scripts.read_plates import convert_to_yolo
from scripts.feature_extraction import extract_features_from_images
from scripts.train_model import main as train_model
from scripts.evaluate_model import main as evaluate_model
from config import images_path, labels_path, yolo_labels_path, test_size, random_state

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
    # Add dataset splitting logic here if not already handled

    # Step 3: Train the model
    print("Training the model...")
    train_model(images_path, labels_path)

    # Step 4: Evaluate the model
    print("Evaluating the model...")
    evaluate_model(images_path, labels_path)

if __name__ == "__main__":
    run_pipeline()
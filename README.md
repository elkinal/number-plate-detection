# AI Practicum: Number Plate Detection

This project involves developing an object detection pipeline using YOLO and other deep learning models. The system processes still images to detect number plates and classify them using a machine learning model.

Initially, the project focused on rule-based detection, but it now incorporates supervised machine learning techniques for classification. Libraries are utilized for object detection and data processing.

---

## Demo Script

### Overview
The demo will showcase the core components of the AI-Driven Number Plate Detection project. Each team member will demonstrate a specific aspect of the pipeline using non-trivial test cases and outputs.

### Agenda

1. **Preprocessing and YOLO-based Detection** (Alexey Elkin):
   - Prepares the dataset, splits it into training and testing sets, and runs a YOLO model to detect and crop number plates.
   - Demonstrates a functional pipeline that identifies number plates from car images and saves them to the `/number_plates/` directory.
   - **Expected Output:** Visualized images with bounding boxes highlighting detected number plates, along with correctly cropped images saved in the designated folder.

2. **Feature Extraction and SVM Training** (Ruji Wang):
   - Extracts HOG features from cropped images and trains an SVM model to classify the detected number plates.
   - Demonstrates feature vector generation and the SVM training process, showing evaluation metrics on test data.
   - **Expected Output:** Logs showing feature vector sizes, training progress, and SVM evaluation metrics (e.g., accuracy, precision, recall).

3. **Output Visualization** (Edward):
   - Runs the output visualization module to display processed images with bounding boxes and classification results.
   - **Expected Output:** Visualized images with bounding boxes and classifications shown.

---

## Setup Instructions

This project is tested with Python 3.11.5.

### Steps:

1. Clone the repository and navigate into the project directory:
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```

2. Set up a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the project:
   ```bash
   python scripts/main.py
   ```

5. Test the classification module:
   ```bash
   python scripts/test_classification.py
   ```

### Note
If you encounter an SSL certificate verification error, you may need to manually install SSL certificates. For macOS, run:
```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```
Replace `3.x` with your Python version (e.g., `3.11`).

---

## Performance Considerations

The system processes still images instead of real-time video due to performance limitations. By focusing on batch processing, we ensure efficient and accurate detection and classification.

---

## Demo Roles

- **Edward:** Demonstrates the `output_visualization.py` script, showing processed images with bounding boxes and classification results. Runs:
  ```bash
  python output_visualization.py
  ```

- **Ruji Wang:** Runs the `classification_module.py` script or `pytest` to validate the classification model. Expected output includes logs of feature vectors, training progress, and test results:
  ```bash
  python classification_module.py
  pytest test_classification.py
  ```

- **Alexey Elkin:** Demonstrates the `main.py` script output, showing processed images with bounding boxes around detected number plates:
  ```bash
  python main.py
  ```

---

## License

This project is licensed under the MIT License.

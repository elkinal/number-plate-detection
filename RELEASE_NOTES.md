# Release Notes: AI-Driven Event Detection

## Group Name: ERA  

### Team Members:
- Ruji Wang (NetID: rw498)  
- Alexey Elkin (NetID: ae339)

---

## Demo Script

### Overview:
The demo will showcase the core components of the AI-Driven Event Detection project. Each team member will demonstrate a specific aspect of the pipeline using non-trivial test cases and outputs.

#### What to Expect:
1. **Preprocessing and YOLO-based Detection** (Alexey Elkin):  
   - Prepares the dataset, splits it into training and test sets, and runs a YOLO model to detect and crop number plates.  
   - Demonstrates a functional pipeline that identifies number plates from car images and saves them to the `/number_plates/` directory.  
   - **Expected Output:** Visualized images with bounding boxes highlighting detected number plates, and correctly cropped images saved in the designated folder.

2. **Feature Extraction and SVM Training** (Ruji Wang):  
   - Extracts HOG features from the cropped images and trains an SVM model to classify the detected number plates.  
   - Demonstrates feature vectors generated from the cropped plates and the SVM training process, showing the resulting accuracy on test data.  
   - **Expected Output:** Logs showing feature vector sizes, training progress, and SVM evaluation metrics (e.g., accuracy, precision, recall).

3. **Object Tracking with YOLO Integration** (Ruji Wang):  
   - Integrates YOLO detections with the tracking module to maintain consistent IDs for objects across frames in a video.  
   - Demonstrates tracked objects with positional histories.  
   - **Expected Output:** Tracked object IDs and their positions visualized frame-by-frame.

---

### Instructions for TA Reproduction:
1. Clone the repository and navigate to the main project directory:
   ```bash
   git clone <repository-url>
   cd scripts
   
2. Run the jupyter notebook and see the visual outputs of the program

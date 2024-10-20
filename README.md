# AI Practicum: Trajectory Based Anomaly Detection

This project involves building an object detection pipeline using YOLO and other deep learning models, processing real-time video feeds or video clips for object detection, then using the trajectories of detected objects, to perform trajectory based anomaly detection.

Currently the plan is rule-based systems at the trajectory based anomaly detection step but as the project evolves, we are going to use machine learning techniques in supervised and unsupervised learning.

Libraries are used for the object detection, and obtaining the trajectories.

## Setup Instructions

This project is tested to be working with Python 3.11.5

1. Clone the repo and navigate into the directory:


2. Set up the Python environment:


```python3 -m venv venv ```
Then, 

``` source venv/bin/activate  # Linux/Mac ```

``` .\venv\Scripts\activate   # Windows ```



3. Install dependencies:

``` pip install -r requirements.txt ```


4. Run what we have so far 
 (Assuming you are in the root of project directory)
 
```python scripts/main.py```

Test script
```python scripts/test_anomaly_detection.py```

## Notes
If you get an SSL certificate verify error,
You may need to manually install SSL certificates in order to ensure Python can validate them.

Open a terminal and run 
/Applications/Python\ 3.x/Install\ Certificates.command

Replace .x with the specific Python version. If your python version is 
3.11.5, then the command would be 
/Applications/Python\ 3.11/Install\ Certificates.command

## Extra Information

The original plan for this project was to have anomaly detection working on a live feed, but with the work that we have done so far, we realized that doing video stuff on our local machines is going to be annoying. We noticed that it was basically processing things at 1-2 frame per second (Macbook Pro m2 pro ), even with the smallest version of YOLO v10 (Yolo v10n) combined with DeepSort. The DeepSort is what really made the frames drop, as before that the framerate in the output view was essentially the same as the original video. We expect that trying to incorporate live feeds would be infeasible due to how long it takes just to process a prerecorded video (the delay would be too much)

## Methods
We used ChatGPT to assist in creating small portions of the code and the test cases that we use to edit the program.

## Demo Script

Edward: Demonstrate the anomaly_detection.py and the related test, showing that bounding boxes show up on the video output signifying various anomalies detected.
Ruji: To run the code, execute python reporting_module.py (if applicable), or alternatively, focus on testing by running pytest test_reporting_module.py. This will test event logging, alert notifications, and performance metrics (precision, recall, latency). The expected output should show all tests passing with real-time alerts, event logs, and metrics printed to the terminal.  

## License
MIT License

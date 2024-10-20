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


## License
MIT License

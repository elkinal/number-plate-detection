import pytest
from reporting_module import ReportingModule
from io import StringIO
import sys

# Helper function to capture printed output
def capture_output(func, *args):
    captured_output = StringIO()
    sys.stdout = captured_output
    func(*args)
    sys.stdout = sys.__stdout__
    return captured_output.getvalue()

@pytest.fixture
def reporter():
    return ReportingModule()

def test_log_event(reporter):
    # Capture the output of the log_event function
    output = capture_output(reporter.log_event, "Object Detection", "Person detected in frame 1")
    
    # Check if the event was logged and printed correctly
    assert "Object Detection: Person detected in frame 1" in output

def test_display_alert(reporter):
    # Capture the output of the display_alert function
    output = capture_output(reporter.display_alert, "Suspicious activity detected")
    
    # Check if the alert message was printed correctly
    assert "[ALERT] Suspicious activity detected" in output

def test_display_metrics(reporter):
    # Capture the output of the display_metrics function
    output = capture_output(reporter.display_metrics, 0.92, 0.88, 0.34)
    
    # Check if the metrics were printed correctly
    assert "Precision: 0.92" in output
    assert "Recall: 0.88" in output
    assert "Latency: 0.34 seconds" in output

def test_output_event_log(reporter):
    # Log some events first
    reporter.log_event("Object Detection", "Person detected in frame 1")
    reporter.log_event("Anomaly Detection", "Unattended bag detected in frame 2")
    
    # Capture the output of the event log
    output = capture_output(reporter.output_event_log)
    
    # Check if the log output includes the logged events
    assert "Object Detection: Person detected in frame 1" in output
    assert "Anomaly Detection: Unattended bag detected in frame 2" in output
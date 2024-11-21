import time
from datetime import datetime

class ReportingModule:
    def __init__(self):
        self.event_log = []

    def log_event(self, event_type, details):
        """Log events such as anomaly detection or object detection"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event = f"{timestamp} - {event_type}: {details}"
        self.event_log.append(event)
        print(event)

    def display_alert(self, alert_message):
        """Display real-time alerts for important events"""
        print(f"\n[ALERT] {alert_message}")

    def display_metrics(self, precision, recall, latency):
        """Display detection statistics and performance metrics"""
        print("\n[PERFORMANCE METRICS]")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Latency: {latency:.2f} seconds")

    def output_event_log(self):
        """Output all recorded events."""
        print("\n[EVENT LOG]")
        for event in self.event_log:
            print(event)


if __name__ == "__main__":
    reporter = ReportingModule()


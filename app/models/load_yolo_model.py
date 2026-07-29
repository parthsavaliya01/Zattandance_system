#app/models/load_yolo_model.py
from ultralytics import YOLO

#Load yolo model here

import threading

_lock = threading.Lock()

def load_yolo_model():
    with _lock:
        yolo_model = YOLO("app/models/best.pt")
        # Safely force fuse inside the lock to prevent multi-threading crashes
        yolo_model.fuse()
        return yolo_model
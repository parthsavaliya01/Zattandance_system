#app/models/load_yolo_model.py
import logging
from ultralytics import YOLO

#Load yolo model here

import threading

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_device_logged = False

def load_yolo_model():
    global _device_logged
    with _lock:
        yolo_model = YOLO("app/models/best.pt")
        # Safely force fuse inside the lock to prevent multi-threading crashes
        yolo_model.fuse()
        if not _device_logged:
            _device_logged = True
            import torch
            if torch.cuda.is_available():
                logger.info(f"[GPU-CHECK] YOLO (torch) running on GPU · {torch.cuda.get_device_name(0)}")
            else:
                logger.error(
                    "[GPU-CHECK] YOLO (torch) is NOT on the GPU (torch.cuda.is_available()=False). "
                    "Tracking will be slow and attendance will be missed. Check `nvidia-smi` on the host."
                )
        return yolo_model

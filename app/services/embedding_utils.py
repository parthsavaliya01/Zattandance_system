# app/services/embedding_utils.py
import logging
import time
import traceback
import uuid
import numpy as np
from app.services.gpu_worker import GPU_RESULT_MAP, GPU_TASK_QUEUE

logger = logging.getLogger(__name__)

def extract_face_embedding(frame, yolo_area: dict = None, timeout=7.0):
    try:
        task_id = uuid.uuid4().hex
        GPU_TASK_QUEUE.put((task_id, {
            "type": "frame",
            "data": frame,           # full BGR frame
            "yolo_area": yolo_area   # YOLO bbox to filter results
        }))

        start = time.time()
        while task_id not in GPU_RESULT_MAP:
            if time.time() - start > timeout:
                return []
            time.sleep(0.002)

        result = GPU_RESULT_MAP.pop(task_id)
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        logger.error(traceback.format_exc())
        return []
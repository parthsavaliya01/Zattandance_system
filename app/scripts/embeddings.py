# app/scripts/embeddings.py
import logging
import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict

# Re-use the InsightFace detector singleton and alignment helper from gpu_worker
# so the detector model is only loaded once across the whole process.
from app.services.gpu_worker import align_face_112, get_insight_app

logger = logging.getLogger(__name__)

from app.core.config import settings

# Minimum InsightFace confidence to accept a detection for storage
_MIN_CONF = getattr(settings, "ENROLL_MIN_CONF", 0.65)



def extract_embedding_from_image(
    face_path: str,
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Detect the best face in an image file, landmark-align it to 112×112,
    and return it as an RGB numpy array together with its facial_area dict.

    The returned crop is fed into generate_face_variants() and then embedded
    by GPUWorker ("face" task type) — NO embedding is computed here.
    This function is purely detection + alignment.

    Returns:
        (face_rgb_112, facial_area_dict)  on success
        (None, None)                       on any failure
    """
    try:
        if not os.path.exists(face_path):
            logger.error(f"[extract_embedding_from_image] File not found: {face_path}")
            return None, None

        img_bgr = cv2.imread(face_path)
        if img_bgr is None:
            logger.error(f"[extract_embedding_from_image] cv2 returned None: {face_path}")
            return None, None

        # ── Detect with InsightFace SCRFD ─────────────────────────────────
        app = get_insight_app()
        detections = app.get(img_bgr)

        if not detections:
            logger.warning(f"[extract_embedding_from_image] No face detected: {face_path}")
            return None, None

        # Pick the highest-confidence detection
        best = max(detections, key=lambda d: float(d.det_score))
        score = float(best.det_score)

        if score < _MIN_CONF:
            logger.warning(
                f"[extract_embedding_from_image] Low confidence "
                f"({score:.2f}) for: {face_path}"
            )
            return None, None

        # ── 5-point landmarks → align to 112×112 ──────────────────────────
        # InsightFace kps order: left_eye, right_eye, nose, mouth_left, mouth_right
        kps = best.kps.astype(np.float32)  # (5, 2)

        aligned_bgr = align_face_112(img_bgr, kps)
        face_rgb    = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        # ── Build facial_area dict (same schema as old DeepFace output) ───
        x1, y1, x2, y2 = best.bbox.astype(int)
        facial_area = {
            "x": int(x1),
            "y": int(y1),
            "w": int(x2 - x1),
            "h": int(y2 - y1),
            "left_eye":    kps[0].tolist(),
            "right_eye":   kps[1].tolist(),
            "nose":        kps[2].tolist(),
            "mouth_left":  kps[3].tolist(),
            "mouth_right": kps[4].tolist(),
        }

        logger.info(
            f"[extract_embedding_from_image] ✅ Aligned face from: {face_path} "
            f"(confidence={score:.2f})"
        )
        return face_rgb, facial_area

    except Exception as e:
        logger.exception(f"❌ [extract_embedding_from_image] Error for '{face_path}': {e}")
        return None, None
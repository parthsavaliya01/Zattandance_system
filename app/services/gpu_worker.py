# app/services/gpu_worker.py
import threading
import queue
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Shared task / result structures  (public API is UNCHANGED)
# ─────────────────────────────────────────────────────────────
GPU_TASK_QUEUE = queue.Queue(maxsize=32)
GPU_RESULT_MAP = {}

# ─────────────────────────────────────────────────────────────
# 5-point reference landmarks for canonical 112×112 output
# Order: left_eye, right_eye, nose, mouth_left, mouth_right
# ─────────────────────────────────────────────────────────────
_REF_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# Minimum InsightFace detection confidence to accept a face
# Lowered for side-profiles (which often score ~0.50 on SCRFD)
_MIN_CONF = 0.45

# ─────────────────────────────────────────────────────────────
# Singleton InsightFace detector
# ─────────────────────────────────────────────────────────────
_INSIGHT_APP: FaceAnalysis = None

def get_insight_app() -> FaceAnalysis:
    """
    Lazy-load InsightFace SCRFD detector (pure ONNX, zero TF dependency).
    buffalo_sc = SCRFD-500M + 5-point landmark head — fast and accurate.
    buffalo_l  = heavier, slightly more accurate (swap name if needed).
    Singleton — safe to call from multiple places; only loads once.
    """
    global _INSIGHT_APP
    if _INSIGHT_APP is None:
        _INSIGHT_APP = FaceAnalysis(
            name="buffalo_sc",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        _INSIGHT_APP.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("[InsightFace] buffalo_sc detector loaded.")
    return _INSIGHT_APP


# ─────────────────────────────────────────────────────────────
# Shared utility functions (importable by embeddings.py)
# ─────────────────────────────────────────────────────────────

def align_face_112(image_bgr: np.ndarray, landmarks_5pt: np.ndarray) -> np.ndarray:
    """
    Warp a BGR image to canonical 112×112 using 5-point affine transform.
    Uses cv2.estimateAffinePartial2D — no skimage needed.

    Args:
        image_bgr:      Any-size BGR frame or crop.
        landmarks_5pt:  float32 (5, 2) —
                        [left_eye, right_eye, nose, mouth_left, mouth_right]
    Returns:
        112×112 BGR aligned face.
    """
    M, _ = cv2.estimateAffinePartial2D(
        landmarks_5pt, _REF_POINTS, method=cv2.RANSAC
    )
    if M is None:
        # Fallback: plain resize if transform estimation fails
        logger.warning("[align_face_112] estimateAffinePartial2D failed — plain resize used.")
        return cv2.resize(image_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    return cv2.warpAffine(image_bgr, M, (112, 112), borderValue=0.0)


def preprocess_adaface(aligned_bgr: np.ndarray) -> np.ndarray:
    """
    BGR 112×112 → ONNX input tensor (1, 3, 112, 112) float32.
    Official AdaFace repo: BGR input, mean=0.5, std=0.5.

    ⚠️  If cosine scores look near-zero or negative, your ONNX was
        exported from an RGB pipeline — uncomment the [RGB-SWAP] line.
    """
    img = aligned_bgr.astype(np.float32) / 255.0    # [0, 1]
    img = (img - 0.5) / 0.5                          # [−1, 1]
    # img = img[:, :, ::-1].copy()                   # [RGB-SWAP] uncomment if needed
    img = img.transpose(2, 0, 1)                     # HWC → CHW
    return np.expand_dims(img, 0).astype(np.float32) # (1, 3, 112, 112)


def _build_facial_area(bbox, kps: np.ndarray) -> dict:
    """
    Build the facial_area dict that stream_worker.is_valid_facial_area()
    expects — same schema as the old DeepFace output.

    Args:
        bbox: [x1, y1, x2, y2] int from InsightFace det.bbox
        kps:  (5, 2) float32 from InsightFace det.kps
              order: left_eye, right_eye, nose, mouth_left, mouth_right
    """
    x1, y1, x2, y2 = bbox
    return {
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


# ─────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────

class GPUWorker(threading.Thread):
    """
    Single background thread that owns the ONNX session and processes
    all embedding requests via GPU_TASK_QUEUE / GPU_RESULT_MAP.

    Payload types
    ─────────────
    "frame" – BGR face-crop from YOLO (≥ 256×256 recommended).
              Runs InsightFace SCRFD for 5-pt landmarks, aligns to
              112×112, then embeds. Returns [] if no confident face.

    "face"  – Pre-aligned 112×112 RGB crop from storage pipeline.
              Skips re-detection; converts BGR and embeds directly.
    """

    def __init__(self, model_path: str = "app/models/adaface_ir101.onnx"):
        super().__init__(daemon=True)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Warm up detector on init so first real request isn't slow
        _ = get_insight_app()

        logger.info(
            f"[GPUWorker] AdaFace IR-101 ONNX loaded · "
            f"model='{model_path}' · input='{self.input_name}'"
        )

    # ── internal helpers ─────────────────────────────────────

    def _embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """ONNX inference → L2-normalised 512-d float32 vector."""
        tensor = preprocess_adaface(aligned_bgr)
        raw = self.session.run(None, {self.input_name: tensor})[0].squeeze()
        norm = np.linalg.norm(raw)
        return (raw / (norm + 1e-6)).astype(np.float32)

    def _process_frame(self, frame_bgr: np.ndarray, yolo_area: dict = None) -> list:
        """
        Run InsightFace SCRFD on the FULL frame (not a crop).
        If yolo_area is provided, only keep detections that overlap
        with the YOLO bounding box — avoids false detections elsewhere.
        """
        faces = []
        app = get_insight_app()

        detections = app.get(frame_bgr)
        if not detections:
            return faces

        for det in detections:
            score = float(det.det_score)
            if score < _MIN_CONF:
                continue

            x1, y1, x2, y2 = det.bbox.astype(int)

            # ── Filter by YOLO area if provided ──────────────────────────────
            if yolo_area:
                ax, ay = yolo_area["x"], yolo_area["y"]
                aw, ah = yolo_area["w"], yolo_area["h"]
                
                # Instead of a loose center check, ensure the face actually belongs to this YOLO box
                # Calculate Intersection
                ix = max(x1, ax)
                iy = max(y1, ay)
                iw = max(0, min(x2, ax + aw) - ix)
                ih = max(0, min(y2, ay + ah) - iy)
                inter_area = iw * ih
                
                if inter_area == 0:
                    continue
                    
                # Reject if the detected face is tiny compared to the YOLO tracking box
                # (This perfectly prevents background faces from hijacking foreground YOLO boxes!)
                face_area = (x2 - x1) * (y2 - y1)
                yolo_box_area = aw * ah
                
                if face_area / yolo_box_area < 0.15:
                    continue

            kps     = det.kps.astype(np.float32)
            aligned = align_face_112(frame_bgr, kps)
            emb     = self._embed(aligned)
            area    = _build_facial_area(det.bbox.astype(int), kps)

            faces.append({"embedding": emb, "facial_area": area})

        faces.sort(
            key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"],
            reverse=True,
        )
        return faces

    def _process_face(self, face_rgb: np.ndarray) -> list:
        """
        Embed a pre-aligned RGB 112×112 variant (from store_embedding pipeline).
        No re-detection — crop was already landmark-aligned by embeddings.py.
        """
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        if face_bgr.shape[:2] != (112, 112):
            face_bgr = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)

        emb = self._embed(face_bgr)
        return [{"embedding": emb, "facial_area": {"x": 0, "y": 0, "w": 112, "h": 112}}]

    # ── main loop ────────────────────────────────────────────

    def run(self):
        logger.info("[GPUWorker] Running — waiting for tasks.")
        while True:
            task_id, payload = GPU_TASK_QUEUE.get()
            try:
                ptype     = payload["type"]
                data      = payload["data"]
                yolo_area = payload.get("yolo_area")   # ← add this

                if ptype == "frame":
                    faces = self._process_frame(data, yolo_area)  # ← pass it
                else:
                    faces = self._process_face(data)

                GPU_RESULT_MAP[task_id] = faces

            except Exception as e:
                logger.error(f"[GPUWorker] Task '{task_id}' failed: {e}", exc_info=True)
                GPU_RESULT_MAP[task_id] = []
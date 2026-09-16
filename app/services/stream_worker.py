# app/services/stream_worker.py
from collections import deque
import logging
from queue import Empty, Queue
import threading
import traceback
import cv2
import time
import os
from datetime import datetime
import numpy as np

import pytz
from app.db.qdrant_service import QdrantService
from app.scripts.qdrant_search import QdrantSearchEngine
from app.services.embedding_utils import extract_face_embedding
from app.services.video_recorder import VideoRecorder
from app.scripts.publisher_singleton import publisher as rabbitmq_publisher
from app.core.config import settings
from app.models.load_yolo_model import load_yolo_model
from logging.handlers import RotatingFileHandler
from app.core.logging_config import setup_missed_proof_logger

logger = logging.getLogger(__name__)
tz = pytz.timezone(settings.TIMEZONE)

qdrant_service = QdrantService()

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds from config (tune in .env without rebuilding)
# ─────────────────────────────────────────────────────────────────────────────
SINGLE_FRAME_THRESHOLD  = settings.FACE_INITIAL_THRESHOLD   # default 0.38
CONFIRMED_AVG_THRESHOLD = settings.FACE_CONFIRM_THRESHOLD   # default 0.40
MIN_CROP_SIZE = 256

# Size-Gated Inference Configuration
MIN_FACE_AREA_FOR_INFERENCE = 2500  # 50x50 pixels
MEDIUM_FACE_AREA = 6400             # 80x80 pixels

# Faces in this score range saved to marginal_reviews/ for threshold tuning
MARGINAL_LOW  = 0.30
MARGINAL_HIGH = SINGLE_FRAME_THRESHOLD

DEBUG = False


class StreamWorker:
    def __init__(self, camera_id, stream_url, monitoring_area):
        self.camera_id       = camera_id
        self.stream_url      = stream_url
        self.monitoring_area = monitoring_area

        self.yolo_model      = load_yolo_model()
        self.good_face_queue = Queue(maxsize=6)
        self.face_buffer     = deque(maxlen=6)
        self.frame_queue     = Queue(maxsize=5)
        self.stop_event      = threading.Event()

        self.search_engine      = QdrantSearchEngine()
        self.rabbitmq_publisher = rabbitmq_publisher

        # ── ByteTrack state tracking ──────────────────────────────
        self.track_states = {}       # Inference-side track states
        self.yolo_track_states = {}  # YOLO-side track states for buffering
        
        # Cooldown — prevents duplicate firings for the same person
        # tune: 30s = walk-past gate, 300s = room entry, 3600s = daily once
        self.global_cooldown: dict = {}
        self.COOLDOWN_SECONDS = 300  # Increased to avoid re-firing if track breaks and reconnects

        self.logger = logging.getLogger(f"camera.{camera_id}.worker")
        self.missed_proof_logger = setup_missed_proof_logger(camera_id)
        self.BASE_OUTPUT_DIR = "outputs"

        # ── Output directories ────────────────────────────────────────────
        self.recognized_dir  = os.path.join(self.BASE_OUTPUT_DIR, "recognized")
        self.marginal_dir    = os.path.join(self.BASE_OUTPUT_DIR, "marginal_reviews")
        self.infer_dir       = os.path.join(self.BASE_OUTPUT_DIR, "infer_reviews")

        for d in [
            self.recognized_dir,
            self.marginal_dir,
            self.infer_dir,
            "test_output/sample_detections",
            "test_output/sample_recognitions",
        ]:
            os.makedirs(d, exist_ok=True)

        self.logger.info(
            f"[CONFIG] single_frame={SINGLE_FRAME_THRESHOLD} "
            f"confirmed_avg={CONFIRMED_AVG_THRESHOLD} "
            f"marginal_range={MARGINAL_LOW}–{MARGINAL_HIGH} "
            f"cooldown={self.COOLDOWN_SECONDS}s  debug={DEBUG}"
        )

        self.recorder = VideoRecorder(camera_id=self.camera_id, logger=self.logger)
        self.recorder.start()

    # ─────────────────────────────────────────
    # RTSP Frame Grabber
    # ─────────────────────────────────────────
    def _capture_loop(self):
        reconnect_delay     = 1
        max_reconnect_delay = 10

        while not self.stop_event.is_set():
            cap = None
            try:
                self.logger.info(f"[{self.camera_id}] Opening RTSP...")
                cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    raise RuntimeError("RTSP open failed")

                reconnect_delay = 1
                self.logger.info(f"[{self.camera_id}] RTSP connected")
                self.recorder.on_camera_connected()

                FRAME_SKIP  = 1
                frame_count = 0
                consecutive_read_failures = 0
                while not self.stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        consecutive_read_failures += 1
                        if consecutive_read_failures == 10:
                            self.recorder.on_camera_disconnected()
                        if consecutive_read_failures >= 30:
                            self.logger.warning(
                                f"[{self.camera_id}] 30 consecutive RTSP read failures. Breaking loop to reconnect..."
                            )
                            break
                        time.sleep(0.05)
                        continue

                    if consecutive_read_failures > 0:
                        self.recorder.on_camera_connected()
                        consecutive_read_failures = 0

                    # Send frame to VideoRecorder (non-blocking)
                    self.recorder.write_frame(frame)

                    frame_count += 1
                    if frame_count % FRAME_SKIP != 0:
                        continue

                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    self.frame_queue.put(frame)

            except Exception as e:
                self.logger.error(f"[{self.camera_id}] Capture error: {e}")
                self.recorder.on_camera_disconnected()
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
            finally:
                if cap:
                    cap.release()

    # ─────────────────────────────────────────
    # Quality helpers
    # ─────────────────────────────────────────
    def preprocess_frame(self, frame):
        h, w, _ = frame.shape
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        return frame

    def is_good_face(self, frame, area):
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            return False

        ratio = w / float(h)
        if ratio < 0.3 or ratio > 3.0:
            if DEBUG:
                self.logger.info(
                    f"[DEBUG is_good_face] FAIL ratio={ratio:.2f} w={w} h={h}"
                )
            return False, 0.0

        gray   = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blur   = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Extremely relaxed for fast-moving CCTV tracking
        # Let AdaFace handle the rejection via low score rather than filtering aggressively here
        passed = blur >= 15 or (blur >= 10 and w >= 60 and h >= 60)

        if DEBUG:
            self.logger.info(
                f"[DEBUG is_good_face] blur={blur:.1f} w={w} h={h} "
                f"→ {'PASS' if passed else 'FAIL'}"
            )
        return passed, blur

    def is_valid_facial_area(self, area: dict) -> bool:
        if not area:
            return False

        x = area.get("x", 0)
        y = area.get("y", 0)
        w = area.get("w", 0)
        h = area.get("h", 0)

        if x <= 2 or y <= 2:
            return False
        if w < 40 or h < 40:
            return False

        ratio = w / float(h)
        # Side profiles are much narrower than front profiles, can drop down to ~0.45
        if ratio < 0.45 or ratio > 1.4:
            return False

        # Relaxed for side-profiles: require at least 3 landmarks out of 5 instead of all 5
        found_landmarks = 0
        for k in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right"):
            if k in area and area[k] is not None:
                found_landmarks += 1
                
        if found_landmarks < 3:
            return False

        return True

    def _iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1    = max(ax, bx);  y1 = max(ay, by)
        x2    = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
        inter = max(0, x2-x1) * max(0, y2-y1)
        union = aw*ah + bw*bh - inter
        return inter / union if union > 0 else 0

    def face_quality(self, face_crop):
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(blur / 100.0, 1.0)

    def _save_face_crop(self, frame: np.ndarray, area: dict, save_path: str) -> bool:
        """Crop face from frame using area dict and save to save_path."""
        try:
            x, y, w, h = area["x"], area["y"], area["w"], area["h"]
            H, W = frame.shape[:2]
            
            # Add padding to show more of the head/shoulders (e.g. 40%)
            pad_w = int(w * 0.4)
            pad_h = int(h * 0.4)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(W, x + w + pad_w)
            y2 = min(H, y + h + pad_h)
            
            if x1 >= x2 or y1 >= y2:
                return False
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            return cv2.imwrite(save_path, frame[y1:y2, x1:x2])
        except Exception as e:
            self.logger.error(f"[_save_face_crop] {e}")
            return False

    # ─────────────────────────────────────────
    # YOLO tracking loop
    # ─────────────────────────────────────────
    def _detect_face_yolo(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except Empty:
                continue

            try:
                # Use track instead of inference
                results = self.yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.4, iou=0.5, verbose=False)

                detected_faces = []
                current_time = time.time()

                for r in results:
                    # Skip if no tracking IDs are returned
                    if r.boxes.id is None:
                        continue
                        
                    for box, track_id in zip(r.boxes, r.boxes.id):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        tid = int(track_id)
                        
                        area = {"x": x1, "y": y1, "w": x2-x1, "h": y2-y1}

                        if area["w"] < 30 or area["h"] < 30:
                            continue

                        passed, blur = self.is_good_face(frame, area)
                        
                        face_crop = frame[area["y"]:area["y"]+area["h"],
                                         area["x"]:area["x"]+area["w"]]
                        if face_crop.size == 0:
                            continue

                        if not passed:
                            if conf > 0.6 and area["w"] >= 60:
                                detected_faces.append({
                                    "frame":    frame,
                                    "crop":     face_crop,
                                    "area":     area,
                                    "score":    conf,
                                    "track_id": tid,
                                    "fallback": False,
                                    "failure_reason": "blurry",
                                    "blur_score": blur
                                })
                            continue

                        h_c, w_c = face_crop.shape[:2]
                        if h_c < MIN_CROP_SIZE or w_c < MIN_CROP_SIZE:
                            scale     = max(MIN_CROP_SIZE/h_c, MIN_CROP_SIZE/w_c)
                            face_crop = cv2.resize(
                                face_crop, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_LINEAR,
                            )

                        face_item = {
                            "frame":    frame,
                            "crop":     face_crop,
                            "area":     area,
                            "score":    conf,
                            "track_id": tid,
                            "fallback": False,
                        }
                        
                        pixel_area = area["w"] * area["h"]
                        
                        # Initialize YOLO track state
                        if tid not in self.yolo_track_states:
                            self.yolo_track_states[tid] = {
                                "start_time": current_time,
                                "frames_processed": 0,
                                "small_face_buffer": [],
                                "last_seen": current_time,
                                "flushed": False
                            }
                            
                        y_state = self.yolo_track_states[tid]
                        y_state["last_seen"] = current_time
                        y_state["frames_processed"] += 1

                        if "start_time" not in y_state:
                            y_state["start_time"] = current_time

                        if settings.MAX_TRACKING_DURATION_SECONDS > 0 and current_time - y_state["start_time"] > settings.MAX_TRACKING_DURATION_SECONDS:
                            continue
                        
                        # Size-Gated Queuing & Buffering
                        if pixel_area < MIN_FACE_AREA_FOR_INFERENCE:
                            y_state["small_face_buffer"].append(face_item)
                            y_state["small_face_buffer"] = sorted(y_state["small_face_buffer"], key=lambda x: x["score"], reverse=True)[:3]
                            continue
                            
                        if pixel_area < MEDIUM_FACE_AREA:
                            # Throttle medium faces (every 3rd frame)
                            if y_state["frames_processed"] % 3 != 0:
                                continue
                                
                        detected_faces.append(face_item)

                # Process stale YOLO tracks for Last Resort Fallback
                stale_tids = [tid for tid, state in self.yolo_track_states.items() if current_time - state["last_seen"] > 5]
                for tid in stale_tids:
                    y_state = self.yolo_track_states[tid]
                    if not y_state["flushed"] and y_state["small_face_buffer"]:
                        for face_item in y_state["small_face_buffer"]:
                            face_item["fallback"] = True
                            if self.good_face_queue.full():
                                try:
                                    self.good_face_queue.get_nowait()
                                except Empty:
                                    pass
                            self.good_face_queue.put(face_item)
                        # self.logger.info(f"[{self.camera_id}] Track {tid} went stale. Flushed {len(y_state['small_face_buffer'])} small faces as fallback.")
                        y_state["flushed"] = True
                    
                    if current_time - y_state["last_seen"] > 30:
                        del self.yolo_track_states[tid]

                if not detected_faces:
                    continue

                # Prioritize larger and higher confidence faces
                stable_faces = sorted(
                    detected_faces,
                    key=lambda f: (0.7 * f["score"])
                        + (0.3 * (f["area"]["w"] * f["area"]["h"])
                           / (frame.shape[0] * frame.shape[1])),
                    reverse=True,
                )[:3]

                for face in stable_faces:
                    if self.good_face_queue.full():
                        try:
                            self.good_face_queue.get_nowait()
                        except Empty:
                            pass
                    self.good_face_queue.put(face)

            except Exception:
                self.logger.error(traceback.format_exc())

    # ─────────────────────────────────────────
    # Inference Loop
    # ─────────────────────────────────────────
    def _inference_loop(self):
        while not self.stop_event.is_set():
            try:
                item = self.good_face_queue.get(timeout=1)
            except Empty:
                continue

            try:
                frame       = item["frame"]
                yolo_area   = item["area"]
                track_id    = item["track_id"]
                is_fallback = item.get("fallback", False)
                    
                # Clean up old tracks periodically (tracks not seen in > 30 seconds)
                current_time = time.time()
                stale_tracks = [tid for tid, state in self.track_states.items() if current_time - state["last_seen"] > 30]
                for tid in stale_tracks:
                    del self.track_states[tid]

                # Initialize state for this track if new
                if track_id not in self.track_states:
                    self.track_states[track_id] = {
                        "start_time": current_time,
                        "consecutive_doctor_id": None,
                        "consecutive_timestamps": [],
                        "attendance_fired": False,
                        "last_seen": current_time,
                        "blurry_saved": False,
                        "invalid_angle_saved": False,
                        "low_score_saved": False,
                        "saved_missed_proof_paths": []
                    }
                
                state = self.track_states[track_id]
                state["last_seen"] = current_time

                if "start_time" not in state:
                    state["start_time"] = current_time

                if settings.MAX_TRACKING_DURATION_SECONDS > 0 and current_time - state["start_time"] > settings.MAX_TRACKING_DURATION_SECONDS:
                    continue

                failure_reason = item.get("failure_reason")
                if failure_reason == "blurry":
                    if not state.get("blurry_saved"):
                        blur_score = item.get("blur_score", 0.0)
                        saved_path = self._save_missed_proof(frame, yolo_area, "blurry", f"{blur_score:.2f}")
                        if saved_path:
                            state["saved_missed_proof_paths"] = state.get("saved_missed_proof_paths", []) + [saved_path]
                        state["blurry_saved"] = True
                    continue

                # ── GPU OPTIMIZATION: Skip AdaFace if this track is already confirmed ──
                if state["attendance_fired"]:
                    continue

                start_time = time.time()

                # ── stage 4: InsightFace SCRFD + AdaFace embed ────────────
                faces = extract_face_embedding(frame, yolo_area=yolo_area, track_id=track_id)

                if not faces:
                    continue

                face      = faces[0]
                embedding = face["embedding"]
                area      = face["facial_area"]   # InsightFace-detected bbox

                if not area:
                    continue

                # ── stage 5: facial area validation ──────────────────────
                if not self.is_valid_facial_area(area):
                    if not state.get("invalid_angle_saved"):
                        saved_path = self._save_missed_proof(frame, yolo_area, "invalid_angle", "0.00")
                        if saved_path:
                            state["saved_missed_proof_paths"] = state.get("saved_missed_proof_paths", []) + [saved_path]
                        state["invalid_angle_saved"] = True
                    continue

                # ── stage 6: Qdrant vector search ─────────────────────────
                results = self.search_engine.search(embedding.tolist())

                if not results:
                    if "history" not in state:
                        state["history"] = []
                    state["history"].append(None)
                    frame_buffer_size = settings.FRAME_BUFFER_SIZE if hasattr(settings, "FRAME_BUFFER_SIZE") else 3
                    if len(state["history"]) > frame_buffer_size:
                        state["history"].pop(0)

                    # Log the miss and current buffer state
                    if settings.TELEMETRY_LOGGING_ENABLED:
                        history_str = [str(x) if x is not None else "Unknown" for x in state["history"]]
                        self.logger.info(
                            f"[MULTI-FRAME MISS] Track {track_id} failed verification on this frame. "
                            f"Recent buffer contents: {history_str}."
                        )

                    if not state.get("low_score_saved"):
                        cand_name, cand_score = self.search_engine.search_candidate(embedding)
                        saved_path = self._save_missed_proof(frame, yolo_area, "low_score", f"{cand_score:.3f}", cand_name)
                        if saved_path:
                            state["saved_missed_proof_paths"] = state.get("saved_missed_proof_paths", []) + [saved_path]
                        state["low_score_saved"] = True
                    continue

                top         = results[0]
                score       = top["score"] # This is the average similarity of the 5 embeddings
                payload     = top.get("payload", {})
                doctor_id   = payload.get("doctor_id")
                doctor_name = payload.get("doctor_name", doctor_id) or "Unknown"
                verification_threshold = settings.VERIFICATION_THRESHOLD if hasattr(settings, "VERIFICATION_THRESHOLD") else 0.36
                if score < verification_threshold or not doctor_id:
                    if "history" not in state:
                        state["history"] = []
                    state["history"].append(None)
                    frame_buffer_size = settings.FRAME_BUFFER_SIZE if hasattr(settings, "FRAME_BUFFER_SIZE") else 3
                    if len(state["history"]) > frame_buffer_size:
                        state["history"].pop(0)

                    # Log the miss and current buffer state
                    if settings.TELEMETRY_LOGGING_ENABLED:
                        history_str = [str(x) if x is not None else "Unknown" for x in state["history"]]
                        self.logger.info(
                            f"[MULTI-FRAME MISS] Track {track_id} had score < MARGINAL_LOW or no doctor_id on this frame. "
                            f"Recent buffer contents: {history_str}."
                        )

                    if not state.get("low_score_saved"):
                        name_to_save = doctor_name if (doctor_id and doctor_name) else "Unknown"
                        saved_path = self._save_missed_proof(frame, yolo_area, "low_score", f"{score:.3f}", name_to_save)
                        if saved_path:
                            state["saved_missed_proof_paths"] = state.get("saved_missed_proof_paths", []) + [saved_path]
                        state["low_score_saved"] = True
                    continue

                # ── stage 7: temporal consistency ─────────────────────────
                multi_frame_mode = settings.MULTI_FRAME_MODE if hasattr(settings, "MULTI_FRAME_MODE") else "majority"
                frame_buffer_size = settings.FRAME_BUFFER_SIZE if hasattr(settings, "FRAME_BUFFER_SIZE") else 3
                min_matches = settings.MIN_MATCHES if hasattr(settings, "MIN_MATCHES") else 2

                passed_attendance = False

                if "history" not in state:
                    state["history"] = []

                # Track identity continuity: If a different verified doctor_id appears on the same track,
                # reset history to start fresh for the new identity (prevents A, B, A mixed identity count).
                existing_valid_docs = [x for x in state["history"] if x is not None]
                if existing_valid_docs and any(x != doctor_id for x in existing_valid_docs):
                    state["history"] = []

                # Add to history buffer
                state["history"].append(doctor_id)
                if len(state["history"]) > frame_buffer_size:
                    state["history"].pop(0)

                # 2-of-3 Sliding Window Confirmation: Count verified occurrences of this doctor_id in the buffer
                match_count = sum(1 for x in state["history"] if x == doctor_id)
                if match_count >= min_matches:
                    passed_attendance = True

                # Log final decision on the track (only once when firing or when first saving/skipping missed proof)
                should_log_decision = False
                if passed_attendance:
                    should_log_decision = True
                elif not state.get("low_score_saved"):
                    should_log_decision = True

                if should_log_decision:
                    threshold = settings.VERIFICATION_THRESHOLD
                    temporal_status = "PASS" if passed_attendance else "FAIL"
                    final_recog = "ACCEPT" if passed_attendance else "REJECT"
                    
                    if passed_attendance:
                        proof_decision_str = "NO (Attendance Fired)"
                        reason = "Temporal and similarity checks passed"
                    elif score < threshold:
                        proof_decision_str = "YES (Saved to missed_proofs/low_score)"
                        reason = "Similarity score below threshold"
                    else:
                        proof_decision_str = "YES (Saved to missed_proofs/unconfirmed)"
                        reason = "Similarity passed, but temporal check failed"

                    self.logger.info(
                        f"[FACE_ATTENDANCE]\n"
                        f"Doctor: {doctor_name}\n"
                        f"Raw Score: {score:.4f}\n"
                        f"Acceptance Threshold: {threshold:.4f}\n"
                        f"Verification: PASS\n"
                        f"Temporal: {temporal_status}\n"
                        f"Attendance Decision: {final_recog}\n"
                        f"Missed Proof Decision: {proof_decision_str}\n"
                        f"Reason: {reason}"
                    )

                if not passed_attendance:
                    # Log the miss and current buffer state
                    if settings.TELEMETRY_LOGGING_ENABLED:
                        history_str = [str(x) if x is not None else "Unknown" for x in state["history"]]
                        self.logger.info(
                            f"[MULTI-FRAME MISS] Track {track_id} verification passed for {doctor_name}, but temporal check failed. "
                            f"Recent buffer contents: {history_str}. Mode: {multi_frame_mode}"
                        )
                    
                    if not state.get("low_score_saved"):
                        category = "unconfirmed" if score >= settings.VERIFICATION_THRESHOLD else "low_score"
                        saved_path = self._save_missed_proof(frame, yolo_area, category, f"{score:.3f}", doctor_name)
                        if saved_path:
                            state["saved_missed_proof_paths"] = state.get("saved_missed_proof_paths", []) + [saved_path]
                        state["low_score_saved"] = True
                    continue

                # Special strict rule for Nancy Amin
                if "nancy amin" in doctor_name.lower():
                    if score < 0.42:
                        state["consecutive_doctor_id"] = None
                        state["consecutive_timestamps"] = []
                        if "history" in state:
                            state["history"] = []
                        continue

                # ── stage 8: global cooldown check ────────────────────────
                now  = datetime.now(tz)
                last = self.global_cooldown.get(doctor_id)
                if last:
                    elapsed_since = (now - last).total_seconds()
                    if elapsed_since < self.COOLDOWN_SECONDS:
                        # Already fired recently. Just mark track as done to save GPU.
                        self.track_states[track_id]["attendance_fired"] = True
                        state["consecutive_doctor_id"] = None
                        state["consecutive_timestamps"] = []
                        self.logger.info(
                            f"[COOLDOWN] {doctor_name} fired {elapsed_since:.0f}s ago "
                            f"— skipping track {track_id} (cooldown={self.COOLDOWN_SECONDS}s)"
                        )
                        continue

                # ── stage 9: fire attendance ─────────────────────────────
                self.global_cooldown[doctor_id] = now
                self.track_states[track_id]["attendance_fired"] = True

                # Clean up any transitional missed proof files saved for this track
                for path in state.get("saved_missed_proof_paths", []):
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                            self.logger.info(f"[MISSED PROOF CLEANUP] Deleted transitional missed proof: {path}")
                    except Exception as e:
                        self.logger.error(f"[MISSED PROOF CLEANUP] Failed to delete {path}: {e}")
                
                elapsed = time.time() - start_time

                self.logger.info(
                    f"✅ [{self.camera_id}] ATTENDANCE FIRED — Track {track_id} "
                    f"doctor={doctor_name} avg_score={score:.4f} time={elapsed:.2f}s"
                )

                # Save recognised face to outputs/recognized/
                ts    = now.strftime("%Y%m%d_%H%M%S")
                fname = f"{self.camera_id}_{doctor_id}_{score:.3f}_{ts}.jpg"
                recog_path = os.path.join(self.recognized_dir, fname)
                if self._save_face_crop(frame, area, recog_path):
                    self.logger.info(f"[RECOGNIZED] saved → {fname}")

                self._handle_detection(
                    frame, yolo_area, doctor_id, doctor_name, score, now
                )

            except Exception:
                self.logger.error(traceback.format_exc())

    # ─────────────────────────────────────────
    # Detection Handler
    # ─────────────────────────────────────────
    def _handle_detection(self, frame, area, doctor_id, doctor_name, score, now):
        ts = now.strftime("%Y%m%d_%H%M%S")

        relative_face_path  = f"detected_face/{self.camera_id}_{doctor_id}_{ts}.jpg"
        relative_frame_path = f"frames/{self.camera_id}_{doctor_id}_{ts}.jpg"
        abs_face_path       = os.path.join(self.BASE_OUTPUT_DIR, relative_face_path)
        abs_frame_path      = os.path.join(self.BASE_OUTPUT_DIR, relative_frame_path)

        os.makedirs(os.path.dirname(abs_face_path),  exist_ok=True)
        os.makedirs(os.path.dirname(abs_frame_path), exist_ok=True)

        if area:
            x, y, w, h = area["x"], area["y"], area["w"], area["h"]
            H, W = frame.shape[:2]
            
            # Add padding to show more of the head/shoulders (e.g. 40%)
            pad_w = int(w * 0.4)
            pad_h = int(h * 0.4)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(W, x + w + pad_w)
            y2 = min(H, y + h + pad_h)
            
            if x1 < x2 and y1 < y2:
                ok = cv2.imwrite(abs_face_path, frame[y1:y2, x1:x2])
                if not ok:
                    self.logger.error(f"Failed to write face: {abs_face_path}")

        ok = cv2.imwrite(abs_frame_path, frame)
        if not ok:
            self.logger.error(f"Failed to write frame: {abs_frame_path}")

        event = {
            "camera_id":           self.camera_id,
            "monitoring_area":     self.monitoring_area,
            "doctor_id":           doctor_id,
            "doctor_name":         doctor_name,
            "original_frame_path": relative_frame_path,
            "detected_face_path":  relative_face_path,
            "timestamp":           now.isoformat(),
        }
        self.rabbitmq_publisher.publish(event)
        self.logger.info(f"[{self.camera_id}] Recognised {doctor_name} (score={score:.3f})")

    def _save_missed_proof(self, frame, area, category, value_str, name="Unknown"):
        try:
            now = datetime.now(tz)
            date_str = now.strftime("%Y%m%d")
            time_str = now.strftime("%H%M%S")
            
            target_dir = os.path.join(self.BASE_OUTPUT_DIR, "missed_proofs", date_str, category)
            os.makedirs(target_dir, exist_ok=True)
            
            safe_name = name.replace(" ", "_")
            if category in ["low_score", "unconfirmed"]:
                filename = f"{date_str}_{time_str}_SCORE-{value_str}_{safe_name}_cam{self.camera_id}.jpg"
            elif category == "blurry":
                filename = f"{date_str}_{time_str}_var-{value_str}_Unknown_cam{self.camera_id}.jpg"
            else:
                filename = f"{date_str}_{time_str}_angle-check_Unknown_cam{self.camera_id}.jpg"
                
            save_path = os.path.join(target_dir, filename)
            
            x, y, w, h = area["x"], area["y"], area["w"], area["h"]
            H, W = frame.shape[:2]
            
            pad_w = int(w * 0.4)
            pad_h = int(h * 0.4)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(W, x + w + pad_w)
            y2 = min(H, y + h + pad_h)
            
            if x1 < x2 and y1 < y2:
                cv2.imwrite(save_path, frame[y1:y2, x1:x2])
                self.missed_proof_logger.info(f"[MISSED PROOF] saved ({category}) → {filename}")
                return save_path
        except Exception as e:
            self.missed_proof_logger.error(f"[MISSED PROOF] failed to save: {e}")
        return None

    def _heartbeat(self):
        while not self.stop_event.is_set():
            time.sleep(30)

    def run(self):
        capture_thread   = threading.Thread(target=self._capture_loop,    daemon=True)
        yolo_thread      = threading.Thread(target=self._detect_face_yolo, daemon=True)
        inference_thread = threading.Thread(target=self._inference_loop,  daemon=False)
        heartbeat_thread = threading.Thread(target=self._heartbeat,       daemon=True)

        capture_thread.start()
        yolo_thread.start()
        inference_thread.start()
        heartbeat_thread.start()

        try:
            while inference_thread.is_alive():
                inference_thread.join(timeout=1)
        except KeyboardInterrupt:
            self.stop_event.set()
            inference_thread.join()
        finally:
            self.recorder.stop()
# app/services/video_recorder.py
import os
import sys
import time
import platform
import logging
import threading
from queue import Queue, Empty
from datetime import datetime
import pytz
import cv2
import numpy as np

from app.core.config import settings

tz = pytz.timezone(settings.TIMEZONE)

def log_startup_diagnostics(logger: logging.Logger):
    """Log environment and system diagnostic information on startup."""
    recordings_root = getattr(settings, "RECORDINGS_PATH", "./recordings")
    abs_recordings = os.path.abspath(recordings_root)
    
    logger.info("=== VideoRecorder Production Diagnostics ===")
    logger.info(f"Current Working Directory: {os.getcwd()}")
    logger.info(f"Absolute Recordings Directory: {abs_recordings}")
    logger.info(f"Operating System: {platform.system()} {platform.release()}")
    logger.info(f"Python Version: {sys.version.split()[0]}")
    logger.info(f"OpenCV Version: {cv2.__version__}")
    logger.info("============================================")


class VideoRecorder:
    """
    Asynchronous, non-blocking video recorder per camera stream.
    Manages daily scheduled recording (e.g. 07:30 to 22:00) into date-based folders:
    recordings/YYYY-MM-DD/camera1.mp4
    """

    def __init__(self, camera_id: str, logger: logging.Logger = None):
        self.camera_id = str(camera_id)
        self.logger = logger or logging.getLogger(f"camera.{self.camera_id}.recorder")

        # Format camera filename (e.g. '1' -> 'camera1', 'camera1' -> 'camera1', 'entry_cam_1' -> 'entry_cam_1')
        if self.camera_id.isdigit():
            self.cam_filename = f"camera{self.camera_id}"
        elif self.camera_id.lower().startswith("camera"):
            self.cam_filename = self.camera_id.lower()
        else:
            self.cam_filename = self.camera_id

        # Name used in standard logs (e.g. 'Camera1' or 'Entry_cam_1')
        self.cam_display_name = self.cam_filename.capitalize() if self.cam_filename.startswith("camera") else self.cam_filename

        # Queue for incoming frames (maxsize=100 provides non-blocking buffer)
        self.frame_queue = Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.record_thread = None

        self.writer = None
        self.current_recording_date = None
        self.current_filepath = None
        self.is_recording_active = False

        self.current_width = None
        self.current_height = None
        self.frames_written = 0

        self.is_camera_connected = True

    def start(self):
        """Start the background recording thread."""
        log_startup_diagnostics(self.logger)

        if not getattr(settings, "RECORDING_ENABLED", True):
            self.logger.info(f"[{self.camera_id}] Video recording is DISABLED in config.")
            return

        self.stop_event.clear()
        self.record_thread = threading.Thread(
            target=self._record_loop,
            name=f"Recorder-{self.camera_id}",
            daemon=True
        )
        self.record_thread.start()
        self.logger.info(f"[{self.camera_id}] VideoRecorder initialized for camera '{self.cam_filename}'.")

    def stop(self):
        """Stop the background recording thread and release writer."""
        self.stop_event.set()
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=2)
        self._stop_writer()

    def write_frame(self, frame: np.ndarray):
        """
        Non-blocking call to enqueue a frame for recording.
        If queue is full, drops the oldest frame to avoid blocking the RTSP capture loop.
        """
        if not getattr(settings, "RECORDING_ENABLED", True) or self.stop_event.is_set():
            return

        if frame is None or frame.size == 0:
            return

        try:
            self.frame_queue.put_nowait(frame)
        except Exception:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except Exception:
                pass

    def on_camera_disconnected(self):
        """Notify recorder that camera stream was disconnected."""
        if self.is_camera_connected:
            self.is_camera_connected = False
            self.logger.warning(f"{self.cam_display_name} disconnected")

    def on_camera_connected(self):
        """Notify recorder that camera stream was reconnected."""
        if not self.is_camera_connected:
            self.is_camera_connected = True
            self.logger.info(f"{self.cam_display_name} reconnected")

    def _is_within_schedule(self, now: datetime) -> bool:
        """Check if current time falls within RECORD_START_TIME and RECORD_STOP_TIME."""
        try:
            start_str = getattr(settings, "RECORD_START_TIME", "07:00")
            stop_str = getattr(settings, "RECORD_STOP_TIME", "21:00")
            start_hour, start_min = map(int, start_str.split(":"))
            stop_hour, stop_min = map(int, stop_str.split(":"))
        except Exception as e:
            self.logger.error(f"[{self.camera_id}] Invalid RECORD_START_TIME/RECORD_STOP_TIME format: {e}")
            start_hour, start_min = 7, 0
            stop_hour, stop_min = 21, 0

        current_time = now.time()
        start_time = datetime.strptime(f"{start_hour:02d}:{start_min:02d}", "%H:%M").time()
        stop_time = datetime.strptime(f"{stop_hour:02d}:{stop_min:02d}", "%H:%M").time()

        if start_time <= stop_time:
            return start_time <= current_time < stop_time
        else:
            # Overnight schedule (e.g. 22:00 to 07:30)
            return current_time >= start_time or current_time < stop_time

    def _create_video_writer(self, filepath: str, fps: float, width: int, height: int):
        """
        Test codecs in order: mp4v, avc1, H264, MJPG, XVID.
        Choose the first codec that successfully opens VideoWriter.
        """
        codecs_to_try = ['mp4v', 'avc1', 'H264', 'MJPG', 'XVID']
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
                if writer.isOpened():
                    self.logger.info(f"Selected codec: {codec}")
                    return writer
            except Exception:
                pass

        self.logger.error(f"[{self.camera_id}] FAILED to open VideoWriter with any tested codec!")
        return None

    def _start_writer(self, now: datetime, frame_sample: np.ndarray):
        date_str = now.strftime("%Y-%m-%d")
        recordings_root = getattr(settings, "RECORDINGS_PATH", "./recordings")
        folder_path = os.path.abspath(os.path.join(recordings_root, date_str))

        folder_created = False
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
                folder_created = True
        except Exception as e:
            self.logger.error(f"[{self.camera_id}] Failed to create directory '{folder_path}': {e}", exc_info=True)
            return

        if not os.access(folder_path, os.W_OK):
            self.logger.error(f"[{self.camera_id}] Directory '{folder_path}' is NOT writable!")
            return

        filename = f"{self.cam_filename}.mp4"
        filepath = os.path.abspath(os.path.join(folder_path, filename))

        # Handle existing files to avoid overwriting previous recordings on restart
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            ts = now.strftime("%H%M%S")
            filename = f"{self.cam_filename}_{ts}.mp4"
            filepath = os.path.abspath(os.path.join(folder_path, filename))

        height, width = frame_sample.shape[:2]
        fps = 25.0  # Default RTSP target FPS

        self.writer = self._create_video_writer(filepath, fps, width, height)
        if not self.writer or not self.writer.isOpened():
            self.logger.error(f"[{self.camera_id}] VideoWriter.isOpened() returned False for path: {filepath}")
            self.writer = None
            return

        self.current_recording_date = date_str
        self.current_filepath = filepath
        self.current_width = width
        self.current_height = height
        self.frames_written = 0
        self.is_recording_active = True

        if folder_created:
            self.logger.info(f"Created folder recordings/{date_str}")
        self.logger.info(f"Recording started for {self.cam_display_name}")

    def _stop_writer(self):
        if self.writer:
            try:
                self.writer.release()
                self.logger.info("Recording finalized successfully.")
            except Exception as e:
                self.logger.error(f"[{self.camera_id}] Error releasing VideoWriter: {e}")
            self.writer = None

        if self.is_recording_active:
            self.logger.info("Recording stopped")
            self.is_recording_active = False

        self.current_recording_date = None
        self.current_filepath = None
        self.current_width = None
        self.current_height = None
        self.frames_written = 0

    def _record_loop(self):
        while not self.stop_event.is_set():
            now = datetime.now(tz)
            should_record = self._is_within_schedule(now)

            if not should_record:
                if self.is_recording_active:
                    self._stop_writer()
                # Drain queue while recording is inactive
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        break
                time.sleep(1)
                continue

            # Within recording schedule
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except Empty:
                continue

            if frame is None or frame.size == 0:
                continue

            today_str = now.strftime("%Y-%m-%d")
            # Handle midnight date change
            if self.is_recording_active and self.current_recording_date != today_str:
                self.logger.info(f"[{self.camera_id}] Date changed ({self.current_recording_date} -> {today_str}). Rotating recording file.")
                self._stop_writer()

            # Handle dynamic frame resolution changes
            h, w = frame.shape[:2]
            if self.is_recording_active and (w != self.current_width or h != self.current_height):
                self.logger.info(f"[{self.camera_id}] Frame size changed ({self.current_width}x{self.current_height} -> {w}x{h}). Re-initializing VideoWriter.")
                self._stop_writer()

            if not self.writer or not self.is_recording_active:
                self._start_writer(now, frame)

            if self.writer and self.writer.isOpened():
                try:
                    self.writer.write(frame)
                    self.frames_written += 1
                except Exception as e:
                    self.logger.error(f"[{self.camera_id}] Error writing video frame: {e}")

        self._stop_writer()

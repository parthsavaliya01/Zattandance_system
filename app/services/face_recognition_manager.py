# app/services/face_recognition_manager.py
import logging
from threading import Event, Thread
import time
from typing import Dict

from app.core.logging_config import setup_camera_logger, setup_logging
from app.db.camera_db import fetch_active_cameras
from app.services.stream_worker import StreamWorker

logger = logging.getLogger(__name__)

# Global registry (per FastAPI instance)
CAMERA_THREADS: Dict[str, Thread] = {}
CAMERA_STOP_EVENTS: Dict[str, Event] = {}
CAMERA_WORKERS: Dict[str, StreamWorker] = {}
CHECK_INTERVAL = 10


def redact_rtsp_url(url: str) -> str:
    if not url:
        return url

    prefix, separator, remainder = url.partition("://")
    if not separator:
        return url

    credential_end = remainder.rfind("@")
    if credential_end == -1:
        return url

    credential = remainder[:credential_end]
    if ":" not in credential:
        return url

    username = credential.split(":", 1)[0]
    return f"{prefix}://{username}:***@{remainder[credential_end + 1:]}"


class FaceRecognition:
    def __init__(self):
        pass

    def _fetch_active_cameras(self):
        return fetch_active_cameras()

    def camera_worker(self, camera_id: str, cam: dict, stop_event: Event):
        """Target function for each camera thread"""
        cam_logger = setup_camera_logger(camera_id)
        cam_logger.info(f"[BOOT] Starting StreamWorker for {camera_id}")
        worker = StreamWorker(
            camera_id=camera_id,
            stream_url=cam["url"],
            monitoring_area=cam["area"]
        )
        CAMERA_WORKERS[camera_id] = worker
        if stop_event.is_set():
            worker.stop_event.set()
        try:
            worker.run()
        finally:
            if CAMERA_WORKERS.get(camera_id) is worker:
                CAMERA_WORKERS.pop(camera_id, None)

    def _start_camera_worker(self, cam: dict, daemon: bool = True):
        cam_id = cam["camera_id"]
        stop_event = Event()
        thread = Thread(
            target=self.camera_worker,
            args=(cam_id, cam, stop_event),
            daemon=daemon,
        )
        CAMERA_STOP_EVENTS[cam_id] = stop_event
        CAMERA_THREADS[cam_id] = thread
        thread.start()
        logger.info(
            "[FACE-RECOGNITION] Camera worker started %s (%s) -> %s",
            cam_id,
            cam["area"],
            redact_rtsp_url(cam["url"]),
        )


    def start_camera(self):
        """
        Start face recognition for all cameras defined in settings.CAMERAS
        """
        started = []
        skipped = []
        cameras = self._fetch_active_cameras()

        for cam in cameras:
            cam_id = cam["camera_id"]
            if cam_id in CAMERA_THREADS and CAMERA_THREADS[cam_id].is_alive():
                skipped.append(cam_id)
                continue

            self._start_camera_worker(cam)
            started.append(cam_id)

        return {
            "started": started,
            "already_running": skipped,
            "total_running": len(CAMERA_THREADS),
        }


    def get_camera_status(self):
        status = {}
        for cam_id, thread in CAMERA_THREADS.items():
            status[cam_id] = {
                # "pid": proc.pid,
                "alive": thread.is_alive(),
            }
        return status
    
    def start_single_camera_by_id(self, camera_id: str):
        cameras = self._fetch_active_cameras()
        cam = next((c for c in cameras if c["camera_id"] == camera_id), None)
        if not cam:
            return False

        if camera_id in CAMERA_THREADS and CAMERA_THREADS[camera_id].is_alive():
            return False

        self._start_camera_worker(cam, daemon=False)
        return True


    def stop_camera(self,camera_id: str):
        thread = CAMERA_THREADS.get(camera_id)
        if thread and thread.is_alive():
            stop_event = CAMERA_STOP_EVENTS.get(camera_id)
            if stop_event:
                stop_event.set()
            CAMERA_THREADS.pop(camera_id,None)
            CAMERA_STOP_EVENTS.pop(camera_id, None)
            CAMERA_WORKERS.pop(camera_id, None)
            logger.info(f"[FACE-RECOGNITION] Camera stopped: {camera_id}")
            return True
        return False
    
    def sync_cameras(self):
        """
        Sync DB cameras with running processes
        """
        cameras = self._fetch_active_cameras()
        active_ids = {c["camera_id"] for c in cameras}

        # Start new cameras and restart workers whose configuration changed.
        for cam in cameras:
            cam_id = cam["camera_id"]
            thread = CAMERA_THREADS.get(cam_id)
            worker = CAMERA_WORKERS.get(cam_id)
            config_changed = worker is not None and (
                worker.stream_url != cam["url"]
                or worker.monitoring_area != cam["area"]
            )

            if thread and thread.is_alive() and not config_changed:
                continue

            if thread:
                stop_event = CAMERA_STOP_EVENTS.pop(cam_id, None)
                if stop_event:
                    stop_event.set()
                CAMERA_THREADS.pop(cam_id, None)
                CAMERA_WORKERS.pop(cam_id, None)

            self._start_camera_worker(cam)

        # Stop cameras removed, disabled, or assigned to an unsupported area.
        for cam_id in list(CAMERA_THREADS.keys()):
            if cam_id not in active_ids:
                self.stop_camera(cam_id)

    def monitor(self):
        """
        Restart crashed workers
        """
        for cam_id, thread in list(CAMERA_THREADS.items()):
            if not thread.is_alive():
                logger.warning(f"[MONITOR] Camera crashed: {cam_id}")
                CAMERA_THREADS.pop(cam_id, None)
                CAMERA_STOP_EVENTS.pop(cam_id, None)
                CAMERA_WORKERS.pop(cam_id, None)

    def run(self):
        logger.info("[MANAGER] Camera Manager started")
        while True:
            try:
                self.sync_cameras()
                self.monitor()
                time.sleep(CHECK_INTERVAL)
            except Exception:
                logger.exception("[MANAGER] Fatal error")
                time.sleep(5)
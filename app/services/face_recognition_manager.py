# app/services/face_recognition_manager.py
import logging
from multiprocessing import Process
from threading import Thread
import time
from typing import Dict

from app.core.logging_config import setup_camera_logger, setup_logging
from app.db.camera_db import fetch_active_cameras
from app.services.stream_worker import StreamWorker

logger = logging.getLogger(__name__)

# Global registry (per FastAPI instance)
CAMERA_THREADS: Dict[str, Thread] = {}
CHECK_INTERVAL = 10


class FaceRecognition:
    def __init__(self):
        pass

    def _fetch_active_cameras(self):
        return fetch_active_cameras()

    def camera_worker(self, camera_id: str, cam: dict):
        """Target function for each camera thread"""
        cam_logger = setup_camera_logger(camera_id)
        cam_logger.info(f"[BOOT] Starting StreamWorker for {camera_id}")
        worker = StreamWorker(
            camera_id=camera_id,
            stream_url=cam["url"],
            monitoring_area=cam["area"]
        )
        worker.run()


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

            # p = Process(
            #     target= self.camera_worker,
            #     args=(cam_id, cam),
            #     daemon=False,   # important for FastAPI lifecycle
            # )
            # p.start()
            # CAMERA_THREADS[cam_id] = p

            t = Thread(target=self.camera_worker, args=(cam_id, cam), daemon=True)
            t.start()
            CAMERA_THREADS[cam_id] = t

            started.append(cam_id)

            logger.info(
                f"[FACE-RECOGNITION] Camera started {cam_id} "
                f"({cam['area']}) -> {cam['url']}"
            )

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

        # p = Process(
        #     target=self.camera_worker,
        #     args=(camera_id, cam),
        #     daemon=False
        # )
        # p.start()

        # CAMERA_THREADS[camera_id] = p

        t = Thread(target=self.camera_worker, args=(camera_id, cam), daemon=False)
        t.start()
        CAMERA_THREADS[camera_id] = t
        return True


    def stop_camera(self,camera_id: str):
        thread = CAMERA_THREADS.get(camera_id)
        if thread and thread.is_alive():
            CAMERA_THREADS.pop(camera_id,None)
            logger.info(f"[FACE-RECOGNITION] Camera stopped: {camera_id}")
            return True
        return False
    
    def sync_cameras(self):
        """
        Sync DB cameras with running processes
        """
        cameras = self._fetch_active_cameras()
        active_ids = {c["camera_id"] for c in cameras}

        # Start missing cameras
        for cam in cameras:
            if cam["camera_id"] not in CAMERA_THREADS:
                self.start_camera(cam)

        # Stop disabled cameras
        for cam_id in list(CAMERA_THREADS.keys()):
            if cam_id not in active_ids:
                self.stop_camera(cam_id)

    def monitor(self):
        """
        Restart crashed workers
        """
        for cam_id, thread in CAMERA_THREADS.items():
            if not thread.is_alive():
                logger.warning(f"[MONITOR] Camera crashed: {cam_id}")
                CAMERA_THREADS.pop(cam_id, None)

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
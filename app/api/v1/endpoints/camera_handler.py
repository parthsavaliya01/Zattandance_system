# app/api/v1/endpoints/camera_handler.py
from fastapi import APIRouter, HTTPException
from app.services.face_recognition_manager import FaceRecognition


router = APIRouter(tags=['Camera Handler'])
face_recognition = FaceRecognition()

@router.post("/start/{camera_id}")
def start_single_camera(camera_id: str):
    ok = face_recognition.start_single_camera_by_id(camera_id)
    if not ok:
        raise HTTPException(404, "Camera not found or already running")
    return {"status": "started", "camera_id": camera_id}


@router.post("/stop/{camera_id}")
def stop_single_camera(camera_id: str):
    ok = face_recognition.stop_camera(camera_id)
    if not ok:
        raise HTTPException(404, "Camera not running")
    return {"status": "stopped", "camera_id": camera_id}

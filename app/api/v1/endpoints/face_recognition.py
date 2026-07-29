# app/api/v1/endpoints/face_recognition.py
from fastapi import APIRouter, HTTPException
from app.services.face_recognition_manager import FaceRecognition

router = APIRouter(
    prefix="/face-recognition",
    tags=["Face Recognition"],
)
face_recognition = FaceRecognition()

@router.post("/start")
def start_face_recognition():
    """
    Start live face recognition on all RTSP cameras
    configured in settings.CAMERAS
    """
    result = face_recognition.start_camera()

    if not result["started"] and result["already_running"]:
        return {
            "status": "already_running",
            "details": result,
        }

    return {
        "status": "started",
        "details": result,
    }


@router.get("/status")
def face_recognition_status():
    """
    Check status of all running camera processes
    """
    return {
        "running_cameras": face_recognition.get_camera_status()
    }

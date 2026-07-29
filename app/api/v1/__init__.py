from fastapi import APIRouter
from app.api.v1.endpoints import camera_handler, health, upload_embedding, face_recognition, image_logs, missed_proofs

api_router = APIRouter()
api_router.include_router(upload_embedding.router)
api_router.include_router(face_recognition.router)
api_router.include_router(camera_handler.router)
api_router.include_router(health.router)
api_router.include_router(image_logs.router)
api_router.include_router(missed_proofs.router)

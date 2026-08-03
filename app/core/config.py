# app/core/config.py
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv(override=True)

class Settings(BaseSettings):
    CAMERA_IP_1: str
    CAMERA_IP_2: str
    CAMERA_USERNAME: str
    CAMERA_PASSWORD:str

    TIMEZONE: str = "Asia/Kolkata"

    MYSQL_HOST: str 
    MYSQL_PORT: int
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_DB: str

    QDRANT_HOST: str
    QDRANT_PORT: int

    RABBITMQ_HOST: str
    RABBITMQ_PORT: int
    RABBITMQ_USERNAME: str
    RABBITMQ_PASSWORD: str
    RABBITMQ_QUEUE: str = 'face_recognition_events'

    DARWINBOX_API_URL: str
    DARWINBOX_API_KEY: str
    DARWINBOX_USERNAME: str
    DARWINBOX_PASSWORD: str

    CORS_ORIGINS: str = "*"

    # DETECTOR_BACKEND: str = 'retinaface'
    # EMBEDDING_MODEL: str = 'Facenet512'  # kept for reference; AdaFace ONNX is used instead

    ADAFACE_MODEL_PATH: str = 'app/models/adaface_ir101.onnx'

    FACE_INITIAL_THRESHOLD: float = 0.38
    FACE_CONFIRM_THRESHOLD: float = 0.40

    RECORDING_ENABLED: bool = True
    RECORD_START_TIME: str = "07:00"
    RECORD_STOP_TIME: str = "21:00"
    RECORDINGS_PATH: str = "./recordings"

    QDRANT_COLLECTION_NAME: str = 'doctors_adaface'
    QDRANT_COLLECTION_CENTROIDS: str = 'doctor_centroids'
    QDRANT_COLLECTION_EMBEDDINGS: str = 'doctor_embeddings'
    CENTROID_THRESHOLD: float = 0.35
    VERIFICATION_THRESHOLD: float = 0.38
    CONSECUTIVE_FRAMES: int = 3
    CONSECUTIVE_TIME_WINDOW: float = 3.0
    TOP_K: int = 3
    ENROLL_FRONT_ONLY: bool = False
    ENROLL_FRONT_FILENAME_KEYWORDS: str = "front_face,front_profile,front"
    ENROLL_MIN_CONF: float = 0.40

    @property
    def CAMERAS(self):
        return {
            "entry_cam_1": {
                "url": f"rtsp://{self.CAMERA_USERNAME}:{self.CAMERA_PASSWORD}@{self.CAMERA_IP_1}:554/Streaming/Channels/101",
                "type": "entry",
            },
            "exit_cam_1": {
                "url": f"rtsp://{self.CAMERA_USERNAME}:{self.CAMERA_PASSWORD}@{self.CAMERA_IP_2}:554/Streaming/Channels/101",
                "type": "exit",
            }
        }
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

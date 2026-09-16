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

    RECORDING_ENABLED: bool = False
    RECORD_START_TIME: str = "07:00"
    RECORD_STOP_TIME: str = "21:00"
    RECORDINGS_PATH: str = "./recordings"

    QDRANT_COLLECTION_NAME: str = 'doctors_adaface'
    QDRANT_COLLECTION_CENTROIDS: str = 'doctor_centroids'
    QDRANT_COLLECTION_EMBEDDINGS: str = 'doctor_embeddings'
    CENTROID_THRESHOLD: float = 0.35
    VERIFICATION_THRESHOLD: float = 0.36
    CONSECUTIVE_FRAMES: int = 2
    CONSECUTIVE_TIME_WINDOW: float = 3.0
    TOP_K: int = 5
    ENROLL_FRONT_ONLY: bool = False
    ENROLL_FRONT_FILENAME_KEYWORDS: str = "front_face,front_profile,front"
    ENROLL_MIN_CONF: float = 0.40

    # Adaptive threshold settings
    ADAPTIVE_THRESHOLD_ENABLED: bool = False
    ADAPTIVE_THRESHOLD_ALPHA: float = 0.1
    MIN_ADAPTIVE_THRESHOLD: float = 0.32
    MAX_ADAPTIVE_THRESHOLD: float = 0.40
    INTRA_DOCTOR_SPREAD_WARNING_THRESHOLD: float = 0.50
    TELEMETRY_LOGGING_ENABLED: bool = False

    # Multi-frame confirmation settings
    MULTI_FRAME_MODE: str = "majority"  # "consecutive" or "majority"
    FRAME_BUFFER_SIZE: int = 3
    MIN_CONSECUTIVE_MATCHES: int = 2
    MIN_MATCHES: int = 2
    SCORE_MARGIN_THRESHOLD: float = 0.010

    # Face quality gate settings
    FACE_QUALITY_GATE_ENABLED: bool = False
    MIN_FACE_SIZE_PX: int = 50
    MAX_YAW_DEGREES: float = 30.0
    MIN_BLUR_SCORE: float = 15.0
    MAX_TRACKING_DURATION_SECONDS: float = 0.0

    MAX_LOG_FILE_SIZE_MB: int = 10


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

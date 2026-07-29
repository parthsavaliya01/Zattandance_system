# main.py
from contextlib import asynccontextmanager
import logging
import os
import multiprocessing as mp

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.logging_config import setup_logging
from app.api.v1 import api_router
from app.models.load_yolo_model import load_yolo_model
from app.services.gpu_worker import GPUWorker

from app.core.config import settings

mp.set_start_method("spawn", force=True)
setup_logging("logs/faceRecognition.log", with_console=True)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    import sys, platform, cv2
    rec_path = os.path.abspath(getattr(settings, "RECORDINGS_PATH", "./recordings"))
    logger.info("=== Application System Diagnostics ===")
    logger.info(f"Current Working Directory: {os.getcwd()}")
    logger.info(f"Absolute Recordings Directory: {rec_path}")
    logger.info(f"Operating System: {platform.system()} {platform.release()}")
    logger.info(f"Python Version: {sys.version.split()[0]}")
    logger.info(f"OpenCV Version: {cv2.__version__}")
    logger.info("======================================")

    logger.info("Starting GPUWorker (AdaFace ONNX + InsightFace SCRFD)...")
    gpu_worker = GPUWorker()
    gpu_worker.start()
    logger.info("GPUWorker started.")

    logger.info("Loading YOLO model...")
    load_yolo_model()
    logger.info("YOLO model ready.")

    logger.info("Starting camera workers...")
    from app.services.face_recognition_manager import FaceRecognition
    FaceRecognition().start_camera()

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down.")


app = FastAPI(lifespan=lifespan)

app.mount(
    "/outputs",
    StaticFiles(directory="/app/outputs"),
    name="media",
)

os.makedirs("/app/recordings", exist_ok=True)
app.mount(
    "/recordings",
    StaticFiles(directory="/app/recordings"),
    name="recordings",
)

# Parse allowed origins from settings
origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
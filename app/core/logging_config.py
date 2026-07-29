import logging
from logging.handlers import RotatingFileHandler
import os


def setup_app_logging(level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return  # ✅ already configured

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [PID:%(process)d] [%(name)s] %(message)s"
    )

    os.makedirs("logs", exist_ok=True)

    file_handler = RotatingFileHandler(
        "logs/faceRecognition.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # silence noise
    for noisy in ["httpx", "qdrant_client", "urllib3", "pika"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

def setup_camera_logger(camera_id: str):
    logger = logging.getLogger(f"camera.{camera_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 🔥 critical

    if logger.handlers:
        return logger  # already configured

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    os.makedirs("logs", exist_ok=True)

    fh = RotatingFileHandler(
        f"logs/camera_{camera_id}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger

def setup_logging(log_file: str,level=logging.INFO,with_console: bool=True):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 🔥 FULL RESET per process
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [PID:%(process)d] [%(name)s] %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=20 * 1024 * 1024, # 20 MB
        backupCount=5,
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if with_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    # Silence noisy libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pika").setLevel(logging.WARNING)
    logging.getLogger("pika.adapters").setLevel(logging.WARNING)

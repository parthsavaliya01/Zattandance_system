import logging
from logging.handlers import BaseRotatingHandler
import os
from app.core.config import settings

class DynamicRotatingFileHandler(BaseRotatingHandler):
    def __init__(self, filename, mode='a', maxBytes=0, encoding=None, delay=False):
        self.maxBytes = maxBytes
        self.base_filepath = os.path.abspath(filename)
        # We must call _determine_active_filename to get the correct active log file on startup.
        active_filename = self._determine_active_filename()
        super().__init__(active_filename, mode, encoding, delay)

    def _determine_active_filename(self) -> str:
        dir_name, base_name = os.path.split(self.base_filepath)
        name, ext = os.path.splitext(base_name)
        os.makedirs(dir_name, exist_ok=True)
        
        # Check base file first
        candidate = self.base_filepath
        if not os.path.exists(candidate):
            return candidate
        if os.path.getsize(candidate) < self.maxBytes:
            return candidate
            
        # Base file is full. Find the first indexed file that is not full or doesn't exist.
        idx = 1
        while True:
            candidate = os.path.join(dir_name, f"{name}_{idx}{ext}")
            if not os.path.exists(candidate):
                return candidate
            if os.path.getsize(candidate) < self.maxBytes:
                return candidate
            idx += 1

    def shouldRollover(self, record):
        if self.stream is None:
            self.stream = self._open()
        if self.maxBytes > 0:
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return True
        return False

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        self.baseFilename = self._determine_active_filename()
        if not self.delay:
            self.stream = self._open()


def setup_app_logging(level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return  # ✅ already configured

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [PID:%(process)d] [%(name)s] %(message)s"
    )

    os.makedirs("logs", exist_ok=True)

    max_bytes = getattr(settings, "MAX_LOG_FILE_SIZE_MB", 50) * 1024 * 1024
    file_handler = DynamicRotatingFileHandler(
        "logs/faceRecognition.log",
        maxBytes=max_bytes,
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

    max_bytes = getattr(settings, "MAX_LOG_FILE_SIZE_MB", 50) * 1024 * 1024
    fh = DynamicRotatingFileHandler(
        f"logs/camera_{camera_id}.log",
        maxBytes=max_bytes,
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

    max_bytes = getattr(settings, "MAX_LOG_FILE_SIZE_MB", 50) * 1024 * 1024
    file_handler = DynamicRotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
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


def setup_missed_proof_logger(camera_id: str):
    log_type = "entry" if camera_id == "111" else "exit"
    log_filename = f"logs/missed_proof_{log_type}.log"
    
    logger = logging.getLogger(f"missed_proof.{camera_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )
    
    os.makedirs("logs", exist_ok=True)
    max_bytes = getattr(settings, "MAX_LOG_FILE_SIZE_MB", 50) * 1024 * 1024
    fh = DynamicRotatingFileHandler(
        log_filename,
        maxBytes=max_bytes,
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# app/services/health_service.py
import logging
from app.db.qdrant_service import QdrantService
from app.scripts.rabbitmq_publisher import RabbitMQPublisher
from app.services.face_recognition_manager import CAMERA_THREADS

logger = logging.getLogger(__name__)

class HealthService:

    def system_health(self):
        return {
            "status": "ok",
            "cameras": self.camera_health(),
            "qdrant": self.qdrant_health(),
            "rabbitmq": self.rabbitmq_health(),
            "gpu": self.gpu_health(),
        }

    def gpu_health(self):
        # Added field only; "status" above is unchanged for existing callers.
        try:
            import torch
            from app.services.gpu_worker import GPU_STATUS

            yolo_on_gpu = torch.cuda.is_available()
            adaface = GPU_STATUS.get("adaface_providers", [])
            detector = GPU_STATUS.get("detector_providers", [])
            adaface_on_gpu = bool(adaface) and adaface[0] == "CUDAExecutionProvider"
            detector_on_gpu = bool(detector) and detector[0] == "CUDAExecutionProvider"
            return {
                "status": "ok" if (yolo_on_gpu and adaface_on_gpu and detector_on_gpu) else "cpu_fallback",
                "yolo_on_gpu": yolo_on_gpu,
                "adaface_providers": adaface,
                "detector_providers": detector,
            }
        except Exception as e:
            logger.error(f"GPU health failed: {e}")
            return {"status": "unknown"}

    def readiness(self):
        return {
            "qdrant": self.qdrant_health(),
            "rabbitmq": self.rabbitmq_health()
        }

    def camera_health(self):
        return {
            cam_id: {
                # "pid": proc.pid,
                "alive": thread.is_alive()
            }
            for cam_id, thread in CAMERA_THREADS.items()
        }

    def qdrant_health(self):
        try:
            QdrantService().connect().get_collections()
            return "ok"
        except Exception as e:
            logger.error(f"Qdrant health failed: {e}")
            return "down"

    def rabbitmq_health(self):
        try:
            pub = RabbitMQPublisher()
            pub.channel.queue_declare(queue="healthcheck", passive=True)
            return "ok"
        except Exception as e:
            logger.error(f"RabbitMQ health failed: {e}")
            return "down"

# app/api/v1/endpoints/darwinbox_client.py
import requests
import logging
from datetime import datetime
from app.core.config import settings

logger = logging.getLogger(__name__)

class DarwinboxClient:
    def __init__(self):
        self.url = settings.DARWINBOX_API_URL  # https://<subdomain>.darwinbox.in/AttendancePunchesApi
        self.api_key = settings.DARWINBOX_API_KEY
        self.auth = (
            settings.DARWINBOX_USERNAME,
            settings.DARWINBOX_PASSWORD,
        )

    def push_attendance(self, punches: dict):
        """
        punches format:
        {
          "EMP001": [
              {
                "id": "...",
                "timestamp": "...",
                "machineid": "...",
                "status": 1
              }
          ]
        }
        """
        payload = {
            "api_key": self.api_key,
            "attendance": punches
        }

        try:
            resp = requests.post(
                self.url,
                json=payload,
                auth=self.auth,
                timeout=10
            )

            if resp.status_code == 200:
                logger.info("[DARWINBOX] Attendance pushed successfully")
                return True, resp.json()

            logger.error(
                f"[DARWINBOX] Failed {resp.status_code}: {resp.text}"
            )
            return False, resp.text

        except Exception as e:
            logger.exception("[DARWINBOX] API call failed")
            return False, str(e)

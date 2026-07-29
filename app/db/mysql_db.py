import logging
import traceback
import mysql.connector
from app.core.config import settings

logger = logging.getLogger(__name__)


def db_connection():
    try:
        return mysql.connector.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DB
        )
    except Exception as e:
        logger.error(f"[MYSQL] Connection failed: {e}")
        logger.error(traceback.format_exc())
        return None
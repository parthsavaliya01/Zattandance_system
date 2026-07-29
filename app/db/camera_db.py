#
import logging
from typing import List, Dict
from app.db.mysql_db import db_connection  

logger = logging.getLogger(__name__)

# Fetch all cameras (Admin view)
def fetch_all_cameras() -> List[Dict]:
    conn = None
    try:
        conn = db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                camera_id,
                camera_stream_url,
                monitoring_area,
                is_active,
                created_at,
                updated_at
            FROM tbl_cameras
        """)

        return cursor.fetchall()

    except Exception as e:
        logger.exception("[CAMERA_DB] Failed to fetch cameras")
        return []

    finally:
        if conn:
            conn.close()

# Fetch only active cameras (Execution plane)
def fetch_active_cameras() -> List[Dict]:
    conn = None
    try:
        conn = db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT camera_id, camera_stream_url, monitoring_area 
            FROM tbl_cameras
            WHERE is_active = 1
            AND is_deleted = 0
            AND LOWER(monitoring_area) IN ('entry', 'exit')
        """)

        rows = cursor.fetchall()
        cameras = []

        for row in rows:
            cameras.append({
                "camera_id": row["camera_id"],
                "url": row["camera_stream_url"],
                "area": row["monitoring_area"].lower(),
            })

        logger.info(f"[CAMERA_DB] Loaded {len(cameras)} active entry/exit cameras")
        return cameras

    except Exception as e:
        logger.exception(f"[CAMERA_DB] Failed to fetch cameras: {e}")
        return []

    finally:
        if conn:
            conn.close()

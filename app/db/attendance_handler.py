# app/db/attendance_db.py
import logging
import traceback
import uuid
from app.core.config import settings
from datetime import datetime, date
import pytz
from app.db.mysql_db import db_connection

logger = logging.getLogger(__name__)
tz = pytz.timezone(settings.TIMEZONE)


class AttendanceService:
    def __init__(self):
        pass

    def _get_conn(self):
        return db_connection()
    
    def get_latest_session(self, doctor_id, attendance_date):
        conn = self._get_conn()
        if conn is None:
            return None
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT *
                FROM tbl_attendance
                WHERE doctor_id = %s AND attendance_date = %s
                ORDER BY COALESCE(in_time, out_time) DESC
                LIMIT 1
            """, (doctor_id, attendance_date))
            return cur.fetchone()
        finally:
            cur.close()
            conn.close()

    def get_latest_open_session(self, doctor_id, attendance_date):
        conn = self._get_conn()
        if conn is None:
            return None
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT *
                FROM tbl_attendance
                WHERE doctor_id = %s
                AND attendance_date = %s
                AND in_time IS NOT NULL
                AND out_time IS NULL
                ORDER BY in_time DESC
                LIMIT 1
            """, (doctor_id, attendance_date))
            return cur.fetchone()
        finally:
            cur.close()
            conn.close()

    def create_exit_only_attendance(
        self, doctor_id, doctor_name, attendance_date,
        out_ts, exit_camera_id, out_image_path=None
    ):
        conn = self._get_conn()
        if conn is None:
            return None
        try:
            cur = conn.cursor()
            now = datetime.now(tz)
            cur.execute("""
                INSERT INTO tbl_attendance
                (doctor_id, attendance_date, out_time, exit_camera_id, out_image_path, created_at, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                doctor_id,
                attendance_date,
                out_ts,
                exit_camera_id,
                out_image_path,
                now,
                now
            ))
            conn.commit()
            return cur.lastrowid
        finally:
            cur.close()
            conn.close()


    def find_open_attendance_for_doctor(self,doctor_id, attendance_date):
        """
        Find attendance record for doctor_id on attendance_date where in_time is not null and out_time is null.
        returns row dict or None
        """
        conn = self._get_conn()
        if conn is None:
            return None
        
        try:
            cur = conn.cursor(dictionary=True)
            query = """
                SELECT * FROM tbl_attendance
                WHERE doctor_id = %s AND attendance_date = %s AND in_time IS NOT NULL AND out_time IS NULL
                LIMIT 1
            """
            cur.execute(query, (doctor_id, attendance_date))
            return cur.fetchone()
        except Exception as e:
            logger.error(f"[MYSQL] Error fetching open attendance for {doctor_id}: {e}")
            logger.error(traceback.format_exc())
        finally:
            cur.close()
            conn.close()

    def create_attendance(self,doctor_id, doctor_name, attendance_date, in_ts, entry_camera_id,in_image_path=None):
        """
        Insert new attendance record. out_time/exit_camera/total_hours left NULL.
        Returns inserted attendance_id.
        """
        conn = self._get_conn()
        if conn is None:
            return None
        
        try:
            cur = conn.cursor()

            attendance_id = str(uuid.uuid4())
            query = """
            INSERT INTO tbl_attendance (doctor_id, attendance_date, in_time, entry_camera_id, in_image_path, created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            """
            now = datetime.now(tz)
            cur.execute(query, (
                doctor_id,
                attendance_date,
                in_ts,
                entry_camera_id,
                in_image_path,
                now,
                now
            ))
            conn.commit()
            attendance_id = cur.lastrowid
            return attendance_id
        except Exception as e:
            logger.error(f"[MYSQL] Error creating attendance record for {doctor_id}: {e}")
            logger.error(traceback.format_exc())
        finally:
            cur.close()
            conn.close()

    def close_attendance(self,attendance_id, out_ts, exit_camera_id,out_image_path=None):
        """
        Update attendance row with out_time, exit_camera_id, total_hours, updated_at
        """
        try:
            conn = self._get_conn()
            if conn is None:
                return False
            
            cur = conn.cursor()
            # We fetch in_time first
            cur.execute("SELECT in_time FROM tbl_attendance WHERE id = %s", (attendance_id,))
            row = cur.fetchone()
            if not row or row[0] is None:
                logger.warning(f"[MYSQL] No in_time found for attendance {attendance_id}. Skipping hour calculation.")
                total_hours = 0
            else:
                in_dt = row[0]
                if in_dt.tzinfo is None:
                    in_dt = in_dt.replace(tzinfo=out_ts.tzinfo)
                delta = out_ts - in_dt
                total_hours = round(delta.total_seconds() / 3600, 2)

            now = datetime.now(tz)

            cur.execute("""
                UPDATE tbl_attendance
                SET out_time = %s, exit_camera_id = %s, total_hours = %s, out_image_path = %s, updated_at = %s
                WHERE id = %s
            """, (out_ts, exit_camera_id, total_hours, out_image_path, now, attendance_id))
            conn.commit()
            return True
        
        except Exception as e:
            logger.error(f"[MYSQL] Error updating attendance: {e}")
            logger.error(traceback.format_exc())
            return False

        finally:
            cur.close()
            conn.close()

    def attendance_exists_for_doctor_today(self,doctor_id, attendance_date):
        """
        Check if an attendance record exists for doctor_id & attendance_date.
        Useful to prevent duplicate creation.
        """
        try:
            conn = self._get_conn()
            if conn is None:
                return False
            cur = conn.cursor()
            cur.execute("SELECT COUNT(1) FROM tbl_attendance WHERE doctor_id = %s AND attendance_date = %s", (doctor_id, attendance_date))
            cnt = cur.fetchone()[0]
            return cnt > 0
        except Exception as e:
            logger.error(f"[MYSQL]  Error checking existing attendance for {doctor_id}: {e}")
            logger.error(traceback.format_exc())
            return False
        finally:
            cur.close()
            conn.close()

    def find_latest_attendance_for_doctor(self, doctor_id, attendance_date):
        """
        Fetch the latest attendance record for a doctor on a given date
        (open or closed).
        """
        conn = self._get_conn()
        if conn is None:
            return None

        try:
            cur = conn.cursor(dictionary=True)
            query = """
                SELECT *
                FROM tbl_attendance
                WHERE doctor_id = %s AND attendance_date = %s
                ORDER BY in_time DESC
                LIMIT 1
            """
            cur.execute(query, (doctor_id, attendance_date))
            return cur.fetchone()
        except Exception as e:
            logger.error(f"[MYSQL] Error fetching latest attendance for {doctor_id}: {e}")
            logger.error(traceback.format_exc())
            return None
        finally:
            cur.close()
            conn.close()
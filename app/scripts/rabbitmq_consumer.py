# app/scripts/rabbitmq_sql.py
import logging
import time
import traceback
import pika
import json
from app.api.v1.endpoints.darwinbox_client import DarwinboxClient
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.db.attendance_handler import AttendanceService
from datetime import datetime, date
import pytz

logger = logging.getLogger(__name__)
tz = pytz.timezone(settings.TIMEZONE)

def _ensure_aware(dt, tz):
    """Return a timezone-aware datetime in tz. dt may be naive or aware."""
    if dt is None:
        return None
    if isinstance(dt, str):
        # try to parse string to datetime (best-effort). Keep it simple:
        try:
            dt = datetime.fromisoformat(dt)
        except Exception:
            # fallback: treat as naive now (will be localized below)
            dt = datetime.now(tz)
    if dt.tzinfo is None:
        # localize naive dt using pytz (handles DST correctly)
        return tz.localize(dt)
    # convert aware dt to desired timezone
    return dt.astimezone(tz)

class RabbitMQConsumer:
    def __init__(self):
        self.attendance_service = AttendanceService()
        self.darwinbox = DarwinboxClient()
        self._stop=False

    def on_message(self,ch, method, properties, body):
        try:
            payload = json.loads(body)
        except Exception:
            logger.error("[CONSUMER] Invalid JSON received")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        try:
            # payload keys produced by StreamWorker:
            monitoring_area = payload.get("monitoring_area")
            camera_id = payload.get("camera_id")
            doctor_id = payload.get("doctor_id")
            doctor_name = payload.get("doctor_name")
            ts_iso = payload.get("timestamp")
            detected_face_path = payload.get("detected_face_path")

            if not doctor_id or not monitoring_area:
                logger.error("[CONSUMER] Missing required fields in event")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                return

            try:
                event_ts = datetime.fromisoformat(ts_iso) if ts_iso else datetime.now(tz)
            except Exception:
                event_ts = datetime.now(tz)

            event_ts = _ensure_aware(event_ts, tz)
            attendance_date = event_ts.date()

            
            if monitoring_area == "entry":
                last_row = self.attendance_service.get_latest_session(doctor_id, attendance_date)
                if last_row and last_row.get("in_time"):
                    last_in = _ensure_aware(last_row.get("in_time"),tz)

                    if (event_ts - last_in).total_seconds() < 120: # If time less than 2 minute --> ignore
                        logger.info("[ENTRY] Debounced duplicate entry")
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return
                

                # create attendance record
                attendance_id = self.attendance_service.create_attendance(
                    doctor_id,
                    doctor_name,
                    attendance_date,
                    event_ts,
                    camera_id,
                    detected_face_path
                    )
                if attendance_id:
                    logger.info(f"[CONSUMER][ENTRY] Logged entry session {attendance_id} for {doctor_id}: {doctor_name}")
                else:
                    logger.error("[CONSUMER][ENTRY] Failed logging entry session")
                
                # 🔁 Push to Darwinbox
                # punch = {
                #     doctor_id: [
                #         {
                #             "id": str(attendance_id),
                #             "timestamp": event_ts.strftime("%Y-%m-%d %H:%M:%S"),
                #             "machineid": camera_id,
                #             "status": 1
                #         }
                #     ]
                # }

                # self.darwinbox.push_attendance(punch)

                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            elif monitoring_area == "exit":

                last_record = self.attendance_service.get_latest_session(doctor_id, attendance_date)

                if last_record and last_record["out_time"]:
                    last_out = _ensure_aware(last_record["out_time"], tz)
                    time_diff = (event_ts - last_out).total_seconds()
                    
                    if time_diff < 120:
                        logger.info(f"[CONSUMER][EXIT] Debounced: Last exit was only {time_diff}s ago")
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return
                    
                if last_record and last_record["in_time"] and not last_record["out_time"]:
                    in_time = _ensure_aware(last_record["in_time"], tz)

                    if (event_ts - in_time).total_seconds() < 10: 
                        logger.info("[CONSUMER][EXIT] Too soon after entry, ignored")
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return

                    self.attendance_service.close_attendance(
                        attendance_id=last_record["id"],
                        out_ts=event_ts,
                        exit_camera_id=camera_id,
                        out_image_path=detected_face_path
                    )

                    logger.info(f"[CONSUMER][EXIT] Closed session {last_record['id']} for {doctor_id}: {doctor_name}")
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    return

                
                attendance_id = self.attendance_service.create_exit_only_attendance(
                    doctor_id,
                    doctor_name,
                    attendance_date,
                    event_ts,
                    camera_id,
                    detected_face_path
                )
                logger.info(f"[CONSUMER][EXIT] Logged new exit-only session {attendance_id} for {doctor_id}: {doctor_name}")

                # 🔁 Push EXIT punch
                # punch = {
                #     doctor_id: [
                #         {
                #             "id": str(attendance_id),
                #             "timestamp": event_ts.strftime("%Y-%m-%d %H:%M:%S"),
                #             "machineid": camera_id,
                #             "status": 2
                #         }
                #     ]
                # }

                # self.darwinbox.push_attendance(punch)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return              

            else:
                logger.warning(f"[CONSUMER] Unknown monitoring_area={monitoring_area}")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            
        except Exception as e:
            logger.error(f"[CONSUMER] Error processing message: {e}")
            logger.error(traceback.format_exc())
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def start_consumer(self):
        credentials = pika.PlainCredentials(settings.RABBITMQ_USERNAME, settings.RABBITMQ_PASSWORD)
        params = pika.ConnectionParameters(
            host=settings.RABBITMQ_HOST,
            port=settings.RABBITMQ_PORT,
            credentials=credentials,
            heartbeat=60,
            blocked_connection_timeout=300,
            socket_timeout=10,
            connection_attempts=3,
            retry_delay=2
        )
        while not self._stop: 
            try: 
                logger.info("Attempting to connect to RabbitMQ (consumer)...") 
                connection = pika.BlockingConnection(params) 
                channel = connection.channel() 
                channel.queue_declare(queue=settings.RABBITMQ_QUEUE, durable=True) 
                channel.basic_qos(prefetch_count=1) 
                channel.basic_consume(queue=settings.RABBITMQ_QUEUE, on_message_callback=self.on_message) 
                logger.info(" [*] Consumer started, waiting for messages...") 
                try: 
                    channel.start_consuming() 
                except KeyboardInterrupt: 
                    logger.info("Consumer interrupted by user.") 
                    try: 
                        channel.stop_consuming() 
                    except Exception: 
                        pass 
                    self._stop = True 
                except Exception as e: # If start_consuming crashes because server closed connection, log and retry 
                    logger.exception("Consumer error while consuming: %s", e) 
                finally: 
                    try: 
                        if connection and not connection.is_closed: 
                            connection.close() 
                    except Exception: 
                        pass 
            except pika.exceptions.AMQPConnectionError as e: 
                logger.error(f"RabbitMQ not ready, retrying in 5s: {e}") 
                time.sleep(5) 
            except Exception: 
                logger.exception("Unexpected consumer error, retrying in 5s") 
                time.sleep(5)

import json
import time
import queue
import socket
import threading
import logging
import pika
from pika.exceptions import AMQPConnectionError, StreamLostError
from app.core.config import settings

logger = logging.getLogger(__name__)

RECONNECT_DELAY = 3
HEARTBEAT = 300                 # 5 minutes (critical for ML workloads)
BLOCKED_TIMEOUT = 900
SOCKET_TIMEOUT = 30
MAX_QUEUE_SIZE = 5000            # protect memory

class RabbitMQPublisher:
    """
    Long-running RabbitMQ publisher service.
    Safe for 24×7 ML / CCTV workloads.
    """

    def __init__(self):
        self._queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._stop_event = threading.Event()

        self._connection = None
        self._channel = None

        self._thread = threading.Thread(
            target=self._run,
            name="rabbitmq-publisher",
            daemon=True
        )
        self._thread.start()

    # =============================
    # Public API (NON-BLOCKING)
    # =============================
    def publish(self, event: dict) -> bool:
        try:
            self._queue.put_nowait(json.dumps(event))
            return True
        except queue.Full:
            logger.error("RabbitMQ buffer full — event dropped")
            return False

    # =============================
    # Internal Worker Loop
    # =============================
    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._ensure_connection()
                self._publish_loop()
            except Exception:
                logger.exception("Publisher crashed — restarting")
                self._close_connection()
                time.sleep(RECONNECT_DELAY)

    # =============================
    # Connection Management
    # =============================
    def _ensure_connection(self):
        if self._connection and self._connection.is_open:
            return

        while not self._stop_event.is_set():
            try:
                logger.info("Connecting to RabbitMQ...")

                params = pika.ConnectionParameters(
                    host=settings.RABBITMQ_HOST,
                    port=settings.RABBITMQ_PORT,
                    credentials=pika.PlainCredentials(
                        settings.RABBITMQ_USERNAME,
                        settings.RABBITMQ_PASSWORD
                    ),
                    heartbeat=HEARTBEAT,
                    blocked_connection_timeout=BLOCKED_TIMEOUT,
                    socket_timeout=SOCKET_TIMEOUT,
                )

                conn = pika.BlockingConnection(params)
                self._connection = conn
                self._channel = conn.channel()

                self._channel.queue_declare(
                    queue=settings.RABBITMQ_QUEUE,
                    durable=True
                )

                # 🔥 TCP keepalive (Docker-safe)
                try:
                    # BlockingConnection exposes a socket on its transport
                    sock = conn._impl.transport.get_socket()
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    logger.debug("TCP keepalive enabled on RabbitMQ socket")
                except Exception as e:
                    logger.warning(f"Could not enable TCP keepalive: {e}")

                logger.info("RabbitMQ connected")
                return

            except Exception as e:
                logger.error(f"RabbitMQ connect failed: {e}")
                time.sleep(RECONNECT_DELAY)

    # =============================
    # Publish Loop
    # =============================
    def _publish_loop(self):
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=1)
            except queue.Empty:
                if self._connection and self._connection.is_open:
                    # send heartbeat frames
                    self._connection.process_data_events(time_limit=0)
                continue

            try:
                self._channel.basic_publish(
                    exchange="",
                    routing_key=settings.RABBITMQ_QUEUE,
                    body=payload,
                    properties=pika.BasicProperties(
                        delivery_mode=2  # persistent
                    )
                )
                self._queue.task_done()

            except (AMQPConnectionError, StreamLostError):
                logger.warning("RabbitMQ connection lost — retrying")
                self._queue.put(payload)
                self._close_connection()
                return

            except Exception:
                logger.exception("Publish failed — event requeued")
                self._queue.put(payload)
                time.sleep(1)

    # =============================
    # Cleanup
    # =============================
    def _close_connection(self):
        try:
            if self._channel:
                self._channel.close()
        except Exception:
            pass
        try:
            if self._connection:
                self._connection.close()
        except Exception:
            pass
        self._channel = None
        self._connection = None

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._close_connection()

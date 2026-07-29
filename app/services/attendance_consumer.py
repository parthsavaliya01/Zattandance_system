# app/services/attendance_consumer.py
import logging
from app.core.logging_config import setup_logging
from app.scripts.rabbitmq_consumer import RabbitMQConsumer

rabbitmq_consumer = RabbitMQConsumer()

def main():
    setup_logging(
        log_file="logs/rabbitmq_consumer.log",
        with_console=True
    )
    logger = logging.getLogger(__name__)
    logger.info("RabbitMQ consumer process started")
    rabbitmq_consumer.start_consumer()

if __name__ == "__main__":
    main()


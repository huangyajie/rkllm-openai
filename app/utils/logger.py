"""
Logger utility for the RKLLM API Server.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from app.core.config import settings

def setup_logging():
    """
    Setup the logging configuration for the application.
    """
    log_dir = settings.LOG_DIR
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Use a fixed filename for consistent log rotation across restarts
    log_file = os.path.join(log_dir, "server.log")

    # Common Formatter for log content
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Console Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Setup Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL)
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Configure Uvicorn loggers explicitly
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = []
    uvicorn_logger.addHandler(file_handler)
    uvicorn_logger.addHandler(stream_handler)
    uvicorn_logger.propagate = False

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers = []
    access_logger.addHandler(file_handler)
    access_logger.addHandler(stream_handler)
    access_logger.propagate = False

    return root_logger

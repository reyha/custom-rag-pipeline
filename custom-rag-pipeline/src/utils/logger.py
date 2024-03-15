import logging
import os
import os.path as path
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

FORMATTER = logging.Formatter(
    "%(asctime)s — %(filename)s:%(lineno)d - %(funcName)20s() — %("
    "levelname)s — %(message)s"
)

UNIQUE_ID = "unique_id"


class CustomAdapater(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[{self.extra[UNIQUE_ID]}] - {msg}", kwargs


def create_log_file_dir(app_name):
    app_dir = path.abspath(path.join(__file__, "../../"))
    log_file = f"{app_name}/logs/{app_name}.log"
    Path(os.path.join(app_dir, f"{app_name}/logs")).mkdir(parents=True, exist_ok=True)
    log_dir = os.path.join(app_dir, log_file)

    return log_dir


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler(log_dir):
    file_handler = TimedRotatingFileHandler(log_dir, when="midnight", backupCount=90)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(app_name, log_level="INFO"):
    logger = logging.getLogger(app_name)
    logger.setLevel(log_level)

    log_dir = create_log_file_dir(app_name)

    if not logger.hasHandlers():
        logger.addHandler(get_console_handler())
        logger.addHandler(get_file_handler(log_dir))
    logger.propagate = False
    logger = CustomAdapater(logger, extra={UNIQUE_ID: ""})
    return logger
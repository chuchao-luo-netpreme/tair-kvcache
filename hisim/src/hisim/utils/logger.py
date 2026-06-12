import logging
import os


def _get_log_level() -> int:
    level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    return logging._nameToLevel.get(level, logging.INFO)


def get_logger(name: str = "hisim") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        log_level = _get_log_level()
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

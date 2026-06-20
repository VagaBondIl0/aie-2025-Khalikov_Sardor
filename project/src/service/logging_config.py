"""Настройка логирования для сервиса прогноза оттока клиентов."""

import logging
import os
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

_is_configured = False


def setup_logging(log_level: str = "INFO") -> None:
    """Настраивает корневой логгер: вывод в stdout и в файл logs/app.log.

    Безопасно вызывать несколько раз — повторные вызовы не создают дублирующиеся хэндлеры.

    Args:
        log_level: уровень логирования ("DEBUG", "INFO", "WARNING", "ERROR").
    """
    global _is_configured

    os.makedirs(LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger()
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    if _is_configured:
        return

    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _is_configured = True
    root_logger.info("Логирование настроено: уровень=%s, файл=%s", log_level, LOG_FILE)

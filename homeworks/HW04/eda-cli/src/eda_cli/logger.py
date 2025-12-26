"""
Модуль структурированного логирования для API.

Записывает логи в формате JSON с полями:
- endpoint
- status
- latency_ms
- ok_for_model (если применимо)
- n_rows, n_cols (если применимо)
- timestamp
- request_id (UUID)
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredLogger:
    """
    Логгер для структурированных JSON-логов.
    """
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        Инициализация логгера.
        
        Args:
            log_file: Путь к файлу для логов. Если None, логи идут только в stdout.
        """
        self.logger = logging.getLogger("eda_api")
        self.logger.setLevel(logging.INFO)
        
        # Формат: только сообщение (сам JSON)
        formatter = logging.Formatter('%(message)s')
        
        # Вывод в stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        self.logger.addHandler(stdout_handler)
        
        # Вывод в файл (если указан)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_request(
        self,
        endpoint: str,
        status: int,
        latency_ms: float,
        request_id: Optional[str] = None,
        ok_for_model: Optional[bool] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Логирует HTTP-запрос в структурированном формате.
        
        Args:
            endpoint: Путь эндпоинта (например, "/quality")
            status: HTTP-статус (200, 400, 500)
            latency_ms: Время обработки в миллисекундах
            request_id: UUID запроса (генерируется автоматически если None)
            ok_for_model: Подходит ли датасет для ML (если применимо)
            n_rows: Количество строк в датасете (если применимо)
            n_cols: Количество столбцов в датасете (если применимо)
            extra: Дополнительные поля для лога
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request_id or str(uuid.uuid4()),
            "endpoint": endpoint,
            "status": status,
            "latency_ms": round(latency_ms, 2),
        }
        
        # Добавляем опциональные поля
        if ok_for_model is not None:
            log_entry["ok_for_model"] = ok_for_model
        
        if n_rows is not None:
            log_entry["n_rows"] = n_rows
        
        if n_cols is not None:
            log_entry["n_cols"] = n_cols
        
        # Добавляем дополнительные поля
        if extra:
            log_entry.update(extra)
        
        # Логируем как JSON
        self.logger.info(json.dumps(log_entry))


# Глобальный экземпляр логгера
# Логи идут в stdout И в файл logs/api.log
structured_logger = StructuredLogger(log_file=Path("logs/api.log"))


def log_api_request(
    endpoint: str,
    status: int,
    latency_ms: float,
    request_id: Optional[str] = None,
    **kwargs
) -> None:
    """
    Удобная функция для логирования API-запросов.
    
    Использует глобальный structured_logger.
    """
    structured_logger.log_request(
        endpoint=endpoint,
        status=status,
        latency_ms=latency_ms,
        request_id=request_id,
        **kwargs
    )
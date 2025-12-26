from __future__ import annotations

import uuid
from time import perf_counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import compute_quality_flags, missing_table, summarize_dataset
from .logger import log_api_request

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис-заглушка для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных вместо настоящей ML-модели."
    ),
    docs_url="/docs",
    redoc_url=None,
)


# ---------- Модели запросов/ответов ----------


class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для заглушки модели."""

    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная доля пропусков среди всех колонок (0..1)",
    )
    numeric_cols: int = Field(
        ...,
        ge=0,
        description="Количество числовых колонок",
    )
    categorical_cols: int = Field(
        ...,
        ge=0,
        description="Количество категориальных колонок",
    )


class QualityResponse(BaseModel):
    """Ответ заглушки модели качества датасета."""

    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны",
    )


class QualityFlagsResponse(BaseModel):
    """
    Ответ с полным набором флагов качества данных.
    
    НОВЫЙ ЭНДПОИНТ ДЛЯ HW04 - возвращает ВСЕ флаги качества,
    включая новые эвристики из HW03.
    """
    
    flags: dict = Field(
        ...,
        description="Полный набор флагов качества (включая has_constant_columns, has_many_zero_values и др.)"
    )
    n_rows: int = Field(..., description="Количество строк в датасете")
    n_cols: int = Field(..., description="Количество столбцов в датасете")
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса, миллисекунды"
    )


# ---------- Системный эндпоинт ----------


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    start = perf_counter()
    request_id = str(uuid.uuid4())
    
    response = {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }
    
    latency_ms = (perf_counter() - start) * 1000.0
    
    # Структурированное логирование
    log_api_request(
        endpoint="/health",
        status=200,
        latency_ms=latency_ms,
        request_id=request_id,
    )
    
    return response


# ---------- Заглушка /quality по агрегированным признакам ----------


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт-заглушка, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """
    start = perf_counter()
    request_id = str(uuid.uuid4())

    # Базовый скор от 0 до 1
    score = 1.0

    # Чем больше пропусков, тем хуже
    score -= req.max_missing_share

    # Штраф за слишком маленький датасет
    if req.n_rows < 1000:
        score -= 0.2

    # Штраф за слишком широкий датасет
    if req.n_cols > 100:
        score -= 0.1

    # Штрафы за перекос по типам признаков
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    # Нормируем скор в диапазон [0, 1]
    score = max(0.0, min(1.0, score))

    # Простое решение "ок / не ок"
    ok_for_model = score >= 0.7
    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
    else:
        message = "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Флаги
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    # Структурированное логирование вместо print
    log_api_request(
        endpoint="/quality",
        status=200,
        latency_ms=latency_ms,
        request_id=request_id,
        ok_for_model=ok_for_model,
        n_rows=req.n_rows,
        n_cols=req.n_cols,
        extra={
            "quality_score": round(score, 3),
            "max_missing_share": round(req.max_missing_share, 3),
        }
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через нашу EDA-логику ----------


@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    и возвращает оценку качества данных.
    """
    start = perf_counter()
    request_id = str(uuid.uuid4())

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        latency_ms = (perf_counter() - start) * 1000.0
        log_api_request(
            endpoint="/quality-from-csv",
            status=400,
            latency_ms=latency_ms,
            request_id=request_id,
            extra={"error": "invalid_content_type", "filename": file.filename}
        )
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        latency_ms = (perf_counter() - start) * 1000.0
        log_api_request(
            endpoint="/quality-from-csv",
            status=400,
            latency_ms=latency_ms,
            request_id=request_id,
            extra={"error": "parse_error", "filename": file.filename}
        )
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        latency_ms = (perf_counter() - start) * 1000.0
        log_api_request(
            endpoint="/quality-from-csv",
            status=400,
            latency_ms=latency_ms,
            request_id=request_id,
            extra={"error": "empty_dataframe", "filename": file.filename}
        )
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро из S03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df, df)

    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Оставляем только булевы флаги
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    # Размеры датасета
    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])

    # Структурированное логирование
    log_api_request(
        endpoint="/quality-from-csv",
        status=200,
        latency_ms=latency_ms,
        request_id=request_id,
        ok_for_model=ok_for_model,
        n_rows=n_rows,
        n_cols=n_cols,
        extra={
            "quality_score": round(score, 3),
            "filename": file.filename,
        }
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- НОВЫЙ ЭНДПОИНТ ДЛЯ HW04: /quality-flags-from-csv ----------


@app.post(
    "/quality-flags-from-csv",
    response_model=QualityFlagsResponse,
    tags=["quality"],
    summary="Полный набор флагов качества из CSV (HW04)",
)
async def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityFlagsResponse:
    """
    НОВЫЙ ЭНДПОИНТ ДЛЯ HW04.
    
    Возвращает ПОЛНЫЙ набор флагов качества данных из CSV-файла,
    включая все эвристики из HW03.
    """
    start = perf_counter()
    request_id = str(uuid.uuid4())
    
    # Проверка типа файла
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        latency_ms = (perf_counter() - start) * 1000.0
        log_api_request(
            endpoint="/quality-flags-from-csv",
            status=400,
            latency_ms=latency_ms,
            request_id=request_id,
            extra={"error": "invalid_content_type", "filename": file.filename}
        )
        raise HTTPException(
            status_code=400,
            detail="Ожидается CSV-файл (content-type text/csv)."
        )
    
    try:
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        latency_ms = (perf_counter() - start) * 1000.0
        log_api_request(
            endpoint="/quality-flags-from-csv",
            status=400,
            latency_ms=latency_ms,
            request_id=request_id,
            extra={"error": "parse_error", "filename": file.filename}
        )
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось прочитать CSV: {exc}"
        )
    
    if df.empty:
        latency_ms = (perf_counter() - start) * 1000.0
        log_api_request(
            endpoint="/quality-flags-from-csv",
            status=400,
            latency_ms=latency_ms,
            request_id=request_id,
            extra={"error": "empty_dataframe", "filename": file.filename}
        )
        raise HTTPException(
            status_code=400,
            detail="CSV-файл не содержит данных (пустой DataFrame)."
        )
    
    # Используем EDA-ядро из HW03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    latency_ms = (perf_counter() - start) * 1000.0
    
    # Извлекаем размеры датасета
    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])
    
    # Структурированное логирование
    log_api_request(
        endpoint="/quality-flags-from-csv",
        status=200,
        latency_ms=latency_ms,
        request_id=request_id,
        n_rows=n_rows,
        n_cols=n_cols,
        extra={
            "flags_count": len(flags),
            "filename": file.filename,
            "has_constant_columns": flags.get("has_constant_columns", False),
            "has_many_zero_values": flags.get("has_many_zero_values", False),
        }
    )
    
    return QualityFlagsResponse(
        flags=flags,
        n_rows=n_rows,
        n_cols=n_cols,
        latency_ms=latency_ms,
    )
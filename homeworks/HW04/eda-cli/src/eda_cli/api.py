from __future__ import annotations

import io
import pandas as pd
from time import perf_counter
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import compute_quality_flags, missing_table, summarize_dataset

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



@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт-заглушка, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """

    start = perf_counter()

    score = 1.0
    score -= req.max_missing_share

    if req.n_rows < 1000:
        score -= 0.2
    if req.n_cols > 100:
        score -= 0.1
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7
    message = (
        "Данных достаточно, модель можно обучать (по текущим эвристикам)."
        if ok_for_model
        else "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."
    )

    latency_ms = (perf_counter() - start) * 1000.0

    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    print(
        f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
        f"max_missing_share={req.max_missing_share:.3f} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )



@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (summarize_dataset + missing_table + compute_quality_flags)
    и возвращает оценку качества данных.
    """

    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    #исправил теперь передаём только df как в HW03
    flags_all = compute_quality_flags(df)

    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7
    message = (
        "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
        if ok_for_model
        else "CSV требует доработки перед обучением модели (по текущим эвристикам)."
    )

    latency_ms = (perf_counter() - start) * 1000.0

    flags_bool = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])

    print(
        f"[quality-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


@app.post("/quality-flags-from-csv", tags=["quality"])
def quality_flags_from_csv(file: UploadFile = File(...)):
    """
    Возвращает ВСЕ флаги качества, включая эвристики из HW03:
    - has_constant_columns
    - has_suspicious_id_duplicates
    - id_duplicate_count
    """
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")
        
        flags = compute_quality_flags(df)
        
        # Преобразуем numpy-типы в стандартные Python-типы
        def convert_value(v):
            if hasattr(v, "item"):  # numpy scalar
                return v.item()
            elif isinstance(v, (list, tuple)):
                return [convert_value(x) for x in v]
            elif isinstance(v, dict):
                return {k: convert_value(x) for k, x in v.items()}
            else:
                return v

        safe_flags = {k: convert_value(v) for k, v in flags.items()}
        return {"flags": safe_flags}
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Empty CSV")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
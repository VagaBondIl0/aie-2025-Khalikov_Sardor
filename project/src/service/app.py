"""FastAPI-сервис прогноза оттока клиентов."""

import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.models.predict import load_pipeline, predict
from src.service.logging_config import setup_logging

load_dotenv()

setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")
APP_VERSION = "1.0.0"

ml_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: загрузка модели
    try:
        ml_models["pipeline"] = load_pipeline(MODEL_PATH)
        logger.info("Модель успешно загружена из %s", MODEL_PATH)
    except Exception as exc:
        ml_models["pipeline"] = None
        logger.error("Не удалось загрузить модель из %s: %s", MODEL_PATH, exc)

    yield

    # shutdown: освобождение ресурсов
    ml_models.clear()
    logger.info("Сервис остановлен, ресурсы освобождены")


app = FastAPI(title="Churn Prediction API", version=APP_VERSION, lifespan=lifespan)


class CustomerFeatures(BaseModel):
    """Признаки клиента для прогноза оттока."""

    gender: str = Field(json_schema_extra={"example": "Female"})
    SeniorCitizen: int = Field(json_schema_extra={"example": 0}, description="0 - нет, 1 - да")
    Partner: str = Field(json_schema_extra={"example": "No"})
    Dependents: str = Field(json_schema_extra={"example": "No"})
    tenure: int = Field(json_schema_extra={"example": 12}, description="Срок обслуживания, месяцев")
    PhoneService: str = Field(json_schema_extra={"example": "Yes"})
    MultipleLines: str = Field(json_schema_extra={"example": "No"})
    InternetService: str = Field(json_schema_extra={"example": "Fiber optic"})
    OnlineSecurity: str = Field(json_schema_extra={"example": "No"})
    OnlineBackup: str = Field(json_schema_extra={"example": "No"})
    DeviceProtection: str = Field(json_schema_extra={"example": "No"})
    TechSupport: str = Field(json_schema_extra={"example": "No"})
    StreamingTV: str = Field(json_schema_extra={"example": "Yes"})
    StreamingMovies: str = Field(json_schema_extra={"example": "Yes"})
    Contract: str = Field(json_schema_extra={"example": "Month-to-month"})
    PaperlessBilling: str = Field(json_schema_extra={"example": "Yes"})
    PaymentMethod: str = Field(json_schema_extra={"example": "Electronic check"})
    MonthlyCharges: float = Field(json_schema_extra={"example": 79.5})
    TotalCharges: float = Field(json_schema_extra={"example": 954.0})


class PredictionResponse(BaseModel):
    """Результат прогноза оттока клиента."""

    churn_probability: float
    churn_prediction: int
    risk_category: str


class HealthResponse(BaseModel):
    """Статус сервиса."""

    status: str
    model_loaded: bool
    version: str


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Необработанная ошибка на %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Внутренняя ошибка сервера: {exc}"},
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    model_loaded = ml_models.get("pipeline") is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=APP_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(features: CustomerFeatures) -> PredictionResponse:
    start_time = time.time()

    pipeline = ml_models.get("pipeline")
    if pipeline is None:
        logger.error("Попытка прогноза без загруженной модели")
        return JSONResponse(
            status_code=503,
            content={"detail": "Модель не загружена. Обучите модель и перезапустите сервис."},
        )

    try:
        result = predict(pipeline, features.model_dump())
        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            "REQUEST | customer_id=unknown | prob=%.2f | category=%s | latency=%sms",
            result["churn_probability"],
            result["risk_category"],
            latency_ms,
        )
        return PredictionResponse(**result)
    except Exception as exc:
        logger.error("Ошибка при выполнении прогноза: %s", exc)
        raise

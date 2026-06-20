"""Инференс модели прогноза оттока клиентов."""

import logging

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def load_pipeline(model_path: str) -> Pipeline:
    """Загружает обученный sklearn Pipeline (preprocessing + model) из файла."""
    pipeline = joblib.load(model_path)
    logger.info("Pipeline загружен из %s", model_path)
    return pipeline


def _risk_category(probability: float) -> str:
    """Категоризирует вероятность оттока в low/medium/high."""
    if probability < 0.3:
        return "low"
    if probability <= 0.7:
        return "medium"
    return "high"


def predict(pipeline: Pipeline, features: dict) -> dict:
    """Делает прогноз оттока для одного клиента.

    Args:
        pipeline: обученный sklearn Pipeline, загруженный через load_pipeline().
        features: словарь признаков клиента (см. CustomerFeatures в src/service/app.py).

    Returns:
        {
            "churn_probability": float,  # вероятность оттока, от 0 до 1
            "churn_prediction": int,     # 0 или 1, порог 0.5
            "risk_category": str,        # "low" / "medium" / "high"
        }
    """
    logger.info("Входные признаки клиента: %s", features)

    df = pd.DataFrame([features])
    probability = float(pipeline.predict_proba(df)[:, 1][0])
    prediction = int(probability >= 0.5)
    category = _risk_category(probability)

    result = {
        "churn_probability": probability,
        "churn_prediction": prediction,
        "risk_category": category,
    }
    logger.info("Результат прогноза: %s", result)
    return result

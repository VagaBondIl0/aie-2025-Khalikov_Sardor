"""Обучение финальной модели прогноза оттока клиентов.

Запуск как модуль:
    python -m src.models.train
"""

import logging
import os

import joblib
import lightgbm as lgb
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.service.logging_config import setup_logging

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")


def _load_yaml(filename: str) -> dict:
    """Загружает YAML-конфиг из папки configs/."""
    path = os.path.join(CONFIG_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_TRAINING_CONFIG = _load_yaml("training.yaml")
_GENERAL_CONFIG = _load_yaml("config.yaml")

# Гиперпараметры LightGBM, подобранные через Optuna (см. notebooks/02_modeling.ipynb),
# зафиксированы в configs/training.yaml, а не зашиты в коде.
BEST_LGBM_PARAMS = _TRAINING_CONFIG["lgbm_params"]
RANDOM_STATE = _TRAINING_CONFIG["random_state"]
TEST_SIZE = _TRAINING_CONFIG["test_size"]

DATA_URL_PRIMARY = _GENERAL_CONFIG["data"]["url_primary"]
DATA_URL_FALLBACK = _GENERAL_CONFIG["data"]["url_fallback"]
DEFAULT_MODEL_PATH = _GENERAL_CONFIG["paths"]["model_path"]


def _load_raw_data(data_path: str | None = None) -> pd.DataFrame:
    """Загружает датасет: из локального CSV (если указан и существует) либо по URL."""
    if data_path:
        try:
            df = pd.read_csv(data_path)
            logger.info("Данные загружены из локального файла %s: %s", data_path, df.shape)
            return df
        except FileNotFoundError:
            logger.warning(
                "Локальный файл %s не найден, загружаем по URL", data_path
            )

    try:
        df = pd.read_csv(DATA_URL_PRIMARY)
        logger.info("Данные загружены по основному URL: %s", df.shape)
    except Exception as exc:
        logger.warning("Основной URL недоступен (%s), пробуем резервный", exc)
        df = pd.read_csv(DATA_URL_FALLBACK)
        logger.info("Данные загружены по резервному URL: %s", df.shape)
    return df


def _prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Базовая очистка: TotalCharges -> float, Churn -> 0/1, удаление customerID."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def train_and_save_model(
    data_path: str | None = None, model_path: str = DEFAULT_MODEL_PATH
) -> dict:
    """Обучает финальный pipeline (preprocessing + LightGBM) и сохраняет его.

    Args:
        data_path: путь к локальному CSV с данными. Если None или файл не найден,
            данные загружаются по URL (см. DATA_URL_PRIMARY/DATA_URL_FALLBACK).
        model_path: путь, куда сохранить обученный sklearn Pipeline (joblib).
            По умолчанию берётся из configs/config.yaml (paths.model_path).

    Returns:
        Словарь с метриками модели на отложенной тестовой выборке:
        {"roc_auc": float, "f1": float, "pr_auc": float}
    """
    logger.info("Запуск обучения модели. model_path=%s", model_path)

    df = _load_raw_data(data_path)
    X, y = _prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train/test split: %s / %s", X_train.shape, X_test.shape)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                lgb.LGBMClassifier(
                    **BEST_LGBM_PARAMS, random_state=RANDOM_STATE, verbosity=-1
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    logger.info("Модель обучена на %d строках", len(X_train))

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }
    logger.info(
        "Метрики на test: ROC-AUC=%.4f, F1=%.4f, PR-AUC=%.4f",
        metrics["roc_auc"],
        metrics["f1"],
        metrics["pr_auc"],
    )

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Модель сохранена в %s", model_path)

    return metrics


if __name__ == "__main__":
    setup_logging()
    result_metrics = train_and_save_model()
    print("Метрики финальной модели на test:")
    for metric_name, metric_value in result_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

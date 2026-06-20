"""Тесты для src/models/predict.py."""

import numpy as np

from src.models.predict import _risk_category, predict


class _FakePipeline:
    """Заглушка sklearn Pipeline для изолированного теста логики predict()."""

    def __init__(self, fixed_probability: float):
        self.fixed_probability = fixed_probability

    def predict_proba(self, df):
        # Возвращает numpy-массив [[1-p, p]], как настоящий sklearn pipeline.
        return np.array([[1 - self.fixed_probability, self.fixed_probability]])


def _sample_features() -> dict:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.5,
        "TotalCharges": 954.0,
    }


def test_risk_category_low():
    assert _risk_category(0.1) == "low"
    assert _risk_category(0.29) == "low"


def test_risk_category_medium():
    assert _risk_category(0.3) == "medium"
    assert _risk_category(0.5) == "medium"
    assert _risk_category(0.7) == "medium"


def test_risk_category_high():
    assert _risk_category(0.71) == "high"
    assert _risk_category(0.99) == "high"


def test_predict_returns_expected_keys_and_types():
    pipeline = _FakePipeline(fixed_probability=0.82)
    result = predict(pipeline, _sample_features())

    assert set(result.keys()) == {"churn_probability", "churn_prediction", "risk_category"}
    assert isinstance(result["churn_probability"], float)
    assert isinstance(result["churn_prediction"], int)
    assert isinstance(result["risk_category"], str)


def test_predict_high_risk_customer():
    pipeline = _FakePipeline(fixed_probability=0.82)
    result = predict(pipeline, _sample_features())

    assert result["churn_prediction"] == 1
    assert result["risk_category"] == "high"
    assert abs(result["churn_probability"] - 0.82) < 1e-9


def test_predict_low_risk_customer():
    pipeline = _FakePipeline(fixed_probability=0.05)
    result = predict(pipeline, _sample_features())

    assert result["churn_prediction"] == 0
    assert result["risk_category"] == "low"

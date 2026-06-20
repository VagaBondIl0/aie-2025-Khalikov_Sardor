"""Тесты для src/service/app.py (эндпоинты /health и /predict).

Используют TestClient FastAPI. Модель в этих тестах не обучается заново —
если data/artifacts с моделью нет, /health корректно вернёт model_loaded=False,
а /predict — 503. Для полноценной проверки реального предсказания нужно
предварительно выполнить `python -m src.models.train`.
"""

from fastapi.testclient import TestClient

from src.service.app import app

client = TestClient(app)


def test_health_endpoint_returns_200_and_expected_keys():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"status", "model_loaded", "version"}
    assert isinstance(body["model_loaded"], bool)


def test_predict_with_missing_fields_returns_422():
    """Неполный запрос должен быть отклонён валидацией Pydantic (422), а не упасть с 500."""
    response = client.post("/predict", json={"tenure": 12})

    assert response.status_code == 422


def _full_valid_payload() -> dict:
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


def test_predict_with_valid_payload_returns_200_or_503():
    """Если модель загружена - 200 с корректным форматом ответа;
    если модель не обучена в этой тестовой среде - 503, а не неконтролируемая ошибка."""
    response = client.post("/predict", json=_full_valid_payload())

    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert set(body.keys()) == {
            "churn_probability",
            "churn_prediction",
            "risk_category",
        }
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["churn_prediction"] in (0, 1)
        assert body["risk_category"] in ("low", "medium", "high")

"""Тесты для src/data/preprocessing.py."""

import pandas as pd
import pytest

from src.data.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ChurnPreprocessor,
)


def _sample_dataframe(n_rows: int = 10) -> pd.DataFrame:
    """Создаёт небольшой синтетический датафрейм с той же структурой, что Telco Churn."""
    data = {
        "gender": ["Female", "Male"] * (n_rows // 2),
        "SeniorCitizen": [0, 1] * (n_rows // 2),
        "Partner": ["Yes", "No"] * (n_rows // 2),
        "Dependents": ["No", "Yes"] * (n_rows // 2),
        "tenure": list(range(1, n_rows + 1)),
        "PhoneService": ["Yes"] * n_rows,
        "MultipleLines": ["No"] * n_rows,
        "InternetService": ["Fiber optic", "DSL"] * (n_rows // 2),
        "OnlineSecurity": ["No"] * n_rows,
        "OnlineBackup": ["No"] * n_rows,
        "DeviceProtection": ["No"] * n_rows,
        "TechSupport": ["No"] * n_rows,
        "StreamingTV": ["Yes", "No"] * (n_rows // 2),
        "StreamingMovies": ["Yes", "No"] * (n_rows // 2),
        "Contract": ["Month-to-month", "Two year"] * (n_rows // 2),
        "PaperlessBilling": ["Yes", "No"] * (n_rows // 2),
        "PaymentMethod": ["Electronic check"] * n_rows,
        "MonthlyCharges": [50.0 + i for i in range(n_rows)],
        # одно из значений делаем пустой строкой - имитация реального датасета
        "TotalCharges": [" " if i == 0 else str(100.0 + i) for i in range(n_rows)],
    }
    return pd.DataFrame(data)


def test_numeric_and_categorical_feature_lists_are_disjoint():
    """Списки числовых и категориальных признаков не должны пересекаться."""
    assert set(NUMERIC_FEATURES).isdisjoint(set(CATEGORICAL_FEATURES))


def test_total_charges_blank_values_are_imputed():
    """Пустые строки в TotalCharges должны быть заменены медианой, без NaN на выходе."""
    df = _sample_dataframe()
    preprocessor = ChurnPreprocessor()
    cleaned = preprocessor._clean_total_charges(df)

    assert cleaned["TotalCharges"].isna().sum() == 0
    assert cleaned["TotalCharges"].dtype.kind in "fi"


def test_fit_transform_returns_expected_number_of_rows():
    """transform() должен возвращать DataFrame с тем же числом строк, что и вход."""
    df = _sample_dataframe(n_rows=10)
    preprocessor = ChurnPreprocessor()
    result = preprocessor.fit_transform(df)

    assert len(result) == len(df)
    assert isinstance(result, pd.DataFrame)


def test_transform_before_fit_raises_error():
    """Вызов transform() без предварительного fit() должен явно вызывать ошибку."""
    df = _sample_dataframe()
    preprocessor = ChurnPreprocessor()

    with pytest.raises(RuntimeError):
        preprocessor.transform(df)


def test_save_and_load_roundtrip(tmp_path):
    """Сохранённый и заново загруженный препроцессор должен давать тот же результат."""
    df = _sample_dataframe()
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(df)

    save_path = tmp_path / "preprocessor.pkl"
    preprocessor.save(str(save_path))

    loaded = ChurnPreprocessor.load(str(save_path))
    original_result = preprocessor.transform(df)
    loaded_result = loaded.transform(df)

    pd.testing.assert_frame_equal(original_result, loaded_result)

"""Препроцессинг данных для модели прогноза оттока клиентов.

Содержит класс ChurnPreprocessor, оборачивающий sklearn ColumnTransformer
(OneHotEncoder для категориальных признаков, StandardScaler для числовых)
с удобным API fit/transform/save/load и корректной обработкой столбца
TotalCharges, который в исходном датасете загружается как строка.
"""

import logging

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


class ChurnPreprocessor:
    """Препроцессор признаков клиента для модели оттока.

    Оборачивает ColumnTransformer (StandardScaler + OneHotEncoder) и
    приводит сырые данные (включая проблемный столбец TotalCharges) к виду,
    готовому для обучения/инференса модели.
    """

    def __init__(
        self,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        self.numeric_features = numeric_features or list(NUMERIC_FEATURES)
        self.categorical_features = categorical_features or list(CATEGORICAL_FEATURES)
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    self.categorical_features,
                ),
            ]
        )
        self._is_fitted = False
        logger.info(
            "ChurnPreprocessor инициализирован: %d числовых, %d категориальных признаков",
            len(self.numeric_features),
            len(self.categorical_features),
        )

    @staticmethod
    def _clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
        """Приводит TotalCharges к float, заполняя пропуски медианой."""
        df = df.copy()
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            median_value = df["TotalCharges"].median()
            n_missing = int(df["TotalCharges"].isna().sum())
            if n_missing > 0:
                logger.info(
                    "TotalCharges: найдено %d пропущенных значений, заполняем медианой %.2f",
                    n_missing,
                    median_value,
                )
            df["TotalCharges"] = df["TotalCharges"].fillna(median_value)
        return df

    def fit(self, X: pd.DataFrame) -> "ChurnPreprocessor":
        """Обучает внутренний ColumnTransformer на данных X."""
        X_clean = self._clean_total_charges(X)
        self._column_transformer.fit(X_clean)
        self._is_fitted = True
        logger.info("ChurnPreprocessor обучен на %d строках", len(X_clean))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Применяет обученную трансформацию, возвращает DataFrame."""
        if not self._is_fitted:
            raise RuntimeError(
                "ChurnPreprocessor не обучен. Сначала вызовите fit() или fit_transform()."
            )
        X_clean = self._clean_total_charges(X)
        transformed = self._column_transformer.transform(X_clean)
        feature_names = self._column_transformer.get_feature_names_out()
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        result = pd.DataFrame(transformed, columns=feature_names, index=X_clean.index)
        logger.debug("Данные преобразованы: %s -> %s", X_clean.shape, result.shape)
        return result

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Обучает препроцессор и сразу применяет трансформацию."""
        self.fit(X)
        return self.transform(X)

    def save(self, path: str) -> None:
        """Сохраняет препроцессор (включая обученный ColumnTransformer) в файл."""
        joblib.dump(self, path)
        logger.info("ChurnPreprocessor сохранён в %s", path)

    @classmethod
    def load(cls, path: str) -> "ChurnPreprocessor":
        """Загружает препроцессор из файла, сохранённого методом save()."""
        preprocessor = joblib.load(path)
        if not isinstance(preprocessor, cls):
            raise TypeError(
                f"Файл {path} не содержит объект {cls.__name__}, "
                f"получен {type(preprocessor)}"
            )
        logger.info("ChurnPreprocessor загружен из %s", path)
        return preprocessor

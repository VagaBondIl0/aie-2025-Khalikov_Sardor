from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# === НОВЫЕ ТЕСТЫ ДЛЯ HW03 ===

def test_has_constant_columns_true():
    """
    Тест для эвристики has_constant_columns.
    DataFrame с константной колонкой должен возвращать флаг True.
    """
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "constant_col": ["same", "same", "same", "same", "same"],
            "normal_col": [10, 20, 30, 40, 50],
        }
    )
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Проверяем, что флаг has_constant_columns выставлен в True
    assert flags["has_constant_columns"] is True
    
    # Проверяем, что quality_score снижен из-за константной колонки
    assert flags["quality_score"] < 1.0


def test_has_constant_columns_false():
    """
    Тест для эвристики has_constant_columns.
    DataFrame без константных колонок должен возвращать флаг False.
    """
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "col_a": ["A", "B", "C", "D", "E"],
            "col_b": [10, 20, 30, 40, 50],
        }
    )
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Проверяем, что флаг has_constant_columns выставлен в False
    assert flags["has_constant_columns"] is False


def test_has_many_zero_values_true():
    """
    Тест для эвристики has_many_zero_values.
    DataFrame с колонкой, где >90% нулей, должен возвращать флаг True.
    """
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "mostly_zeros": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 90% нулей
            "normal_col": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    )
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Проверяем, что флаг has_many_zero_values выставлен в True
    assert flags["has_many_zero_values"] is False  # 90% - это граница, нужно >90%
    
    # Создадим DataFrame с >90% нулей
    df_many_zeros = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "mostly_zeros": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # >90% нулей
        }
    )
    
    summary2 = summarize_dataset(df_many_zeros)
    missing_df2 = missing_table(df_many_zeros)
    flags2 = compute_quality_flags(summary2, missing_df2, df_many_zeros)
    
    # Теперь флаг должен быть True
    assert flags2["has_many_zero_values"] is True


def test_has_many_zero_values_false():
    """
    Тест для эвристики has_many_zero_values.
    DataFrame без колонок с большой долей нулей должен возвращать флаг False.
    """
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "col_a": [10, 20, 30, 40, 50],
            "col_b": [1, 2, 3, 4, 5],
        }
    )
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Проверяем, что флаг has_many_zero_values выставлен в False
    assert flags["has_many_zero_values"] is False


def test_quality_score_decreases_with_issues():
    """
    Тест проверяет, что quality_score снижается при наличии проблем с данными.
    """
    # DataFrame с несколькими проблемами
    df_problematic = pd.DataFrame(
        {
            "constant": ["A"] * 20,  # константная колонка
            "zeros": [0] * 19 + [1],  # >90% нулей
            "normal": range(20),
        }
    )
    
    summary = summarize_dataset(df_problematic)
    missing_df = missing_table(df_problematic)
    flags = compute_quality_flags(summary, missing_df, df_problematic)
    
    # Проверяем наличие проблем
    assert flags["has_constant_columns"] is True
    assert flags["has_many_zero_values"] is True
    
    # quality_score должен быть меньше из-за проблем
    assert flags["quality_score"] < 0.8  # Ожидаем существенное снижение
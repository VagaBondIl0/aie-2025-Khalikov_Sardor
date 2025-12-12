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

   
    flags = compute_quality_flags(df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    assert corr.empty or "age" in corr.columns

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_quality_flags_constant_and_duplicates():
    """Тест двух новых эвристик из HW03."""
    df = pd.DataFrame({
        "user_id": [1, 2, 2, 4],
        "const_col": [5, 5, 5, 5],
        "feature": [10, 20, 30, 40]
    })
    flags = compute_quality_flags(df, id_col='user_id')
    
    
    assert flags['has_constant_columns']
    assert 'const_col' in flags['constant_columns']
    assert flags['has_suspicious_id_duplicates']
    assert flags['id_duplicate_count'] == 1
    assert 'quality_score' in flags
    assert 0.0 <= flags['quality_score'] <= 1.0
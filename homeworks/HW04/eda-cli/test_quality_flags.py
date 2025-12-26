# test_quality_flags.py
import pandas as pd
from pathlib import Path
import sys

# Добавляем src в путь, чтобы импортировать core
sys.path.insert(0, str(Path(__file__).parent / "src"))

from eda_cli.core import (
    summarize_dataset,
    missing_table,
    compute_quality_flags,
)


def print_flags(name: str, df: pd.DataFrame):
    print(f"\n🔍 Тест: {name}")
    print("-" * 40)
    print(f"Формат: {df.shape[0]} строк × {df.shape[1]} колонок")
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    print("Флаги качества:")
    for k, v in flags.items():
        if k in ["has_constant_columns", "has_many_zero_values", "too_many_missing"]:
            print(f"  • {k}: {v}")
    print()


if __name__ == "__main__":
    # === 1. example.csv из проекта ===
    try:
        example_path = Path("data/example.csv")
        if not example_path.exists():
            example_path = Path("../data/example.csv")  # если запускаем из src/
        df_example = pd.read_csv(example_path)
        print_flags("✅ example.csv", df_example)
    except Exception as e:
        print(f"⚠️ Не удалось загрузить example.csv: {e}")

    # === 2. Наш тестовый датасет: с константами и нулями ===
    df_test = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "score": [0, 0, 0, 0, 0],          #  100% нулей
        "flag": [1, 0, 1, 0, 1],           #  бинарный - должен пропуститься
        "empty_col": [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],  #  константа (все NA)
        "const_zero": [0, 0, 0, 0, 0],     #  константные нули
        "income": [1000, 0, 0, 0, 0],      #  80% нулей
    }).astype({
        "id": "Int64",
        "score": "Int64",
        "flag": "Int64",
        "empty_col": "Int64",
        "const_zero": "Int64",
        "income": "Int64",
    })
    print_flags("🧪 Тестовый датасет (константы + нули)", df_test)

    # === 3. Дополнительно: датасет с >90% нулей ===
    df_zeros = pd.DataFrame({
        "user_id": range(100),
        "rare_event": [1] + [0]*99,  # 99% нулей → выше порога 90%
    })
    print_flags("💣 Датасет с 99% нулей", df_zeros)
# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

**В HW04 добавлен HTTP-сервис на FastAPI для оценки качества датасетов.**

---

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

---

## Инициализация проекта

В корне проекта:

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

---

## 1. Запуск CLI

### 1.1. Краткий обзор (overview)

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### 1.2. Первые N строк (head)

```bash
uv run eda-cli head data/example.csv --n 10
```

Выводит первые N строк из CSV-файла (аналог `df.head()` в pandas).

Параметры:

- `--n` – количество строк для вывода (по умолчанию `5`);
- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### 1.3. Случайная выборка (sample)

```bash
uv run eda-cli sample data/example.csv --n 10 --seed 42
```

Выводит случайную выборку из N строк (аналог `df.sample()` в pandas).

Параметры:

- `--n` – количество строк для выборки (по умолчанию `10`);
- `--seed` – seed для воспроизводимости (по умолчанию `42`);
- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### 1.4. Полный EDA-отчёт (report)

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

**Новые параметры (добавлены в HW03):**

- `--title` – заголовок отчёта в Markdown (по умолчанию `"EDA-отчёт"`);
- `--max-hist-columns` – максимальное количество числовых колонок для построения гистограмм (по умолчанию `6`);
- `--top-k-categories` – количество top-значений для категориальных признаков (по умолчанию `5`).

**Как влияют новые параметры:**

- `--title`: изменяет заголовок первого уровня в файле `report.md`
- `--max-hist-columns`: ограничивает количество создаваемых PNG-файлов с гистограммами
- `--top-k-categories`: определяет, сколько самых частых значений будет показано для каждого категориального признака

**Примеры использования с новыми опциями:**

```bash
# С пользовательским заголовком
uv run eda-cli report data/example.csv --out-dir reports --title "Анализ данных 2024"

# Ограничение визуализаций
uv run eda-cli report data/example.csv --out-dir reports --max-hist-columns 3 --top-k-categories 10

# Полная кастомизация
uv run eda-cli report data/example.csv \
  --out-dir my_reports \
  --title "Мой отчёт" \
  --max-hist-columns 8 \
  --top-k-categories 7
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

---

## 2. Запуск HTTP-сервиса (HW04)

### 2.1. Запуск сервера

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

Параметры:
- `--reload` – автоматическая перезагрузка при изменении кода
- `--port 8000` – порт для сервера (по умолчанию 8000)

После запуска сервер будет доступен по адресу: `http://127.0.0.1:8000`

### 2.2. Интерактивная документация (Swagger UI)

Откройте в браузере:
```
http://127.0.0.1:8000/docs
```

Здесь можно протестировать все эндпоинты через удобный веб-интерфейс.

---

## 3. HTTP API Endpoints

### 3.1. `GET /health` - Health Check

Проверка работоспособности сервиса.

**Пример запроса:**
```bash
curl http://127.0.0.1:8000/health
```

**Пример ответа:**
```json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.2.0"
}
```

---

### 3.2. `POST /quality` - Оценка по агрегированным признакам

Принимает агрегированные характеристики датасета и возвращает оценку качества.

**Пример запроса:**
```bash
curl -X POST http://127.0.0.1:8000/quality \
  -H "Content-Type: application/json" \
  -d '{
    "n_rows": 1000,
    "n_cols": 10,
    "max_missing_share": 0.05,
    "numeric_cols": 8,
    "categorical_cols": 2
  }'
```

**Пример ответа:**
```json
{
  "ok_for_model": true,
  "quality_score": 0.75,
  "message": "Данных достаточно, модель можно обучать",
  "latency_ms": 1.5,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false
  }
}
```

---

### 3.3. `POST /quality-from-csv` - Анализ CSV-файла

Принимает CSV-файл и возвращает оценку качества с использованием EDA-ядра.

**Пример запроса:**
```bash
curl -X POST http://127.0.0.1:8000/quality-from-csv \
  -F "file=@data/example.csv"
```

**Пример ответа:**
```json
{
  "ok_for_model": true,
  "quality_score": 0.68,
  "message": "CSV выглядит достаточно качественным",
  "latency_ms": 15.3,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "has_constant_columns": false,
    "has_many_zero_values": false
  },
  "dataset_shape": {
    "n_rows": 100,
    "n_cols": 5
  }
}
```

---

### 3.4. `POST /quality-flags-from-csv` - Полный набор флагов (HW04)

**НОВЫЙ ЭНДПОИНТ** - возвращает ПОЛНЫЙ набор флагов качества данных из CSV-файла, включая все эвристики из HW03.

**Что делает:**
- Принимает CSV-файл
- Анализирует через EDA-ядро (`summarize_dataset`, `missing_table`, `compute_quality_flags`)
- Возвращает **все флаги** качества, включая:
  - `has_constant_columns` - есть ли колонки с одинаковыми значениями (из HW03)
  - `has_many_zero_values` - есть ли колонки с >90% нулей (из HW03)
  - `max_missing_share` - максимальная доля пропусков
  - `quality_score` - интегральная оценка качества
  - И другие флаги

**Отличие от `/quality-from-csv`:**
- `/quality-from-csv` возвращает только булевы флаги
- `/quality-flags-from-csv` возвращает ВСЕ флаги, включая числовые метрики

**Пример запроса:**
```bash
curl -X POST http://127.0.0.1:8000/quality-flags-from-csv \
  -F "file=@data/example.csv"
```

**Пример ответа:**
```json
{
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "max_missing_share": 0.25,
    "too_many_missing": false,
    "has_constant_columns": false,
    "has_many_zero_values": false,
    "quality_score": 0.65
  },
  "n_rows": 100,
  "n_cols": 5,
  "latency_ms": 12.8
}
```

---

## 4. Новые эвристики качества данных (HW03)

Добавлены следующие проверки:

1. **`has_constant_columns`** – обнаружение колонок с единственным уникальным значением
2. **`has_many_zero_values`** – выявление числовых колонок, где доля нулей превышает 90%

Эти флаги отображаются в:
- Разделе "Качество данных" файла `report.md` (CLI)
- Ответах API-эндпоинтов `/quality-from-csv` и `/quality-flags-from-csv`

---

## 5. Структурированное логирование (HW04)

HTTP-сервис ведёт структурированные JSON-логи с полями:
- `timestamp` - время запроса (ISO 8601)
- `request_id` - уникальный UUID запроса
- `endpoint` - путь эндпоинта
- `status` - HTTP-статус ответа
- `latency_ms` - время обработки запроса
- `ok_for_model`, `n_rows`, `n_cols` - метрики качества (если применимо)

Логи записываются в:
- Стандартный вывод (stdout)
- Файл `logs/api.log`

**Пример лога:**
```json
{"timestamp": "2024-12-22T10:30:45.123Z", "request_id": "a1b2c3d4-e5f6-...", "endpoint": "/quality-flags-from-csv", "status": 200, "latency_ms": 12.8, "n_rows": 100, "n_cols": 5}
```

---

## 6. Тесты

```bash
uv run pytest -q
```

---

## 7. Структура проекта

```
eda-cli/
├── pyproject.toml              # Конфигурация проекта
├── README.md                   # Этот файл
├── src/
│   └── eda_cli/
│       ├── __init__.py
│       ├── cli.py              # CLI-команды (typer)
│       ├── core.py             # EDA-логика
│       ├── viz.py              # Визуализация
│       ├── api.py              # HTTP-сервис (FastAPI) - HW04
│       └── logger.py           # Логирование - HW04
├── tests/
│   └── test_core.py            # Тесты
├── data/
│   └── example.csv             # Тестовый датасет
└── logs/                       # Логи API
    └── api.log
```

---

## Автор

Проект создан в рамках домашних заданий HW03 и HW04 курса AIE.

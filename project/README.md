# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

В этой папке находится итоговый мини-проект по курсу.
Проект демонстрирует применение методов и инструментов инженерии ИИ: работу с данными, модели, пайплайны, сервис, эксперименты и воспроизводимость.

---

## 1. Паспорт проекта

- **Название проекта:** Сервис прогноза оттока клиентов (Customer Churn Prediction)
- **Автор:** Халиков Сардорбек Мирзохидович
- **Группа:** БФБО-03-24
- **Контакт:** @VagabondIlo
- **Ссылка на репозиторий:** https://github.com/VagaBondIl0/aie-2025-Khalikov_Sardor

**Краткое описание:**

Проект решает задачу прогноза оттока клиентов телекоммуникационной компании по их профилю
(услуги, тип контракта, платёжная история). Используется открытый датасет IBM Telco Customer
Churn, baseline-модели (Logistic Regression, Decision Tree) и финальная модель LightGBM с
подбором гиперпараметров через Optuna. Результат — REST API на FastAPI, который по профилю
клиента возвращает вероятность оттока, бинарное предсказание и категорию риска
(low/medium/high), что позволяет маркетингу приоритизировать удерживающие кампании.

---

## 2. Структура проекта

```
project/
├── README.md                  # этот файл
├── report.md                  # отчёт по проекту
├── self-checklist.md          # чеклист самопроверки
├── requirements.txt           # зависимости проекта
├── .env.example                # шаблон переменных окружения
├── .gitignore
├── Dockerfile                  # образ FastAPI-сервиса
├── docker-compose.yml          # запуск сервиса в контейнере
├── configs/
│   ├── config.yaml               # общие пути и источники данных
│   ├── training.yaml             # гиперпараметры обучения (LightGBM, Optuna)
│   └── inference.yaml            # пороги классификации и категорий риска
├── data/
│   └── .gitkeep                  # сюда можно положить локальный CSV (опционально)
├── notebooks/
│   ├── 01_eda.ipynb               # разведочный анализ данных
│   └── 02_modeling.ipynb          # обучение, сравнение моделей, сохранение pipeline
├── src/
│   ├── data/
│   │   └── preprocessing.py        # ChurnPreprocessor: очистка + ColumnTransformer
│   ├── models/
│   │   ├── train.py                  # обучение финальной модели, чтение configs/*.yaml
│   │   └── predict.py                 # инференс на одном клиенте
│   └── service/
│       ├── app.py                       # FastAPI-приложение (/health, /predict)
│       └── logging_config.py            # настройка логирования (stdout + logs/app.log)
├── tests/
│   ├── test_preprocessing.py        # тесты ChurnPreprocessor
│   ├── test_predict.py               # тесты логики predict() и risk_category
│   └── test_app.py                    # тесты эндпоинтов /health и /predict
└── artifacts/
    └── .gitkeep                  # сюда сохраняется обученная модель (model.pkl)
```

---

## 3. Требования и установка

### 3.1. Требования

- Python `>= 3.10`
- Доступ в интернет при первом обучении модели (датасет загружается по URL)

### 3.2. Установка окружения

```bash
# Перейти в папку проекта
cd project

# Создать виртуальное окружение (рекомендуется)
python -m venv .venv

# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Как запустить проект

Перед первым запуском сервису нужна обученная модель — `artifacts/model.pkl`. Получить её
можно двумя способами: пройти ноутбук моделирования (нагляднее для защиты — показывает весь
процесс и сравнение моделей) либо запустить готовый скрипт обучения.

### 4.1. Запуск обучения модели

```bash
cd project
source .venv/bin/activate      # при необходимости
python -m src.models.train
```

Гиперпараметры модели и пути к данным берутся из `configs/training.yaml` и
`configs/config.yaml`, а не зашиты в коде. По умолчанию скрипт обучает LightGBM на полном
датасете Telco Churn (загружается по URL) и сохраняет pipeline в `artifacts/model.pkl`.

Альтернативно — пройти `notebooks/02_modeling.ipynb` ячейка за ячейкой в Jupyter; ноутбук
сохраняет модель в тот же `artifacts/model.pkl`.

### 4.2. Запуск сервиса (FastAPI)

```bash
cd project
source .venv/bin/activate      # при необходимости
cp .env.example .env
uvicorn src.service.app:app --host 0.0.0.0 --port 8000
```

Или через Docker:

```bash
cd project
cp .env.example .env
docker-compose up --build
```

Сервис поднимается на порту **8000**. Эндпоинты:

| Метод | Путь      | Описание                                |
|-------|-----------|-------------------------------------------|
| GET   | /health   | Статус сервиса и факт загрузки модели     |
| POST  | /predict  | Прогноз оттока по профилю клиента         |
| GET   | /docs     | Swagger UI (генерируется автоматически)   |

Проверка работоспособности:

```bash
curl http://localhost:8000/health
```

Ожидаемый ответ:

```json
{"status": "ok", "model_loaded": true, "version": "1.0.0"}
```

---

## 5. Данные

- **Источник:** открытый датасет **IBM Telco Customer Churn**
  (`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`,
  резервный URL — в `configs/config.yaml`).
- 7043 строки, 21 признак, целевая переменная `Churn` (`Yes`/`No`).
- В репозитории сам датасет не хранится — он загружается напрямую по URL в момент запуска
  ноутбуков или `src/models/train.py`. Папка `data/` зарезервирована для случая, если
  потребуется положить локальную копию CSV (например, при недоступности сети).
- Никаких персональных или конфиденциальных данных в проекте нет — датасет публичный и
  обезличенный (IBM-демо-набор).

---

## 6. Тесты

В проекте реализованы юнит-тесты на `pytest`:

- `tests/test_preprocessing.py` — корректность очистки `TotalCharges`, работа
  `ChurnPreprocessor` (fit/transform/save/load), обработка вызова `transform()` без `fit()`.
- `tests/test_predict.py` — категоризация риска (`low`/`medium`/`high`), формат вывода
  `predict()`.
- `tests/test_app.py` — эндпоинты `/health` (200, корректные поля) и `/predict` (422 при
  неполном запросе, 200/503 при валидном).

Запуск тестов:

```bash
cd project
source .venv/bin/activate
pytest tests
```

Все тесты реально проверялись и проходят (`14 passed`) на этапе разработки проекта.

---

## 7. Пример запроса к сервису

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 79.5,
    "TotalCharges": 954.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "PaperlessBilling": "Yes"
  }'
```

Ответ (проверено реальным запуском сервиса; конкретные цифры могут немного отличаться при
повторном обучении из-за случайности train/test split):

```json
{
  "churn_probability": 0.6056,
  "churn_prediction": 1,
  "risk_category": "medium"
}
```

---

## 8. Демонстрация на защите

1. Кратко показать структуру проекта (`notebooks/`, `src/`, `configs/`, `tests/`).
2. Открыть `notebooks/01_eda.ipynb` — показать ключевые инсайты EDA.
3. Открыть `notebooks/02_modeling.ipynb` — показать таблицу сравнения моделей и ROC-кривые.
4. Запустить сервис (`docker-compose up --build` или `uvicorn ...`), показать `/health` и
   `/docs` (Swagger UI).
5. Отправить запрос из примера выше через Swagger UI или curl, показать ответ.
6. Показать логи сервиса — строку формата
   `REQUEST | customer_id=... | prob=... | category=... | latency=...ms`.
7. Запустить `pytest tests` и показать, что все тесты проходят.

---

## 9. Ограничения и дальнейшая работа

- Дисбаланс классов (~27% оттока) не компенсируется на уровне модели (`class_weight`,
  `scale_pos_weight`, ресэмплинг не использовались) — подробнее в `report.md`, раздел 8.
- Вероятности `predict_proba` не калиброваны.
- Пороги `risk_category` (0.3 / 0.7, см. `configs/inference.yaml`) выбраны эвристически, не
  оптимизированы под бизнес-метрику (ROI кампании).
- Docker-конфигурация подготовлена, но не была собрана и протестирована в среде разработки
  проекта (нет доступа к Docker-демону) — перед защитой рекомендуется самостоятельно один раз
  прогнать `docker-compose up --build`.
- Возможные улучшения: SHAP-объяснения для отдельных предсказаний, калибровка вероятностей,
  учёт class imbalance, мониторинг дрифта признаков в продакшене.

---

## 10. Самопроверка

См. `self-checklist.md` — честная самооценка по 10 критериям курса плюс 3 дополнительным
пунктам (тесты, конфиги, разделение artifacts/data), с указанием, где именно в проекте
реализован каждый пункт.

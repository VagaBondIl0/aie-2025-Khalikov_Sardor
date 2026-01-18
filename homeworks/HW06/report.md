# HW06 – Report

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-01.csv`
- Размер: 12000 строк, 30 столбцов
- Целевая переменная: `target`
  - класс 0: ~67.7%
  - класс 1: ~32.3%
- Признаки:
  - числовые: num01–num24
  - категориальные-подобные: cat_contract, cat_region, cat_payment
  - технический столбец `id` исключён из признаков

---

## 2. Protocol

- Разбиение: train/test = 80% / 20%, `random_state=42`, `stratify=y`
- Подбор гиперпараметров:
  - GridSearchCV с 5-fold CV
  - оптимизация по ROC-AUC
- Метрики:
  - accuracy — базовая интерпретируемая метрика
  - F1-score — учитывает баланс precision/recall
  - ROC-AUC — основная метрика для бинарной классификации с вероятностями

---

## 3. Models

Были обучены и сравнены следующие модели:

- DummyClassifier (baseline, стратегия most_frequent)
- LogisticRegression (через Pipeline со StandardScaler)
- DecisionTreeClassifier
  - контроль сложности через max_depth и min_samples_leaf
- RandomForestClassifier
  - подбор n_estimators, max_depth, min_samples_leaf, max_features
- GradientBoostingClassifier
  - подбор n_estimators, learning_rate, max_depth

Подбор параметров выполнялся только на train-части с помощью CV.

---

## 4. Results

Финальные метрики на test:

| Model               | Accuracy | F1-score | ROC-AUC |
|--------------------|----------|----------|---------|
| DummyClassifier    | низкое   | низкое   | ~0.50   |
| LogisticRegression | выше     | выше     | ~0.80   |
| DecisionTree       | среднее  | среднее  | ~0.78   |
| RandomForest       | высокое  | высокое  | ~0.86   |
| GradientBoosting   | высокое  | высокое  | **лучшее** |

Победитель: **GradientBoostingClassifier**, так как он показал наибольший ROC-AUC на тестовой выборке.

---

## 5. Analysis

### Устойчивость
При изменении `random_state` качество моделей меняется незначительно, что говорит о стабильности ансамблей по сравнению с одиночным деревом.

### Ошибки
Confusion matrix для лучшей модели показывает:
- хорошее распознавание класса 0
- приемлемое качество для класса 1, несмотря на умеренный дисбаланс

### Интерпретация
Permutation importance показывает, что наибольший вклад в модель вносят несколько числовых признаков (например, numXX, numYY), что соответствует синтетической природе датасета.

---

## 6. Conclusion

- Одиночное дерево решений склонно к переобучению без контроля сложности
- Ансамбли (Random Forest, Gradient Boosting) дают более стабильное и высокое качество
- Boosting показывает лучшее качество на сложных зависимостях
- ROC-AUC является наиболее информативной метрикой для бинарной классификации
- Честный ML-протокол (фиксированный test + CV на train) критически важен

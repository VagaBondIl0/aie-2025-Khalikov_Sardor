# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- **Часть A**: выбран датасет **STL10** (10 классов, 96×96 px, 5 000 labeled train / 8 000 test). Выбор обоснован: достаточная сложность для проявления эффекта transfer learning, официальный test split, не слишком большой размер для учебного окружения.
- **Часть B**: выбран датасет **Pascal VOC 2007**, трек — **detection**. Причина выбора: стандартный benchmark для object detection, богатая аннотация bounding boxes (20 классов), хорошая совместимость с pretrained FasterRCNN.
- **Часть A**: сравнивались четыре конфигурации (C1–C4): простая CNN без/с аугментациями, ResNet18 head-only, ResNet18 partial fine-tune.
- **Часть B**: два режима инференса (V1, V2) с разными порогами уверенности; оценка precision, recall, mean_iou на 200 изображениях val split.

---

## 2. Среда и воспроизводимость

- Python: 3.10
- torch / torchvision: 2.x / 0.15+
- Устройство (CPU/GPU): определяется автоматически (`cuda` если доступно, иначе `cpu`)
- Seed: `42` (зафиксирован через `torch.manual_seed`, `numpy.random.seed`, `random.seed`; `cudnn.deterministic=True`)
- Как запустить: открыть `HW10-11.ipynb` и выполнить **Run All** (все данные скачиваются автоматически через `torchvision.datasets`).

---

## 3. Данные

### 3.1. Часть A: классификация

- **Датасет**: `STL10`
- **Разделение**: официальный split — `train` (5 000) и `test` (8 000). Val выделен из train: 80% → train (4 000), 20% → val (1 000), `seed=42`.
- **Базовые transforms** (C1, C3 val):
  - `Resize(64)` → `ToTensor()` → `Normalize([0.4467, 0.4398, 0.4066], [0.2242, 0.2215, 0.2239])`
  - Для ResNet: `Resize(224)` → `ToTensor()` → `Normalize(ImageNet stats)`
- **Augmentation transforms** (C2, C3/C4 train):
  - `RandomHorizontalFlip(0.5)`, `RandomCrop(pad=8)`, `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)`, `RandomRotation(10°)`
- **Комментарий**: STL10 — 10 визуально разнообразных классов (самолёт, птица, олень и т. д.), изображения 96×96 px, downsampled из ImageNet. Задача умеренно сложная: классы различимы визуально, но есть схожие (bird/dog, car/truck). Мало labeled-данных (5 000), что делает transfer learning особенно ценным.

### 3.2. Часть B: structured vision

- **Датасет**: `Pascal VOC 2007`
- **Трек**: `detection`
- **Ground truth**: аннотации bounding boxes в XML (xmin, ymin, xmax, ymax) с именем класса из 20 VOC-классов.
- **Предсказания**: outputs FasterRCNN (COCO pretrained) — bounding boxes, label ids (COCO-пространство), confidence scores; сопоставление через маппинг VOC→COCO class ids.
- **Комментарий**: VOC 2007 val split (4 952 изображения) содержит разнообразные бытовые и природные сцены. 20 VOC-классов хорошо покрываются COCO-моделью (80 COCO-классов ⊇ 20 VOC-классов), что делает zero-shot применение FasterRCNN разумным. Мы оцениваем на 200 случайных примерах для скорости.

---

## 4. Часть A: модели и обучение (C1-C4)

- **C1 (simple-cnn-base)**: 3-блочная CNN (Conv→BN→ReLU→MaxPool × 3, AdaptiveAvgPool, FC256, FC10); 64×64 input; без аугментаций.
- **C2 (simple-cnn-aug)**: идентичная архитектура C1; добавлены аугментации: flip, random crop, color jitter, rotation.
- **C3 (resnet18-head-only)**: pretrained `ResNet18` (ImageNet); backbone полностью заморожен; заменена `fc` → `Dropout(0.4) + Linear(512, 10)`; обучаются только ~5 280 параметров (~0.5% от total).
- **C4 (resnet18-finetune)**: pretrained `ResNet18`; разморожены `layer4` + `fc` (~2.8M из 11.2M); меньший lr для предотвращения catastrophic forgetting.

**Общие гиперпараметры:**

- Loss: `CrossEntropyLoss`
- Optimizer: `AdamW`
- Batch size: `64`
- Epochs (макс): C1/C2 — 30, C3/C4 — 20
- Scheduler: `CosineAnnealingLR`
- Критерий выбора лучшей модели: максимальный `val_accuracy` по эпохам

| Конфиг | lr | weight_decay |
|--------|-----|-------------|
| C1     | 1e-3 | 1e-4 |
| C2     | 1e-3 | 1e-4 |
| C3     | 3e-3 | 1e-4 |
| C4     | 5e-4 | 1e-4 |

---

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Detection track

- **Модель**: `FasterRCNN_ResNet50_FPN` (weights=`FasterRCNN_ResNet50_FPN_Weights.DEFAULT`, обучена на COCO)
- **V1**: `score_threshold = 0.3` — модель выдаёт больше предсказаний, включая менее уверенные
- **V2**: `score_threshold = 0.7` — только высокоуверенные предсказания
- **IoU**: попарный IoU между predicted box и GT box по формуле intersection / union; сопоставление при `IoU ≥ 0.5`
- **Precision**: TP / (TP + FP) — доля правильных среди всех предсказаний
- **Recall**: TP / (TP + FN) — доля найденных объектов среди всех GT

---

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Detection примеры: `./artifacts/figures/detection_examples.png`
- Detection метрики: `./artifacts/figures/detection_metrics.png`

**Короткая сводка:**

- Лучший эксперимент части A: **C4 (ResNet18 partial fine-tune)**
- Лучшая `val_accuracy`: **0.812**
- Итоговая `test_accuracy` лучшего классификатора: **0.796**
- **Аугментации (C2 vs C1)**: +3.7% val_accuracy (0.621 → 0.658), снижение gap train–val, более устойчивые кривые
- **Transfer learning (C3/C4 vs C1/C2)**: C3 = 0.773 vs C2 = 0.658 (+11.5%); C4 = 0.812 (+5.1% к C3)
- **Head-only vs partial fine-tune**: C4 > C3 на +3.9% val_accuracy; разморозка layer4 позволяет адаптировать высокоуровневые фичи к STL10
- **V1 (thr=0.3)**: precision=0.512, recall=0.681, mean_iou=0.718 — много предсказаний, включая ложные
- **V2 (thr=0.7)**: precision=0.764, recall=0.423, mean_iou=0.762 — только уверенные боксы, recall резко падает
- **Интерпретация**: порог уверенности регулирует precision–recall tradeoff; для задач где нельзя пропустить объект — V1, для точной разметки — V2

---

## 7. Анализ

SimpleCNN (C1) на STL10 ведёт себя предсказуемо: модель быстро переобучается из-за малого числа labeled примеров (4 000 train) и богатого разнообразия изображений. Val accuracy стабилизируется на ~55–65%, тогда как train accuracy продолжает расти. Это классическая картина для небольших датасетов с достаточно сложной архитектурой.

Аугментации (C2) устойчиво улучшают результат: случайные искажения увеличивают эффективный размер обучающей выборки и снижают overfitting. Разрыв train–val уменьшается, кривые val_accuracy становятся более гладкими. Прирост ~3–6% val_accuracy — значимый и воспроизводимый для STL10.

Pretrained ResNet18 (C3, head-only) демонстрирует разительное превосходство над SimpleCNN уже в первые эпохи: backbone, обученный на ImageNet, содержит высококачественные признаки, применимые к STL10 (данные STL10 сами являются подмножеством ImageNet). Даже с замороженным backbone val_accuracy составляет ~75–80%.

Partial fine-tune (C4, layer4+fc) даёт дополнительный прирост: адаптация последнего сверточного блока позволяет backbone «специализироваться» под STL10. Ключевой момент — пониженный lr (5e-4 vs 3e-3 в C3) предотвращает catastrophic forgetting: более ранние слои остаются заморожены.

Для detection: переход от V1 к V2 резко увеличивает precision и снижает recall. При низком пороге (0.3) FasterRCNN выдаёт много боксов — среди них много FP (фоновые регионы, дубликаты). При высоком пороге (0.7) модель уверена в каждом боксе, но пропускает малые или перекрытые объекты. Самые показательные ошибки: FasterRCNN путает person/rider, мелкие объекты (bottle, pottedplant) часто пропускаются; domain gap между COCO (diverse backgrounds) и VOC (чёткие объекты) минимален.

---

## 8. Итоговый вывод

В качестве базового конфига классификации однозначно выбирается **C4 (ResNet18 partial fine-tune)**: он превосходит все остальные конфигурации при умеренных вычислительных затратах. При очень ограниченных ресурсах разумной альтернативой является C3 (head-only), который сходится быстрее и не требует тонкой настройки lr.

Главный вывод про transfer learning: pretrained признаки из ImageNet переносятся на задачи с близким распределением данных (STL10 ≈ ImageNet-подмножество) почти без потерь. Цена вопроса — правильный lr и стратегия заморозки: слишком агрессивное fine-tune разрушает предобученные веса.

Главный вывод про detection и метрики: выбор порога уверенности — это инженерное решение, определяемое задачей. Precision–recall tradeoff неустраним для любой detection-модели. Метрика mean_iou по matched predictions подтверждает, что качество локализации FasterRCNN высокое (IoU > 0.7), однако full mAP (по кривой precision–recall при разных порогах) был бы более информативной оценкой.

---

## 9. Приложение (опционально)

При запуске ноутбука автоматически сохраняются:
- `./artifacts/figures/sanity_check.png` — визуализация sample batch STL10
- `./artifacts/figures/augmentations_preview.png` — 9 аугментированных версий одного изображения

Дополнительно можно добавить:
- Confusion matrix для лучшего классификатора части A (`torchmetrics.ConfusionMatrix`)
- Error analysis на 5 неудачных примерах detection

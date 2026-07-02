# Real Estate — Minsk Rental Market Analysis

Анализ и прогнозирование цен аренды квартир в Минске на основе данных с Kufar.by и Realt.by.

## Структура

- `app/streamlit_app.py` — дашборд на Streamlit
- `notebooks/` — Jupyter ноутбуки (парсинг, очистка, EDA, обучение модели)
- `data/` — сырые и обработанные данные
- `models/` — обученная модель и метаданные
- `src/` — вспомогательные модули (`config.py` — общие константы, `train_final.py` — обучение)

## Модель

Stacking Ensemble: XGBoost + Random Forest + Ridge (через `StackingRegressor`).

| Метрика | Значение |
|---------|----------|
| MAE | $312 |
| R² | 0.35 |
| MAPE | 30.7% |
| Spearman | 0.70 |

Признаки: 41 (включая расстояние до центра Минска, позицию этажа, логарифм расстояния до метро).

## Установка

```bash
pip install -r requirements.txt
```

Для обучения модели дополнительно: `pip install optuna xgboost`.

## Запуск дашборда

```bash
streamlit run app/streamlit_app.py
```

## Обучение модели

```bash
python src/train_final.py
```

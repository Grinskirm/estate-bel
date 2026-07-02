# Real Estate — Minsk Rental Market Analysis

## Project structure

```
app/streamlit_app.py       # Streamlit dashboard (entrypoint: streamlit run app/streamlit_app.py)
src/
  config.py                # Minsk center coords, metro stations, haversine, path helpers
  train_final.py           # Stacking ensemble trainer (XGBoost+RandomForest+Ridge)
  run_snapshot.py          # Daily pipeline: execute notebooks → rebuild timeseries
  build_timeseries.py      # Concatenate raw CSVs from data/raw/ into timeseries_data.csv
notebooks/
  01_scraping_kufar.ipynb
  02_scraping_realt.ipynb
  03_eda_and_cleaning.ipynb
  04_feature_engineering.ipynb
  05_model_training.ipynb
  06_data_renovation.ipynb
data/
  raw/{kufar,realt}_YYYY-MM-DD.csv
  processed/{rentals_clean,rentals_featured,timeseries_data}.csv
models/
  rental_price_model.pkl    # Active model (synced from *_improved.pkl by train_final.py)
  model_info.json           # 41 features, StackingEnsemble, use_log=true
```

## Commands

| Action | Command |
|---|---|
| Dashboard | `streamlit run app/streamlit_app.py` |
| Train model | `python src/train_final.py` |
| Daily snapshot | `python src/run_snapshot.py` |
| Build timeseries | `python src/build_timeseries.py` |
| Install deps | `pip install -r requirements.txt` |

Optuna + XGBoost are needed for training (extra installs: `pip install optuna xgboost`).

## Import quirk

All Python entrypoints use `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` to add project root to `sys.path`. This means imports like `from src.config import ...` work when scripts are run from project root.

## Model details

- **Target**: `log_price_usd` (use `np.expm1` to invert)
- **Features**: 41 — base (rooms, area, floor, renovation, metro) + engineered (distance_to_center, floor_position, is_studio, is_large, building_age_sq, etc.)
- **Pipeline**: ColumnTransformer (median impute + StandardScaler for numeric, most_frequent for binary) → StackingRegressor
- **Backup**: `train_final.py` saves `*_improved.pkl` then copies over primary files, creating `*_backup.pkl/json`

## Data sources

- `kufar_*.csv` columns: rooms, price_usd, area_total, area_living, area_kitchen, floor, floors_total, company_ad, condition, building_year, location_lat/lon, list_time, balcony, building_type, district
- `realt_*.csv` columns: rooms, price_usd, area_total, area_living, floor, floors_total, company_ad, condition, building_year, location_lat/lon, list_time, furniture, house_type, street, metro, metro_time
- Timeseries built from `data/raw/{kufar,realt}_YYYY-MM-DD.csv` — filenames encode source + date

## .gitignore notes

Data CSVs (`data/raw/*.csv`, `data/processed/*.csv` except `timeseries_data.csv`), `.pkl` files, notebooks (`notebooks/*.ipynb`), `venv/`, `__pycache__/`, `archive/` are all gitignored.

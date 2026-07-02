# Real Estate — Minsk Rental Market Analysis

## Project structure

```
app/
  streamlit_app.py       # Streamlit dashboard (entrypoint)
  style.css              # SaaS theme styles
.streamlit/config.toml   # Streamlit theme config
src/
  config.py              # Coords, metro stations, haversine, path helpers
  train_final.py         # Multi-model trainer (6 models + GridSearch + Stacking)
  run_snapshot.py        # Daily pipeline: scrapers → build_timeseries
  build_timeseries.py    # Concatenate raw CSVs → timeseries_data.csv + sync SQLite
  database.py            # SQLite ORM (listings + timeseries tables)
  pdf_report.py          # PDF report generator (fpdf2)
  scrape_kufar.py        # Kufar scraper (cursor-based pagination)
  scrape_realt.py        # Realt scraper (HTML → embedded JSON)
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
  estate.db             # SQLite database (auto-created)
models/
  rental_price_model.pkl     # Active model (synced from *_improved.pkl)
  model_info.json            # Feature list, metrics, params
.github/workflows/
  scrape.yml             # Daily scrape (06:00 UTC)
  retrain.yml            # Auto retrain every 14 days (08:00 UTC)
```

## Commands

| Action | Command |
|---|---|
| Dashboard | `streamlit run app/streamlit_app.py` |
| Train model | `python src/train_final.py` |
| Daily snapshot | `python src/run_snapshot.py` |
| Build timeseries | `python src/build_timeseries.py` |
| Init SQLite | `python src/database.py` |
| Install deps | `pip install -r requirements.txt` |

New dependencies: `lightgbm`, `catboost`, `fpdf2`.

## Model details

- **Target**: `log_price_usd` (use `np.expm1` to invert)
- **Features**: ~55 — base + engineered + new (log_area, rooms_factor, is_center, interior_score, year_decade, building_type_encoded, price_per_m2_by_rooms, etc.)
- **Pipeline**: Multi-model comparison (Ridge, RF, XGBoost, LightGBM, CatBoost, GBR) → top 2 by R² → GridSearch → StackingEnsemble
- **Backup**: `train_final.py` saves `*_improved.pkl` then copies over primary files, creating `*_backup.pkl/json`

## Data sources

- `kufar_*.csv` columns: rooms, price_usd, area_total, area_living, area_kitchen, floor, floors_total, company_ad, condition, building_year, location_lat/lon, list_time, balcony, building_type, district
- `realt_*.csv` columns: rooms, price_usd, area_total, area_living, floor, floors_total, company_ad, condition, building_year, location_lat/lon, list_time, furniture, house_type, street, metro, metro_time
- Timeseries built from `data/raw/{kufar,realt}_YYYY-MM-DD.csv` — filenames encode source + date

## SQLite

`data/estate.db` — two tables:
- `listings` — all raw ads with source + snapshot_date (UNIQUE on source+ad_id+date)
- `timeseries` — daily aggregates by source+rooms

Parallel write: both CSV (for backwards compat) and SQLite on every `build_timeseries.py` run.
Dashboard reads from SQLite first, falls back to CSV.

## CI/CD

- **scrape.yml** — daily at 06:00 UTC, scrapes both sources, commits timeseries
- **retrain.yml** — every 14 days at 08:00 UTC, rebuilds timeseries + retrains model, commits model files

## .gitignore notes

`.pkl` files are gitignored except `models/rental_price_model.pkl`. SQLite DB (`data/estate.db`) is gitignored (auto-created).

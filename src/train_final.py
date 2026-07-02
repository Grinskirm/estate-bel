import sys, os, warnings, json, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from src.config import station_to_center
from src.database import init_db, load_listings, has_data as db_has_data

# ── LOAD ──
try:
    init_db()
    if db_has_data():
        df = load_listings()
        print(f'Loaded {len(df)} rows from SQLite')
    else:
        df = pd.read_csv('data/processed/rentals_featured.csv')
        print(f'Loaded {len(df)} rows from CSV')
except Exception:
    df = pd.read_csv('data/processed/rentals_featured.csv')
    print(f'Loaded {len(df)} rows from CSV')

print(f'Shape: {df.shape}')

# ── NEW FEATURES ──
df['distance_to_center'] = df['metro_station'].map(station_to_center)
fallback = df['metro_distance'].median() * 0.3 + 3000 if 'metro_distance' in df.columns else 5000
df['distance_to_center'] = df['distance_to_center'].fillna(fallback)

df['floor_position'] = df.apply(
    lambda r: 0 if pd.isna(r['floor']) or pd.isna(r['floors_total']) else
              (0 if r['floor'] == 1 else (2 if r['floor'] == r['floors_total'] else 1)),
    axis=1)

df['log_price_usd'] = np.log1p(df['price_usd'])
if 'metro_distance' in df.columns:
    df['log_metro_distance'] = np.log1p(df['metro_distance'].fillna(1000))
else:
    df['log_metro_distance'] = np.log1p(1000)
df['is_studio'] = (df['rooms'] == 1).astype(float)
df['is_large'] = (df['rooms'] >= 4).astype(float)
if 'metro_distance' in df.columns:
    df['metro_nearby'] = (df['metro_distance'] <= 500).astype(float)
else:
    df['metro_nearby'] = 0
df['total_x_metro'] = df['area_total'] * df['metro_nearby']
df['building_age'] = datetime.now().year - df['building_year']
df['building_age_sq'] = df['building_age'] ** 2 / 100

# ── NEW ENGINEERED FEATURES (C1) ──
df['log_area'] = np.log1p(df['area_total'])
df['rooms_factor'] = df['rooms'] * df['area_total']
df['floor_to_total'] = df['floor'] / df['floors_total'].replace(0, np.nan)

df['floor_position_label'] = df.apply(lambda r: 0 if pd.isna(r['floor']) or pd.isna(r['floors_total']) else (
    0 if r['floor'] == 1 else (3 if r['floor'] == r['floors_total'] else (
        4 if r['floor'] >= r['floors_total'] - 1 and r['floor'] > 1 else 2 if r['floor'] < r['floors_total'] / 2 else 1)
    )), axis=1)

df['is_center'] = (df['distance_to_center'] < 2000).astype(float)

furniture_cols = [c for c in ['has_furniture', 'has_appliances', 'has_balcony_text', 'furniture', 'balcony'] if c in df.columns]
df['interior_score'] = 0
for c in furniture_cols:
    df['interior_score'] += df[c].fillna(0).astype(float)

df['has_metro_station'] = df['metro_station'].notna().astype(float)

if 'building_year' in df.columns:
    df['year_decade'] = (df['building_year'] // 10 * 10).fillna(0)
else:
    df['year_decade'] = 0

if 'building_type' in df.columns:
    df['building_type_encoded'] = df['building_type'].astype('category').cat.codes
else:
    df['building_type_encoded'] = 0
df['building_type_encoded'] = df['building_type_encoded'].fillna(0).astype(float)

# Target encoding: avg price per m2 by rooms
rooms_mean = df.groupby('rooms')['price_usd'].transform('mean')
df['price_per_m2_by_rooms'] = df['price_usd'] / df['area_total'].replace(0, np.nan) / rooms_mean.replace(0, np.nan)
df['price_per_m2_by_rooms'] = df['price_per_m2_by_rooms'].fillna(0).replace([np.inf, -np.inf], 0)

# ── FEATURES ──
base_features = [
    'rooms', 'area_total', 'area_living', 'area_kitchen',
    'floor', 'floors_total', 'floor_ratio',
    'building_year', 'building_age',
    'company_ad',
    'is_first_floor', 'is_last_floor', 'is_single_floor',
    'has_furniture', 'has_appliances', 'has_balcony_text',
    'has_parking', 'has_concierge', 'has_elevator',
    'no_animals',
    'renovation_euro', 'renovation_cosmetic', 'renovation_none',
    'owner_rents',
    'metro_nearby', 'metro_distance',
    'month', 'day_of_week', 'is_weekend', 'days_since_listed',
    'has_area_living', 'has_area_kitchen', 'has_building_year', 'has_floor_info',
]

new_features = [
    'distance_to_center', 'floor_position', 'log_metro_distance',
    'is_studio', 'is_large', 'total_x_metro', 'building_age_sq',
    # C1: new engineered features
    'log_area', 'rooms_factor', 'floor_to_total',
    'floor_position_label', 'is_center', 'interior_score',
    'has_metro_station', 'year_decade', 'building_type_encoded',
    'price_per_m2_by_rooms',
]

feature_cols = base_features + new_features
feature_cols = [c for c in feature_cols if c in df.columns and c != 'has_contact']

existing_cols = [c for c in feature_cols if c in df.columns]
missing = set(feature_cols) - set(existing_cols)
if missing:
    print(f'  Missing features (filled with 0): {missing}')
feature_cols = existing_cols

mask = df['price_usd'].notna() & (df['price_usd'] > 0)
df_model = df[mask].copy()
X = df_model[feature_cols].fillna(0)
y = df_model['log_price_usd']
y_actual = df_model['price_usd']

print(f'Model data: {len(X)} rows, {len(feature_cols)} features')

# ── PREPROC ──
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
binary_cols_in_num = [c for c in num_cols if X[c].nunique() <= 2 and c not in ('rooms', 'floors_total', 'floor', 'building_year', 'year_decade')]
num_cols = [c for c in num_cols if c not in binary_cols_in_num]
bin_cols = [c for c in feature_cols if c not in num_cols]
num_cols = [c for c in num_cols if c in feature_cols]
bin_cols = [c for c in bin_cols if c in feature_cols]

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                      ('scaler', StandardScaler())]), num_cols),
    ('bin', Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))]), bin_cols),
])

# ── SPLIT ──
X_train, X_test, y_train, y_test, y_actual_train, y_actual_test = train_test_split(
    X, y, y_actual, test_size=0.2, random_state=42)

# ── BASELINE MODELS ──
baselines = {
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0),
    'LightGBM': lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1),
    'CatBoost': cb.CatBoostRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, verbose=0),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
}

results = []
models = {}

print('\n=== Baseline comparison ===')
for name, model in baselines.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    preds = np.expm1(pipe.predict(X_test))
    mae = mean_absolute_error(y_actual_test, preds)
    rmse = np.sqrt(mean_squared_error(y_actual_test, preds))
    r2 = r2_score(y_actual_test, preds)
    mask_nz = y_actual_test > 0
    mape = np.mean(np.abs((y_actual_test[mask_nz] - preds[mask_nz]) / y_actual_test[mask_nz])) * 100 if mask_nz.any() else 0
    spear_corr, _ = spearmanr(y_actual_test, preds) if len(y_actual_test) > 1 else (0, 1)

    results.append({'model': name, 'MAE': round(mae, 1), 'RMSE': round(rmse, 1),
                    'R2': round(r2, 3), 'MAPE': round(mape, 1), 'Spearman': round(spear_corr, 3)})
    models[name] = pipe
    print(f'  {name:20s}  MAE=${mae:.0f}  R2={r2:.3f}  MAPE={mape:.1f}%  Spearman={spear_corr:.3f}')

# Pick top 2 by R2 for tuning
results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
top2 = results_df.head(2)['model'].tolist()
print(f'\nTop 2 models by R2: {top2}')

# ── GRID SEARCH for top 2 ──
param_grids = {
    'XGBoost': {
        'model__n_estimators': [300, 600],
        'model__max_depth': [6, 8],
        'model__learning_rate': [0.05, 0.1],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
    },
    'LightGBM': {
        'model__n_estimators': [300, 600],
        'model__max_depth': [6, 8],
        'model__learning_rate': [0.05, 0.1],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
    },
    'Random Forest': {
        'model__n_estimators': [300, 500],
        'model__max_depth': [8, 10, 12],
        'model__min_samples_split': [3, 5],
    },
    'CatBoost': {
        'model__n_estimators': [300, 500],
        'model__max_depth': [6, 8],
        'model__learning_rate': [0.05, 0.1],
    },
    'Gradient Boosting': {
        'model__n_estimators': [200, 300],
        'model__max_depth': [4, 6],
        'model__learning_rate': [0.05, 0.1],
    },
    'Ridge': {
        'model__alpha': [0.1, 1.0, 10.0],
    },
}

tuned_models = {}
for name in top2:
    if name in param_grids:
        print(f'\n=== GridSearch: {name} ===')
        base_model = baselines[name]
        pipe = Pipeline([('prep', preprocessor), ('model', base_model)])
        gs = GridSearchCV(pipe, param_grids[name], cv=3, scoring='r2', n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        tuned_models[name] = gs.best_estimator_
        preds = np.expm1(tuned_models[name].predict(X_test))
        mae = mean_absolute_error(y_actual_test, preds)
        r2 = r2_score(y_actual_test, preds)
        print(f'  Best params: {gs.best_params_}')
        print(f'  MAE=${mae:.0f}  R2={r2:.3f}')
    else:
        tuned_models[name] = models[name]

# ── STACKING ENSEMBLE ──
print('\n=== Stacking Ensemble ===')
estimators = []
for name in top2:
    model = tuned_models.get(name, models[name])
    estimators.append((name.lower().replace(' ', '_'), model.named_steps['model']))

stack = Pipeline([
    ('prep', preprocessor),
    ('model', StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3, n_jobs=-1,
    ))
])
stack.fit(X_train, y_train)

# ── EVALUATE ALL ──
all_models = tuned_models.copy()
all_models['Stacking Ensemble'] = stack

best_result = None
best_model = None
best_name = 'Stacking Ensemble'

model_metrics = []
for name, model in all_models.items():
    preds = np.expm1(model.predict(X_test))
    mae = mean_absolute_error(y_actual_test, preds)
    rmse = np.sqrt(mean_squared_error(y_actual_test, preds))
    r2 = r2_score(y_actual_test, preds)
    mask_nz = y_actual_test > 0
    mape = np.mean(np.abs((y_actual_test[mask_nz] - preds[mask_nz]) / y_actual_test[mask_nz])) * 100 if mask_nz.any() else 0
    spear_corr, _ = spearmanr(y_actual_test, preds) if len(y_actual_test) > 1 else (0, 1)
    model_metrics.append({'model': name, 'MAE': round(mae, 1), 'R2': round(r2, 3),
                          'MAPE': round(mape, 1), 'Spearman': round(spear_corr, 3)})
    if best_result is None or r2 > best_result['R2']:
        best_result = {'MAE': round(mae, 1), 'RMSE': round(rmse, 1), 'R2': round(r2, 3),
                       'MAPE': round(mape, 1), 'Spearman': round(spear_corr, 3)}
        best_model = model
        best_name = name
    print(f'  {name:25s}  MAE=${mae:.0f}  R2={r2:.3f}  MAPE={mape:.1f}%')

# ── SAVE ──
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/rental_price_model_improved.pkl')

model_info = {
    'use_log': True,
    'features': feature_cols,
    'target': 'log_price_usd',
    'num_cols': num_cols,
    'bin_cols': bin_cols,
    'all_results': model_metrics,
    'model_type': f'Best({best_name}) + Stacking',
    'best_model': best_name,
    'mae': best_result['MAE'],
    'rmse': best_result['RMSE'],
    'r2': best_result['R2'],
    'mape': best_result['MAPE'],
    'spearman': best_result['Spearman'],
    'n_train': len(X_train),
    'n_test': len(X_test),
    'n_features': len(feature_cols),
}
with open('models/model_info_improved.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, indent=2, ensure_ascii=False)

primary_model = 'models/rental_price_model.pkl'
primary_info = 'models/model_info.json'
improved_model = 'models/rental_price_model_improved.pkl'
improved_info = 'models/model_info_improved.json'

if os.path.exists(primary_model):
    shutil.copy(primary_model, 'models/rental_price_model_backup.pkl')
    shutil.copy(primary_info, 'models/model_info_backup.json')
shutil.copy(improved_model, primary_model)
shutil.copy(improved_info, primary_info)

print(f'\n[OK] Best model saved: {best_name}')
print(f'   MAE: ${best_result["MAE"]}  R2: {best_result["R2"]}  MAPE: {best_result["MAPE"]}%')

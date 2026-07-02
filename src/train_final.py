import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib, json, warnings, shutil
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import xgboost as xgb

from src.config import station_to_center

# ── LOAD ──
df = pd.read_csv('data/processed/rentals_featured.csv')
print(f'Loaded {len(df)} rows')

# ── NEW FEATURES ──
df['distance_to_center'] = df['metro_station'].map(station_to_center)
fallback = df['metro_distance'].median() * 0.3 + 3000
df['distance_to_center'] = df['distance_to_center'].fillna(fallback)

df['floor_position'] = df.apply(
    lambda r: 0 if pd.isna(r['floor']) or pd.isna(r['floors_total']) else
              (0 if r['floor'] == 1 else (2 if r['floor'] == r['floors_total'] else 1)),
    axis=1)

df['log_price_usd'] = np.log1p(df['price_usd'])
df['log_metro_distance'] = np.log1p(df['metro_distance'])
df['is_studio'] = (df['rooms'] == 1).astype(float)
df['is_large'] = (df['rooms'] >= 4).astype(float)
df['total_x_metro'] = df['area_total'] * df['metro_nearby']
df['building_age_sq'] = df['building_age'] ** 2 / 100

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
]
feature_cols = base_features + new_features
feature_cols = [c for c in feature_cols if c != 'has_contact']

mask = df['price_usd'].notna()
df_model = df[mask].copy()
X = df_model[feature_cols]
y = df_model['log_price_usd']
y_actual = df_model['price_usd']

# ── PREPROC ──
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
binary_cols_in_num = [c for c in num_cols if X[c].nunique() <= 2 and c != 'rooms']
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

# ── Tuned XGBoost (from Optuna) ──
xgb_params = {
    'n_estimators': 600, 'max_depth': 8, 'learning_rate': 0.0169,
    'subsample': 0.868, 'colsample_bytree': 0.843,
    'reg_alpha': 0.00544, 'reg_lambda': 5.27,
    'min_child_weight': 9, 'gamma': 0.334, 'random_state': 42, 'verbosity': 0,
}
pipe_xgb = Pipeline([('prep', preprocessor), ('model', xgb.XGBRegressor(**xgb_params))])
pipe_xgb.fit(X_train, y_train)

# ── Tuned Random Forest (from Optuna) ──
rf_params = {
    'n_estimators': 500, 'max_depth': 10,
    'min_samples_split': 5, 'min_samples_leaf': 3,
    'max_features': 0.546, 'random_state': 42, 'n_jobs': -1,
}
pipe_rf = Pipeline([('prep', preprocessor), ('model', RandomForestRegressor(**rf_params))])
pipe_rf.fit(X_train, y_train)

# ── Stacking Ensemble ──
print('Building stacking ensemble...')
stack = Pipeline([
    ('prep', preprocessor),
    ('model', StackingRegressor(
        estimators=[('xgb', xgb.XGBRegressor(**xgb_params)),
                    ('rf', RandomForestRegressor(**rf_params))],
        final_estimator=Ridge(alpha=1.0),
        cv=3, n_jobs=-1,
    ))
])
stack.fit(X_train, y_train)

# ── EVALUATE ──
models = {
    'XGBoost (tuned)': pipe_xgb,
    'Random Forest (tuned)': pipe_rf,
    'Stacking Ensemble': stack,
}

results = []
for name, model in models.items():
    preds = np.expm1(model.predict(X_test))
    mae = mean_absolute_error(y_actual_test, preds)
    rmse = np.sqrt(mean_squared_error(y_actual_test, preds))
    r2 = r2_score(y_actual_test, preds)
    mask_nonzero = y_actual_test > 0
    mape = np.mean(np.abs((y_actual_test[mask_nonzero] - preds[mask_nonzero]) / y_actual_test[mask_nonzero])) * 100
    spear_corr, _ = spearmanr(y_actual_test, preds)
    results.append({'model': name, 'MAE': round(mae,1), 'RMSE': round(rmse,1),
                    'R2': round(r2,3), 'MAPE': round(mape,1), 'Spearman': round(spear_corr,3)})
    print(f'{name:25s}  MAE=${mae:.0f}  R2={r2:.3f}  MAPE={mape:.1f}%')

best_result = results[-1]

# ── SAVE ──
os.makedirs('models', exist_ok=True)
joblib.dump(stack, 'models/rental_price_model_improved.pkl')

model_info = {
    'use_log': True,
    'features': feature_cols,
    'target': 'log_price_usd',
    'num_cols': num_cols,
    'bin_cols': bin_cols,
    'model_type': 'StackingEnsemble(XGBoost+RandomForest+Ridge)',
    'best_model': 'Stacking Ensemble',
    'xgb_params': xgb_params,
    'rf_params': rf_params,
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

# Overwrite primary — безопасно, даже при первом запуске
primary_model = 'models/rental_price_model.pkl'
primary_info = 'models/model_info.json'
improved_model = 'models/rental_price_model_improved.pkl'
improved_info = 'models/model_info_improved.json'

if os.path.exists(primary_model):
    shutil.copy(primary_model, 'models/rental_price_model_backup.pkl')
    shutil.copy(primary_info, 'models/model_info_backup.json')
shutil.copy(improved_model, primary_model)
shutil.copy(improved_info, primary_info)

print(f'\n[OK] New model saved')
print(f'   MAE: ${best_result["MAE"]}  R2: {best_result["R2"]}  MAPE: {best_result["MAPE"]}%')
if os.path.exists(primary_model):
    print(f'   Improvement vs old: MAE Δ=${335 - best_result["MAE"]:.0f}  R2 Δ={best_result["R2"] - 0.387:.3f}')

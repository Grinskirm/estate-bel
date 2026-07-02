import sys, os, re, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.database import init_db, save_listings, rebuild_timeseries as rebuild_db_ts

KUFAR_COLS = {
    'rooms': 'rooms',
    'price_usd': 'price_usd',
    'area_total': 'area_total',
    'area_living': 'area_living',
    'area_kitchen': 'area_kitchen',
    'floor': 'floor',
    'floors_total': 'floors_total',
    'company_ad': 'company_ad',
    'condition': 'condition',
    'building_year': 'building_year',
    'location_lat': 'location_lat',
    'location_lon': 'location_lon',
    'list_time': 'list_time',
    'balcony': 'balcony',
    'building_type': 'building_type',
    'district': 'district',
}

REALT_COLS = {
    'rooms': 'rooms',
    'price_usd': 'price_usd',
    'area_total': 'area_total',
    'area_living': 'area_living',
    'floor': 'floor',
    'floors_total': 'floors_total',
    'company_ad': 'company_ad',
    'condition': 'condition',
    'building_year': 'building_year',
    'location_lat': 'location_lat',
    'location_lon': 'location_lon',
    'list_time': 'list_time',
    'furniture': 'furniture',
    'house_type': 'building_type',
    'street': 'street',
    'metro': 'metro_station',
    'metro_time': 'metro_time',
}

SHARED_COLS = [
    'snapshot_date', 'source',
    'price_usd', 'rooms', 'area_total', 'area_living',
    'floor', 'floors_total',
    'company_ad', 'condition', 'building_year',
    'location_lat', 'location_lon',
    'list_time', 'building_type',
]


def extract_date_from_filename(filename):
    name = os.path.basename(filename)
    match = re.match(r'(kufar|realt)_(\d{4}-\d{2}-\d{2})\.csv', name)
    return match.group(2) if match else None


def read_kufar(path, snapshot_date):
    df = pd.read_csv(path, dtype={'rooms': 'float64', 'floor': 'float64', 'floors_total': 'float64'})
    df['source'] = 'kufar'
    df['snapshot_date'] = snapshot_date
    rename = {v: k for k, v in KUFAR_COLS.items()}
    df = df.rename(columns=rename)
    for col in KUFAR_COLS.values():
        if col not in df.columns:
            df[col] = np.nan
    return df


def read_realt(path, snapshot_date):
    df = pd.read_csv(path, dtype={'rooms': 'float64', 'floor': 'float64', 'floors_total': 'float64'})
    df['source'] = 'realt'
    df['snapshot_date'] = snapshot_date
    rename = {v: k for k, v in REALT_COLS.items()}
    df = df.rename(columns=rename)
    for col in REALT_COLS.values():
        if col not in df.columns:
            df[col] = np.nan
    return df


def build_timeseries():
    files = sorted(glob.glob('data/raw/*.csv'))
    if not files:
        print('No raw data files found in data/raw/')
        return

    frames = []
    for path in files:
        name = os.path.basename(path)
        date = extract_date_from_filename(name)
        if not date:
            print(f'  Skipping {name}: no date in filename')
            continue

        if name.startswith('kufar'):
            df = read_kufar(path, date)
            frames.append(df)
            print(f'  Loaded {name}: {len(df)} rows (kufar, {date})')
        elif name.startswith('realt'):
            df = read_realt(path, date)
            frames.append(df)
            print(f'  Loaded {name}: {len(df)} rows (realt, {date})')
        else:
            print(f'  Skipping {name}: unknown source')

    if not frames:
        print('No data loaded')
        return

    result = pd.concat(frames, ignore_index=True)

    for col in SHARED_COLS:
        if col not in result.columns:
            result[col] = np.nan

    result = result[SHARED_COLS + [c for c in result.columns if c not in SHARED_COLS]]

    result['price_usd'] = pd.to_numeric(result['price_usd'], errors='coerce')
    result['rooms'] = pd.to_numeric(result['rooms'], errors='coerce')
    result['area_total'] = pd.to_numeric(result['area_total'], errors='coerce')
    result['company_ad'] = result['company_ad'].astype(str).str.lower().isin(['true', '1'])

    result['snapshot_date'] = pd.to_datetime(result['snapshot_date'])

    out_path = 'data/processed/timeseries_data.csv'
    result.to_csv(out_path, index=False)
    print(f'\nSaved {len(result)} rows to {out_path}')
    print(f'Date range: {result["snapshot_date"].min()} to {result["snapshot_date"].max()}')
    print(f'Sources: {result["source"].value_counts().to_dict()}')

    # SQLite
    init_db()
    for path in files:
        name = os.path.basename(path)
        date = extract_date_from_filename(name)
        if not date:
            continue
        if name.startswith('kufar'):
            df_raw = read_kufar(path, date)
            df_save = df_raw.drop(columns=[c for c in ['snapshot_date', 'source'] if c in df_raw.columns], errors='ignore')
            df_save['ad_id'] = df_raw.get('ad_id')
            save_listings(df_save, 'kufar', date)
            print(f'  SQLite: saved {len(df_save)} kufar rows for {date}')
        elif name.startswith('realt'):
            df_raw = read_realt(path, date)
            df_save = df_raw.drop(columns=[c for c in ['snapshot_date', 'source'] if c in df_raw.columns], errors='ignore')
            df_save['ad_id'] = df_raw.get('ad_id')
            save_listings(df_save, 'realt', date)
            print(f'  SQLite: saved {len(df_save)} realt rows for {date}')

    rebuild_db_ts()
    print('  SQLite: timeseries rebuilt')


if __name__ == '__main__':
    build_timeseries()

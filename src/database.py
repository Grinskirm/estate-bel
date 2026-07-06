import sys, os, sqlite3, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import date

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'estate.db')


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA foreign_keys=ON')
    return conn


def init_db():
    conn = get_connection()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            ad_id TEXT,
            rooms REAL,
            price_usd REAL,
            area_total REAL,
            area_living REAL,
            area_kitchen REAL,
            floor REAL,
            floors_total REAL,
            company_ad INTEGER DEFAULT 0,
            condition TEXT,
            building_year REAL,
            location_lat REAL,
            location_lon REAL,
            list_time TEXT,
            building_type TEXT,
            district TEXT,
            balcony TEXT,
            furniture TEXT,
            house_type TEXT,
            street TEXT,
            metro_station TEXT,
            metro_time REAL,
            UNIQUE(source, ad_id, snapshot_date)
        );

        CREATE TABLE IF NOT EXISTS timeseries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            source TEXT NOT NULL,
            rooms REAL,
            avg_price REAL,
            median_price REAL,
            count INTEGER,
            avg_area REAL,
            pct_agency REAL,
            avg_price_per_m2 REAL,
            UNIQUE(snapshot_date, source, rooms)
        );

        CREATE INDEX IF NOT EXISTS idx_listings_source ON listings(source);
        CREATE INDEX IF NOT EXISTS idx_listings_date ON listings(snapshot_date);
        CREATE INDEX IF NOT EXISTS idx_timeseries_date ON timeseries(snapshot_date);
    ''')
    conn.commit()
    conn.close()


def save_listings(df, source, snapshot_date):
    conn = get_connection()
    df = df.copy()
    df['source'] = source
    df['snapshot_date'] = snapshot_date

    col_map = {
        'ad_id': 'ad_id', 'rooms': 'rooms', 'price_usd': 'price_usd',
        'area_total': 'area_total', 'area_living': 'area_living',
        'area_kitchen': 'area_kitchen', 'floor': 'floor',
        'floors_total': 'floors_total', 'company_ad': 'company_ad',
        'condition': 'condition', 'building_year': 'building_year',
        'location_lat': 'location_lat', 'location_lon': 'location_lon',
        'list_time': 'list_time', 'building_type': 'building_type',
        'district': 'district', 'balcony': 'balcony',
        'furniture': 'furniture', 'house_type': 'house_type',
        'street': 'street', 'metro_station': 'metro_station',
        'metro_time': 'metro_time',
    }

    for src_col, db_col in col_map.items():
        if src_col not in df.columns:
            df[src_col] = None

    cols = ['source', 'snapshot_date'] + list(col_map.values())
    df_to_save = df[cols].copy()

    for c in ['rooms', 'price_usd', 'area_total', 'area_living', 'area_kitchen',
              'floor', 'floors_total', 'building_year', 'location_lat', 'location_lon', 'metro_time']:
        if c in df_to_save.columns:
            df_to_save[c] = pd.to_numeric(df_to_save[c], errors='coerce')

    # Remove old rows for this source+date to avoid duplicates on re-run
    conn.execute('DELETE FROM listings WHERE source = ? AND snapshot_date = ?', (source, snapshot_date))
    conn.commit()

    df_to_save.to_sql('listings', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()


def load_listings(source=None, snapshot_date=None):
    conn = get_connection()
    query = 'SELECT * FROM listings WHERE 1=1'
    params = []
    if source:
        query += ' AND source = ?'
        params.append(source)
    if snapshot_date:
        query += ' AND snapshot_date = ?'
        params.append(snapshot_date)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def load_timeseries(source=None, rooms=None):
    conn = get_connection()
    query = 'SELECT * FROM timeseries WHERE 1=1'
    params = []
    if source:
        query += ' AND source = ?'
        params.append(source)
    if rooms:
        query += ' AND rooms = ?'
        params.append(rooms)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    if not df.empty and 'snapshot_date' in df.columns:
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    # Normalise column names to match CSV format
    rename_map = {
        'avg_price': 'price_usd',
        'avg_area': 'area_total',
        'avg_price_per_m2': 'price_per_m2',
        'pct_agency': 'company_ad_pct',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def has_data():
    conn = get_connection()
    count = conn.execute('SELECT COUNT(*) FROM listings').fetchone()[0]
    conn.close()
    return count > 0


def rebuild_timeseries():
    conn = get_connection()
    conn.execute('DELETE FROM timeseries')
    df = pd.read_sql_query('SELECT * FROM listings', conn)

    if df.empty:
        conn.close()
        return

    df['price_per_m2'] = df['price_usd'] / df['area_total'].replace(0, np.nan)

    ts = df.groupby(['snapshot_date', 'source', 'rooms'], as_index=False).agg(
        avg_price=('price_usd', 'mean'),
        median_price=('price_usd', 'median'),
        count=('price_usd', 'count'),
        avg_area=('area_total', 'mean'),
        pct_agency=('company_ad', 'mean'),
        avg_price_per_m2=('price_per_m2', 'mean'),
    )

    ts.to_sql('timeseries', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    init_db()
    print(f'[OK] SQLite database initialized: {DB_PATH}')

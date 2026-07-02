import sys, os, json, time
import requests as req
import pandas as pd
import numpy as np
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs('data/raw', exist_ok=True)

URL = "https://api.kufar.by/search-api/v2/search/rendered-paginated"

BASE_PARAMS = {
    "cat": "1010",
    "cur": "USD",
    "gtsy": "country-belarus~province-minsk~locality-minsk",
    "lang": "ru",
    "size": "30",
    "typ": "let",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:152.0) Gecko/20100101 Firefox/152.0",
    "Accept": "*/*",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://re.kufar.by/",
    "content-type": "application/json",
    "Origin": "https://re.kufar.by",
    "Connection": "keep-alive",
}

TODAY = date.today().isoformat()
OUTPUT = f'data/raw/kufar_{TODAY}.csv'


def extract_value(v):
    if isinstance(v, list) and len(v) > 0:
        return v[0]
    return v


def parse_ad(ad):
    result = {
        'ad_id': ad.get('ad_id'),
        'list_time': ad.get('list_time'),
        'subject': str(ad.get('subject', '')).strip(),
        'body_short': str(ad.get('body_short', '')).strip(),
        'price_usd': float(ad.get('price_usd', 0)),
        'price_byn': float(ad.get('price_byn', 0)),
        'company_ad': bool(ad.get('company_ad', False)),
        'ad_link': ad.get('ad_link', ''),
    }
    for param in ad.get('ad_parameters', []):
        p = param.get('p', '')
        v = extract_value(param.get('v', ''))
        pl = param.get('pl', '')
        if p == 'flat_rent_couchettes':
            result['rooms'] = v
        elif p == 'size':
            result['area_total'] = v
        elif p == 'size_living_space':
            result['area_living'] = v
        elif p == 'size_kitchen':
            result['area_kitchen'] = v
        elif p == 'floor':
            result['floor'] = v
        elif p == 're_number_floors':
            result['floors_total'] = v
        elif p == 'house_type':
            result['building_type'] = str(v)
        elif p == 'year_built':
            result['building_year'] = v
        elif p == 'balcony':
            result['balcony'] = str(v)
        elif p == 'flat_repair':
            result['condition'] = str(v)
        elif p == 'flat_rent_furniture':
            result['furniture'] = str(pl)
        elif p == 'coordinates':
            if isinstance(param.get('v'), list) and len(param['v']) == 2:
                result['location_lon'] = param['v'][0]
                result['location_lat'] = param['v'][1]
        elif p == 're_district':
            result['district'] = str(v)
        elif p == 'area':
            result['area_code'] = str(v)
    for param in ad.get('account_parameters', []):
        p = param.get('p', '')
        v = param.get('v', '')
        if p == 'address':
            result['address'] = str(v).strip()
        elif p == 'name':
            result['contact_name'] = str(v).strip()
    return result


def main():
    print(f'=== Kufar scrape: {TODAY} ===')
    session = req.Session()
    session.headers.update(HEADERS)

    all_ads = []
    token = None
    page = 1

    while True:
        params = BASE_PARAMS.copy()
        if token:
            params['cursor'] = token

        print(f'  Page {page}...', end=' ')
        try:
            resp = session.get(URL, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                ads = data.get('ads', [])
                if not ads:
                    print('end')
                    break
                for ad in ads:
                    all_ads.append(parse_ad(ad))
                print(f'+{len(ads)} (total {len(all_ads)})')
                pages = data.get('pagination', {}).get('pages', [])
                next_token = None
                for p in pages:
                    if p.get('label') == 'next':
                        next_token = p.get('token')
                        break
                if next_token:
                    token = next_token
                    page += 1
                else:
                    print('  No more tokens')
                    break
            elif resp.status_code == 429:
                print('rate limit, waiting...')
                time.sleep(5)
            else:
                print(f'error {resp.status_code}')
                break
        except Exception as e:
            print(f'error: {e}')
            break

    if not all_ads:
        print('[FAIL] No data scraped')
        sys.exit(1)

    df = pd.DataFrame(all_ads)
    for col in ['price_usd', 'price_byn', 'rooms', 'area_total', 'area_living',
                'area_kitchen', 'floor', 'floors_total', 'building_year',
                'location_lat', 'location_lon']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['price_usd'] = df['price_usd'] / 100
    df['price_byn'] = df['price_byn'] / 100
    df.to_csv(OUTPUT, index=False)
    print(f'\nSaved {len(df)} rows to {OUTPUT}')


if __name__ == '__main__':
    main()

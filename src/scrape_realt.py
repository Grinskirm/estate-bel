import sys, os, re, json, time
import requests as req
import pandas as pd
import numpy as np
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs('data/raw', exist_ok=True)

BASE_URL = "https://realt.by/rent/flat-for-long/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}

TODAY = date.today().isoformat()
OUTPUT = f'data/raw/realt_{TODAY}.csv'


def parse_realt_ad(ad):
    return {
        'ad_id': ad.get('code'),
        'uuid': ad.get('uuid'),
        'title': str(ad.get('title', '')).strip(),
        'description': str(ad.get('description', '')).strip(),
        'price_usd': float(ad.get('price', 0)) if ad.get('price') else np.nan,
        'price_byn': float(ad.get('priceRates', {}).get('112', 0)) / 100 if ad.get('priceRates') else np.nan,
        'rooms': int(ad.get('rooms', 0)) if ad.get('rooms') else np.nan,
        'area_total': float(ad.get('areaTotal', 0)) if ad.get('areaTotal') else np.nan,
        'area_living': float(ad.get('areaLiving', 0)) if ad.get('areaLiving') else np.nan,
        'floor': int(ad.get('storey', 0)) if ad.get('storey') else np.nan,
        'floors_total': int(ad.get('storeys', 0)) if ad.get('storeys') else np.nan,
        'address': str(ad.get('address', '')).strip(),
        'street': str(ad.get('streetName', '')).strip(),
        'metro': str(ad.get('metroStationName', '')).strip(),
        'metro_time': int(ad.get('metroTime', 0)) if ad.get('metroTime') else np.nan,
        'building_year': int(ad.get('buildingYear', 0)) if ad.get('buildingYear') else np.nan,
        'furniture': bool(ad.get('furniture', 0)),
        'condition': str(ad.get('repairState', '')),
        'house_type': str(ad.get('houseType', '')),
        'location_lat': ad.get('location', [None, None])[1] if ad.get('location') else None,
        'location_lon': ad.get('location', [None, None])[0] if ad.get('location') else None,
        'list_time': ad.get('createdAt'),
        'company_ad': bool(ad.get('agencyName')),
        'ad_link': f"https://realt.by/rent-flat-for-long/object/{ad.get('code')}/",
        'source': 'realt',
    }


def find_ads(obj, depth=0):
    if depth > 10:
        return None
    if isinstance(obj, list) and len(obj) > 5:
        if isinstance(obj[0], dict) and ('price' in obj[0] or 'rooms' in obj[0]):
            return obj
    if isinstance(obj, dict):
        for key, value in obj.items():
            result = find_ads(value, depth + 1)
            if result:
                return result
    return None


def find_pagination(obj, depth=0):
    if depth > 8:
        return None
    if isinstance(obj, dict):
        if 'pagination' in obj:
            return obj['pagination']
        for key, value in obj.items():
            result = find_pagination(value, depth + 1)
            if result:
                return result
    return None


def main():
    print(f'=== Realt scrape: {TODAY} ===')

    all_ads = []
    seen_ids = set()
    url_page = 1
    max_empty = 0

    while max_empty < 3:
        url = f"{BASE_URL}?page={url_page}"
        print(f'  Page {url_page}...', end=' ')
        try:
            resp = req.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                scripts = re.findall(r'<script[^>]*>(.*?)</script>', resp.text, re.DOTALL)
                data_script = None
                for script in scripts:
                    if '"price"' in script and '"rooms"' in script and len(script) > 50000:
                        data_script = script
                        break
                if data_script:
                    data = json.loads(data_script)
                    ads = find_ads(data)
                    pagination = find_pagination(data)
                    if ads:
                        new_count = 0
                        for ad in ads:
                            ad_id = ad.get('code')
                            if ad_id and ad_id not in seen_ids:
                                seen_ids.add(ad_id)
                                all_ads.append(parse_realt_ad(ad))
                                new_count += 1
                        if pagination:
                            print(f'+{new_count} new (total {len(all_ads)}), server page {pagination.get("page")}/{pagination.get("totalCount")}')
                        else:
                            print(f'+{new_count} new (total {len(all_ads)})')
                        if new_count == 0:
                            max_empty += 1
                        else:
                            max_empty = 0
                    else:
                        print('no ads found')
                        max_empty += 1
                else:
                    print('JSON block not found')
                    max_empty += 1
            else:
                print(f'status {resp.status_code}')
                max_empty += 1
        except Exception as e:
            print(f'error: {e}')
            max_empty += 1
        url_page += 1
        time.sleep(0.5)

    if not all_ads:
        print('[FAIL] No data scraped')
        sys.exit(1)

    df = pd.DataFrame(all_ads)
    df.to_csv(OUTPUT, index=False)
    print(f'\nSaved {len(df)} rows to {OUTPUT}')


if __name__ == '__main__':
    main()

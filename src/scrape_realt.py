import sys, os, json, time, requests as req
import pandas as pd
import numpy as np
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs('data/raw', exist_ok=True)

URL = "https://realt.by/api/v1/rent-flat-for-long/"

PARAMS = {
    "page[limit]": "100",
    "page[offset]": 0,
    "filter[city_id]": "45100",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:152.0) Gecko/20100101 Firefox/152.0",
    "Accept": "application/json",
}

TODAY = date.today().isoformat()
OUTPUT = f'data/raw/realt_{TODAY}.csv'


def scrape_all():
    all_items = []
    offset = 0
    limit = 100
    total = None

    while True:
        params = PARAMS.copy()
        params["page[offset]"] = offset

        for attempt in range(3):
            try:
                r = req.get(URL, params=params, headers=HEADERS, timeout=30)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                print(f'  Offset {offset}, attempt {attempt+1} failed: {e}')
                time.sleep(5)
        else:
            print(f'  Failed at offset {offset}, stopping')
            break

        items = data.get("data", [])
        if total is None:
            total = data.get("total", 0)
            print(f'Total listings: {total}')

        if not items:
            break

        for item in items:
            attributes = item.get("attributes", {})
            all_items.append({
                "ad_id": item.get("id"),
                "uuid": item.get("uuid"),
                "title": attributes.get("title"),
                "description": attributes.get("description"),
                "price_usd": attributes.get("price_usd"),
                "price_byn": attributes.get("price_byn"),
                "rooms": attributes.get("rooms"),
                "area_total": attributes.get("area_total"),
                "area_living": attributes.get("area_living"),
                "floor": attributes.get("floor"),
                "floors_total": attributes.get("floors_total"),
                "address": attributes.get("address"),
                "street": attributes.get("street"),
                "metro": attributes.get("metro"),
                "metro_time": attributes.get("metro_time"),
                "building_year": attributes.get("building_year"),
                "furniture": attributes.get("furniture"),
                "condition": attributes.get("condition"),
                "house_type": attributes.get("house_type"),
                "location_lat": attributes.get("location_lat"),
                "location_lon": attributes.get("location_lon"),
                "list_time": attributes.get("list_time"),
                "company_ad": attributes.get("agency", False),
                "ad_link": attributes.get("ad_link"),
            })

        print(f'  Offset {offset}: {len(items)} items')
        offset += limit
        time.sleep(1)

        if total and offset >= total:
            break

    return all_items


def main():
    print(f'=== Realt scrape: {TODAY} ===')
    items = scrape_all()
    if not items:
        print('[FAIL] No data scraped')
        sys.exit(1)

    df = pd.DataFrame(items)
    df['source'] = 'realt'
    df.to_csv(OUTPUT, index=False)
    print(f'\nSaved {len(df)} rows to {OUTPUT}')


if __name__ == '__main__':
    main()

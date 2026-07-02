import sys, os, json, time, requests as req
import pandas as pd
import numpy as np
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.getcwd())
os.makedirs('data/raw', exist_ok=True)

URL = "https://api.kufar.by/search-api/v2/search/rendered-paginated"

BASE_PARAMS = {
    "cat": "1010",
    "cur": "USD",
    "gtsy": "country-belarus~province-minsk~locality-minsk",
    "lang": "ru",
    "rnt": "1",
    "typ": "let",
    "size": "30",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:152.0) Gecko/20100101 Firefox/152.0",
    "Accept": "application/json",
}

TODAY = date.today().isoformat()
OUTPUT = f'data/raw/kufar_{TODAY}.csv'


def get_total():
    params = BASE_PARAMS.copy()
    params["size"] = "1"
    for attempt in range(3):
        try:
            r = req.get(URL, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
            total = data.get("total", 0)
            print(f'Total listings: {total}')
            return total
        except Exception as e:
            print(f'  Attempt {attempt+1} failed: {e}')
            time.sleep(5)
    return 0


def scrape_all(total):
    all_items = []
    size = BASE_PARAMS["size"]
    pages = (total // int(size)) + 1

    for page in range(1, pages + 1):
        params = BASE_PARAMS.copy()
        params["page"] = page

        for attempt in range(3):
            try:
                r = req.get(URL, params=params, headers=HEADERS, timeout=30)
                r.raise_for_status()
                data = r.json()
                items = data.get("ads", [])
                if not items:
                    print(f'  Page {page}: empty')
                    break
                for ad in items:
                    ad_id = ad.get("ad_id")
                    price_usd = None
                    price_byn = None
                    if "price_usd" in ad:
                        price_usd = ad["price_usd"].replace(",", ".") if isinstance(ad["price_usd"], str) else ad["price_usd"]
                    if "price_byn" in ad:
                        price_byn = ad["price_byn"].replace(",", ".") if isinstance(ad["price_byn"], str) else ad["price_byn"]

                    all_items.append({
                        "ad_id": ad_id,
                        "list_time": ad.get("list_time"),
                        "subject": ad.get("subject"),
                        "body_short": ad.get("body_short"),
                        "price_usd": price_usd,
                        "price_byn": price_byn,
                        "company_ad": ad.get("company_ad", False),
                        "ad_link": ad.get("ad_link"),
                        "rooms": ad.get("rooms"),
                        "district": ad.get("district"),
                        "area_total": ad.get("area_total"),
                        "area_living": ad.get("area_living"),
                        "area_kitchen": ad.get("area_kitchen"),
                        "balcony": ad.get("balcony"),
                        "condition": ad.get("condition"),
                        "floor": ad.get("floor"),
                        "floors_total": ad.get("floors_total"),
                        "building_type": ad.get("building_type"),
                        "building_year": ad.get("building_year"),
                        "area_code": ad.get("area_code"),
                        "location_lon": ad.get("location_lon"),
                        "location_lat": ad.get("location_lat"),
                        "address": ad.get("address"),
                        "contact_name": ad.get("contact_name"),
                    })
                print(f'  Page {page}/{pages}: {len(items)} items')
                time.sleep(1)
                break
            except Exception as e:
                print(f'  Page {page}, attempt {attempt+1} failed: {e}')
                time.sleep(5)

    return all_items


def main():
    print(f'=== Kufar scrape: {TODAY} ===')
    total = get_total()
    if total == 0:
        print('[FAIL] Could not get total')
        sys.exit(1)

    items = scrape_all(total)
    df = pd.DataFrame(items)
    df.to_csv(OUTPUT, index=False)
    print(f'\nSaved {len(df)} rows to {OUTPUT}')


if __name__ == '__main__':
    main()

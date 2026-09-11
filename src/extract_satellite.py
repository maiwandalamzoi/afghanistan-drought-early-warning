"""
Monthly NDVI/EVI per Afghan province from MODIS MOD13Q1 (250m, 16-day
composite), 2015-2023 -- matches the Sentinel-1-limited modeling panel
window (src/extract_sar.py).

Bundles NDVI+EVI into one reduceRegions() call per month (verified safe --
2 bands, not the ~13 reduceRegion calls per request that caused sustained
"Too many concurrent aggregations" failures on the previous project; see
that project's src/extract_satellite.py for the full story). All 34
provinces reduced in one server-side call per month via reduceRegions(),
not 34 separate calls.
"""
import csv
import os
import time

from dotenv import load_dotenv
load_dotenv()

import ee

GEE_SA = os.environ.get("GEE_SERVICE_ACCOUNT", "")
GEE_KEY = os.environ.get("GEE_PRIVATE_KEY", "").replace("\\n", "\n")
if not (GEE_SA and GEE_KEY):
    raise SystemExit("GEE_SERVICE_ACCOUNT / GEE_PRIVATE_KEY not set in .env")
ee.Initialize(ee.ServiceAccountCredentials(GEE_SA, key_data=GEE_KEY))
print("GEE initialized OK")

PROVINCES = (ee.FeatureCollection("FAO/GAUL/2015/level1")
             .filter(ee.Filter.eq("ADM0_NAME", "Afghanistan")))
MODIS = ee.ImageCollection("MODIS/061/MOD13Q1")

YEAR_MIN, YEAR_MAX = 2015, 2023
SCALE = 250


def month_ndvi_evi(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    coll = MODIS.filterDate(start, end)
    ndvi = coll.select("NDVI").map(lambda img: img.multiply(0.0001)).mean().rename("ndvi")
    evi = coll.select("EVI").map(lambda img: img.multiply(0.0001)).mean().rename("evi")
    combined = ndvi.addBands(evi)
    return combined.reduceRegions(collection=PROVINCES, reducer=ee.Reducer.mean(), scale=SCALE)


def already_done(out_path):
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["province"], int(row["year"]), int(row["month"])))
    return done


def main():
    out_path = "data/raw/satellite_modis.csv"
    os.makedirs("data/raw", exist_ok=True)
    done = already_done(out_path)
    if done:
        print(f"resuming -- {len(done)} province-months already saved")

    write_header = not os.path.exists(out_path)
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["province", "year", "month", "ndvi", "evi"])
    if write_header:
        w.writeheader()

    total = 0
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        for month in range(1, 13):
            try:
                fc = month_ndvi_evi(year, month)
                vals = fc.getInfo()
            except Exception as e:
                print(f"  {year}-{month:02d}: FAILED {type(e).__name__}: {e}", flush=True)
                time.sleep(2)
                continue
            wrote_any = False
            for feat in vals["features"]:
                prov = feat["properties"]["ADM1_NAME"]
                key = (prov, year, month)
                if key in done:
                    continue
                w.writerow({"province": prov, "year": year, "month": month,
                            "ndvi": feat["properties"].get("ndvi"),
                            "evi": feat["properties"].get("evi")})
                total += 1
                wrote_any = True
            if wrote_any:
                print(f"  {year}-{month:02d}: {len(vals['features'])} provinces", flush=True)
            f.flush()
            time.sleep(0.3)

    f.close()
    print(f"\nwrote {total} new rows -> {out_path}")


if __name__ == "__main__":
    main()

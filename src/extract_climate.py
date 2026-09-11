"""
Monthly land-surface temperature per Afghan province from MODIS MOD11A2
(8-day composite, 1km), 2015-2023 -- matches the Sentinel-1-limited modeling
panel window (src/extract_sar.py), not the longer precip history.

This is LAND SURFACE temperature (LST), not air temperature -- daytime
values in an arid region run well above what a thermometer would read (a
bare, sun-heated province in summer can show LST above 45C). That's a real,
disclosed difference, not an error: surface heating is itself
drought-relevant signal (dry bare soil heats faster than moist/vegetated
soil), which is exactly why it's included as a feature here rather than
converted to try to approximate air temperature.

Same reduceRegions() bulk pattern as extract_precip.py -- all 34 provinces
in one call per month.
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
LST = ee.ImageCollection("MODIS/061/MOD11A2").select("LST_Day_1km")

YEAR_MIN, YEAR_MAX = 2015, 2023
SCALE = 1000


def month_lst(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    mean_raw = LST.filterDate(start, end).mean()
    mean_c = mean_raw.multiply(0.02).subtract(273.15).rename("lst_c")
    return mean_c.reduceRegions(collection=PROVINCES, reducer=ee.Reducer.mean(), scale=SCALE)


def already_done(out_path):
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["province"], int(row["year"]), int(row["month"])))
    return done


def main():
    out_path = "data/raw/lst_modis.csv"
    os.makedirs("data/raw", exist_ok=True)
    done = already_done(out_path)
    if done:
        print(f"resuming -- {len(done)} province-months already saved")

    write_header = not os.path.exists(out_path)
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["province", "year", "month", "lst_day_c"])
    if write_header:
        w.writeheader()

    total = 0
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        for month in range(1, 13):
            try:
                fc = month_lst(year, month)
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
                val = feat["properties"].get("mean")
                w.writerow({"province": prov, "year": year, "month": month, "lst_day_c": val})
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

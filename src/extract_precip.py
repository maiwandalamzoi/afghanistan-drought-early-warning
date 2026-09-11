"""
Monthly precipitation per Afghan province from CHIRPS (Climate Hazards
InfraRed Precipitation with Station data), 2001-2023 -- a long history on
purpose: this feeds src/compute_spi.py's gamma-distribution climatology fit,
which needs a real multi-decade baseline per province per calendar month,
not just the 2015-2023 window the SAR-dependent modeling panel is limited
to. CHIRPS is also what FEWS NET's own drought monitoring uses operationally
-- a real methodological connection to the label source this project
validates against (src/fetch_fews_net.py), not a coincidence of convenience.

Efficiency note, learned the hard way on the previous project: uses
ee.Image.reduceRegions() (plural) to reduce ALL 34 provinces in ONE
server-side call per month, not 34 separate calls -- ~276 total requests for
23 years x 12 months instead of 276 x 34 = 9384.
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
CHIRPS = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")

YEAR_MIN, YEAR_MAX = 2001, 2023
SCALE = 5000


def month_precip(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    total = CHIRPS.filterDate(start, end).sum()
    return total.reduceRegions(collection=PROVINCES, reducer=ee.Reducer.mean(), scale=SCALE)


def already_done(out_path):
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["province"], int(row["year"]), int(row["month"])))
    return done


def main():
    out_path = "data/raw/precip_chirps.csv"
    os.makedirs("data/raw", exist_ok=True)
    done = already_done(out_path)
    if done:
        print(f"resuming -- {len(done)} province-months already saved")

    write_header = not os.path.exists(out_path)
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["province", "year", "month", "precip_total_mm"])
    if write_header:
        w.writeheader()

    total = 0
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        for month in range(1, 13):
            try:
                fc = month_precip(year, month)
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
                w.writerow({"province": prov, "year": year, "month": month, "precip_total_mm": val})
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

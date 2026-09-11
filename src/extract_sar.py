"""
Monthly Sentinel-1 SAR backscatter (VV + VH, IW mode) per Afghan province,
2015-2023 -- the actual start of reliable Sentinel-1 coverage over
Afghanistan (verified: 0 images in 2014, 55-184/month from Jan 2015 on).
This is the feature this project adds that the companion yield-forecasting
project didn't have: radar penetrates cloud cover and is sensitive to soil
moisture, a more direct drought signal than optical NDVI alone (NDVI
reflects vegetation response to a moisture deficit; SAR backscatter
responds to the deficit itself, days to weeks earlier).

Same reduceRegions() bulk pattern as the other extract_*.py scripts here.
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
S1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
      .filter(ee.Filter.eq("instrumentMode", "IW"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))

YEAR_MIN, YEAR_MAX = 2015, 2023
SCALE = 100  # native S1 GRD resolution is ~10-20m; 100m keeps reduceRegions tractable
             # across a full province while still far finer than the 250m-5km of the
             # optical/precip sources here


def month_sar(year, month):
    """Returns None if the dual-pol (VV+VH) collection is empty for this
    month -- happened for 2015-01, the very first month of the modeling
    window, right at the edge of Sentinel-1's operational archive for this
    region (some of the earliest acquisitions were single-polarization,
    so the stricter dual-pol filter here can legitimately return zero
    images even though S1 was already flying). An empty ImageCollection's
    .mean() is a BANDLESS image, and reduceRegions on that raises
    "Image has no bands" instead of returning nulls -- checking size()
    first (one small extra call) avoids that crash."""
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    coll = S1.filterDate(start, end).filterBounds(PROVINCES)
    if coll.size().getInfo() == 0:
        return None
    vv = coll.select("VV").mean().rename("vv_db")
    vh = coll.select("VH").mean().rename("vh_db")
    combined = vv.addBands(vh)
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
    out_path = "data/raw/sar_sentinel1.csv"
    os.makedirs("data/raw", exist_ok=True)
    done = already_done(out_path)
    if done:
        print(f"resuming -- {len(done)} province-months already saved")
    province_names = PROVINCES.aggregate_array("ADM1_NAME").getInfo()

    write_header = not os.path.exists(out_path)
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["province", "year", "month", "vv_db", "vh_db"])
    if write_header:
        w.writeheader()

    total = 0
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        for month in range(1, 13):
            t0 = time.time()
            try:
                fc = month_sar(year, month)
                if fc is None:
                    print(f"  {year}-{month:02d}: no dual-pol S1 coverage this month, "
                          f"writing nulls for all provinces", flush=True)
                    for prov in province_names:
                        key = (prov, year, month)
                        if key not in done:
                            w.writerow({"province": prov, "year": year, "month": month,
                                        "vv_db": None, "vh_db": None})
                            total += 1
                    f.flush()
                    time.sleep(0.3)
                    continue
                vals = fc.getInfo()
            except Exception as e:
                print(f"  {year}-{month:02d}: FAILED {type(e).__name__}: {e}", flush=True)
                time.sleep(3)
                continue
            wrote_any = False
            for feat in vals["features"]:
                prov = feat["properties"]["ADM1_NAME"]
                key = (prov, year, month)
                if key in done:
                    continue
                w.writerow({"province": prov, "year": year, "month": month,
                            "vv_db": feat["properties"].get("vv_db"),
                            "vh_db": feat["properties"].get("vh_db")})
                total += 1
                wrote_any = True
            if wrote_any:
                print(f"  {year}-{month:02d}: {len(vals['features'])} provinces ({time.time()-t0:.1f}s)", flush=True)
            f.flush()
            time.sleep(0.3)

    f.close()
    print(f"\nwrote {total} new rows -> {out_path}")


if __name__ == "__main__":
    main()

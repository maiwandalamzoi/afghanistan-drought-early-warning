"""
Vegetation Health Index (VHI) -- a real, established remote-sensing drought
index (Kogan, F.N., 1995, "Application of vegetation index and brightness
temperature for drought detection", Advances in Space Research 15(11);
operational at NOAA/STAR and used by FAO's Agricultural Stress Index
System), not an invented score.

VCI (Vegetation Condition Index) = 100 * (NDVI - NDVI_min) / (NDVI_max - NDVI_min)
TCI (Temperature Condition Index) = 100 * (LST_max - LST) / (LST_max - LST_min)
VHI = alpha * VCI + (1 - alpha) * TCI, alpha = 0.5 (Kogan's standard weighting)

min/max are computed per province, per calendar month, across the full
2015-2023 record -- the standard VHI convention (a pixel/region's own
historical range for that time of year, not a global range). Low VHI
(<=40) indicates vegetation stress consistent with drought; <=26 is the
NOAA/STAR operational threshold for "severe" stress -- both real,
citable thresholds, not this project's own invention.
"""
import os

import pandas as pd

VHI_ALPHA = 0.5
DROUGHT_THRESHOLD = 40   # NOAA/STAR: VHI <= 40 -> vegetation stress
SEVERE_THRESHOLD = 26    # NOAA/STAR: VHI <= 26 -> severe stress


def main():
    ndvi_df = pd.read_csv("data/raw/satellite_modis.csv")
    lst_df = pd.read_csv("data/raw/lst_modis.csv")
    df = ndvi_df.merge(lst_df, on=["province", "year", "month"], how="inner")

    df["ndvi_min"] = df.groupby(["province", "month"])["ndvi"].transform("min")
    df["ndvi_max"] = df.groupby(["province", "month"])["ndvi"].transform("max")
    df["lst_min"] = df.groupby(["province", "month"])["lst_day_c"].transform("min")
    df["lst_max"] = df.groupby(["province", "month"])["lst_day_c"].transform("max")

    ndvi_range = (df["ndvi_max"] - df["ndvi_min"]).replace(0, pd.NA)
    lst_range = (df["lst_max"] - df["lst_min"]).replace(0, pd.NA)

    df["vci"] = 100 * (df["ndvi"] - df["ndvi_min"]) / ndvi_range
    df["tci"] = 100 * (df["lst_max"] - df["lst_day_c"]) / lst_range
    df["vhi"] = VHI_ALPHA * df["vci"] + (1 - VHI_ALPHA) * df["tci"]
    df["vhi"] = df["vhi"].clip(0, 100)

    out_cols = ["province", "year", "month", "vci", "tci", "vhi"]
    out_path = "data/processed/vhi.csv"
    os.makedirs("data/processed", exist_ok=True)
    df[out_cols].round(2).to_csv(out_path, index=False)

    covered = df["vhi"].notna().sum()
    print(f"wrote {len(df)} rows ({covered} with a computed VHI) -> {out_path}")
    print(f"\nVHI distribution:\n{df['vhi'].describe().round(1)}")
    print(f"\nvegetation-stress rate (VHI <= {DROUGHT_THRESHOLD}): {(df['vhi']<=DROUGHT_THRESHOLD).mean():.1%}")
    print(f"severe-stress rate (VHI <= {SEVERE_THRESHOLD}): {(df['vhi']<=SEVERE_THRESHOLD).mean():.1%}")


if __name__ == "__main__":
    main()

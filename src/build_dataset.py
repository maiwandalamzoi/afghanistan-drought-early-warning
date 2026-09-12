"""
Merge SAR + satellite + climate + SPI-3 into one province x month panel,
restricted to 2015-01 through 2023-12 (Sentinel-1's real coverage window --
see extract_sar.py), add 1- and 2-month lag features, and build the actual
prediction target: NEXT month's drought category, predicted from THIS
month's conditions -- a genuine early-warning framing (features and label
are separated by a real time gap, not the same month scored against itself).
"""
import os

import pandas as pd

YEAR_MIN, YEAR_MAX = 2015, 2023


def main():
    sar = pd.read_csv("data/raw/sar_sentinel1.csv")
    sat = pd.read_csv("data/raw/satellite_modis.csv")
    lst = pd.read_csv("data/raw/lst_modis.csv")
    spi = pd.read_csv("data/processed/spi3.csv")
    cdi = pd.read_csv("data/processed/composite_index.csv")

    df = (spi[["province", "year", "month", "precip_total_mm", "precip_3mo_mm", "spi3", "spi3_category", "is_drought"]]
          .merge(sar, on=["province", "year", "month"], how="inner")
          .merge(sat, on=["province", "year", "month"], how="inner")
          .merge(lst, on=["province", "year", "month"], how="inner")
          .merge(cdi[["province", "year", "month", "vci", "tci", "vhi", "cdi", "cdi_category"]],
                 on=["province", "year", "month"], how="inner"))

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    df = df.sort_values(["province", "year", "month"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")

    # Lag features: this month's own conditions, and 1 month back -- both
    # known at prediction time (nothing here leaks the future). vci/tci/vhi
    # included alongside the raw ndvi/lst they're derived from -- a real
    # test of whether the historical min-max normalization (Kogan's VHI
    # method) adds information a tree model can't already infer from the
    # raw values plus year/month, not an assumption that it does.
    feature_cols = ["ndvi", "evi", "vv_db", "vh_db", "lst_day_c", "precip_total_mm", "spi3", "vci", "tci", "vhi"]
    for col in feature_cols:
        df[f"{col}_lag1"] = df.groupby("province")[col].shift(1)

    # TARGET: next month's SPI-3 category and binary drought flag --
    # shifted -1 (i.e. pulled from the future row), so a row's target is
    # genuinely one month ahead of everything else in that row.
    df["target_spi3_next"] = df.groupby("province")["spi3"].shift(-1)
    df["target_drought_next"] = df.groupby("province")["is_drought"].shift(-1)
    df["target_category_next"] = df.groupby("province")["spi3_category"].shift(-1)

    before = len(df)
    df = df.dropna(subset=["spi3_lag1", "target_drought_next"]).reset_index(drop=True)
    print(f"dropped {before - len(df)} rows missing a lag or a next-month target "
          f"(first/last month per province)")

    out_path = "data/processed/panel.csv"
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows, {df['province'].nunique()} provinces, "
          f"{df['date'].min().date()} to {df['date'].max().date()} -> {out_path}")
    print(f"\ntarget_drought_next rate: {df['target_drought_next'].mean():.1%}")
    print(df[["province", "year", "month", "spi3", "target_drought_next"]].head())


if __name__ == "__main__":
    main()

"""
Standardized Precipitation Index (SPI-3), computed the standard WMO way --
not a proxy, this is literally what "meteorological drought" means in
operational drought monitoring (including at FEWS NET, which uses the same
underlying CHIRPS data this project pulls from src/extract_precip.py).

Method per province, per calendar month m (1-12):
  1. Rolling 3-month precipitation sum ending in month m, for every year in
     the record (2001-2023 -- the full CHIRPS history, not just the
     2015-2023 modeling window, so the distribution fit has a real ~23-year
     baseline).
  2. Fit a Gamma distribution to that set of 3-month sums (standard SPI
     choice -- precipitation totals are right-skewed and bounded at zero,
     which a Gamma fits far better than a Normal).
  3. Transform each value through the fitted Gamma CDF to a percentile, then
     through the standard Normal's inverse CDF (ppf) to get the SPI value
     itself: mean 0, std 1, by construction.
  4. Classify using the standard WMO thresholds.

Zero-precipitation months are real in Afghanistan's dry season, not missing
data -- handled via a mixture model (probability of zero, then Gamma for the
nonzero part), the standard SPI treatment for this, not silently dropped or
replaced with a small constant.
"""
import numpy as np
import pandas as pd
from scipy import stats

SPI_CATEGORIES = [
    (2.0, float("inf"), "extremely_wet"),
    (1.5, 2.0, "very_wet"),
    (1.0, 1.5, "moderately_wet"),
    (-1.0, 1.0, "near_normal"),
    (-1.5, -1.0, "moderate_drought"),
    (-2.0, -1.5, "severe_drought"),
    (float("-inf"), -2.0, "extreme_drought"),
]


def classify(spi):
    if pd.isna(spi):
        return None
    for lo, hi, label in SPI_CATEGORIES:
        if lo <= spi < hi:
            return label
    return None


def fit_and_transform(sums):
    """sums: array of 3-month precip totals for one province x one calendar
    month, across all years. Returns SPI values (same length, order
    preserved) via the zero-inflated Gamma method."""
    sums = np.asarray(sums, dtype=float)
    n = len(sums)
    n_zero = int((sums == 0).sum())
    p_zero = n_zero / n if n > 0 else 0.0

    nonzero = sums[sums > 0]
    spi = np.full(n, np.nan)
    if len(nonzero) < 4:
        # Not enough nonzero observations to fit a distribution honestly --
        # leave as NaN rather than force a fit scipy itself would warn about.
        return spi

    # Gamma fit with location fixed at 0 (standard for SPI -- precip totals
    # start at 0, a free location parameter would let the fit drift off the
    # physically meaningful support).
    shape, loc, scale = stats.gamma.fit(nonzero, floc=0)

    for i, v in enumerate(sums):
        if v <= 0:
            # zero-inflated: P(X <= 0) = p_zero exactly, by definition
            cdf = p_zero
        else:
            cdf = p_zero + (1 - p_zero) * stats.gamma.cdf(v, shape, loc=loc, scale=scale)
        cdf = min(max(cdf, 1e-6), 1 - 1e-6)  # keep ppf finite at the tails
        spi[i] = stats.norm.ppf(cdf)
    return spi


def main():
    df = pd.read_csv("data/raw/precip_chirps.csv")
    df = df.sort_values(["province", "year", "month"]).reset_index(drop=True)

    # 3-month rolling sum ending in the labeled month, per province.
    df["precip_3mo_mm"] = (
        df.groupby("province")["precip_total_mm"]
          .rolling(3, min_periods=3).sum()
          .reset_index(level=0, drop=True)
    )

    df["spi3"] = np.nan
    for province in df["province"].unique():
        for month in range(1, 13):
            mask = (df["province"] == province) & (df["month"] == month) & df["precip_3mo_mm"].notna()
            idx = df.index[mask]
            if len(idx) < 4:
                continue
            spi_vals = fit_and_transform(df.loc[idx, "precip_3mo_mm"].values)
            df.loc[idx, "spi3"] = spi_vals

    df["spi3_category"] = df["spi3"].apply(classify)
    df["is_drought"] = df["spi3"] < -1.0  # moderate-or-worse, standard SPI threshold

    out_path = "data/processed/spi3.csv"
    import os
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(out_path, index=False)

    covered = df["spi3"].notna().sum()
    print(f"wrote {len(df)} rows ({covered} with a computed SPI-3) -> {out_path}")
    print("\nSPI-3 category counts (2001-2023, all provinces):")
    print(df["spi3_category"].value_counts())
    print(f"\ndrought rate (SPI-3 < -1.0): {df['is_drought'].mean():.1%}")


if __name__ == "__main__":
    main()

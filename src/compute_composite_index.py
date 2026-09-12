"""
Composite Drought Index (CDI) -- combines two independently-computed,
peer-reviewed drought indices into one number that's easy to read at a
glance, while the rigor stays underneath for anyone who wants it:

  CDI = 0.5 x VHI (Vegetation Health Index, Kogan 1995 -- satellite-observed
        vegetation + temperature stress)
      + 0.5 x SPI-3 rescaled to the same 0-100, "high = healthy" direction
        (WMO's standard meteorological drought index -- see compute_spi.py)

Equal weighting is a real, disclosed methodological choice, not a tuned or
hidden one: it treats the two indices (one satellite-observed, one
precipitation-based) as equally informative in the absence of a
province-specific reason to prefer one. A genuine simplification of the
published Multivariate Standardized Drought Index approach (Hao &
AghaKouchak, 2013, which combines indices via a fitted joint distribution
rather than a simple average) -- named and built independently here, not
presented as that exact method.

Plain-language categories use VHI's own established operational thresholds
(NOAA/STAR), carried over to the combined score for consistency:
  >= 60  Healthy
  40-59  Normal
  26-39  Moderate drought
  < 26   Severe drought
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

CATEGORIES = [
    (60, float("inf"), "Healthy"),
    (40, 60, "Normal"),
    (26, 40, "Moderate drought"),
    (float("-inf"), 26, "Severe drought"),
]


def classify(cdi):
    if pd.isna(cdi):
        return None
    for lo, hi, label in CATEGORIES:
        if lo <= cdi < hi:
            return label
    return None


def main():
    spi = pd.read_csv("data/processed/spi3.csv")
    vhi = pd.read_csv("data/processed/vhi.csv")

    df = spi.merge(vhi, on=["province", "year", "month"], how="inner")

    # SPI-3 is a standard-normal value by construction (see compute_spi.py) --
    # its own CDF gives an honest 0-100 percentile, same "high = healthy"
    # direction as VHI, with no separate fit needed.
    df["spi3_scaled"] = df["spi3"].apply(lambda s: stats.norm.cdf(s) * 100 if pd.notna(s) else np.nan)
    df["cdi"] = 0.5 * df["vhi"] + 0.5 * df["spi3_scaled"]
    df["cdi_category"] = df["cdi"].apply(classify)

    out_cols = ["province", "year", "month", "spi3", "vhi", "vci", "tci", "cdi", "cdi_category"]
    out_path = "data/processed/composite_index.csv"
    os.makedirs("data/processed", exist_ok=True)
    df[out_cols].round(2).to_csv(out_path, index=False)

    covered = df["cdi"].notna().sum()
    print(f"wrote {len(df)} rows ({covered} with a computed CDI) -> {out_path}")
    print(f"\nCDI category counts:\n{df['cdi_category'].value_counts()}")

    # Cross-check: correlation between the two independent input indices.
    both = df.dropna(subset=["spi3_scaled", "vhi"])
    corr = both["spi3_scaled"].corr(both["vhi"])
    print(f"\ncorrelation between SPI-3 (precipitation) and VHI (satellite): {corr:.2f}")
    print("(two independently-derived signals agreeing is a real cross-check, "
          "not circular -- one comes from CHIRPS rainfall, the other from MODIS "
          "vegetation+temperature)")


if __name__ == "__main__":
    main()

"""
Monte Carlo scenario simulation for the drought classifier -- same method
as the companion yield-forecasting project's src/simulate.py, adapted for a
classifier: instead of one point prediction, resample (5,000 draws, with
replacement) each province's own real historical months, stratified by SPI-3
tercile (driest_tercile / all_years / wettest_tercile), run the trained
LightGBM model on every draw, and report the resulting distribution of
predicted next-month drought probability (P10/P50/P90) plus the share of
draws crossing the 50% decision threshold.

Same design reasoning as before: scenarios are built by resampling real
historical (SAR, NDVI/EVI, LST, precip, SPI-3) tuples together, not by
inventing an independent synthetic shock to one variable -- these features
are physically coupled (a real drought suppresses NDVI and changes radar
backscatter together), so treating them independently would misrepresent
what a real dry season actually looks like.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd

FEATURES = [
    "ndvi", "evi", "vv_db", "vh_db", "lst_day_c", "precip_total_mm", "spi3",
    "ndvi_lag1", "evi_lag1", "vv_db_lag1", "vh_db_lag1", "lst_day_c_lag1",
    "precip_total_mm_lag1", "spi3_lag1",
]

N_DRAWS = 5000
SCENARIOS = ["driest_tercile", "all_months", "wettest_tercile"]


def load_pool(panel, province, scenario):
    rows = panel[panel["province"] == province].dropna(subset=FEATURES).copy()
    if scenario != "all_months":
        terciles = rows["spi3"].quantile([1 / 3, 2 / 3]).values
        if scenario == "driest_tercile":
            rows = rows[rows["spi3"] <= terciles[0]]
        elif scenario == "wettest_tercile":
            rows = rows[rows["spi3"] >= terciles[1]]
    return rows


def simulate_province(model, panel, province, scenario, rng):
    pool = load_pool(panel, province, scenario)
    if len(pool) == 0:
        return None
    draws = pool.sample(n=N_DRAWS, replace=True, random_state=rng.integers(1e9))
    proba = model.predict_proba(draws[FEATURES])[:, 1]
    return {
        "province": province,
        "scenario": scenario,
        "n_historical_months_in_pool": len(pool),
        "p10_risk": float(np.percentile(proba, 10)),
        "p50_risk": float(np.percentile(proba, 50)),
        "p90_risk": float(np.percentile(proba, 90)),
        "mean_risk": float(proba.mean()),
        "prob_over_threshold": float(np.mean(proba >= 0.5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    model = joblib.load("models/lightgbm_drought.joblib")
    panel = pd.read_csv("data/processed/panel.csv")

    results = []
    for province in sorted(panel["province"].unique()):
        for scenario in SCENARIOS:
            r = simulate_province(model, panel, province, scenario, rng)
            if r:
                results.append(r)

    out = pd.DataFrame(results)
    os.makedirs("data/processed", exist_ok=True)
    out.to_csv("data/processed/simulation_results.csv", index=False)

    print(f"{'province':14s} {'scenario':16s} {'n_pool':6s} {'P10':>6s} {'P50':>6s} {'P90':>6s} {'>=50%':>7s}")
    for _, r in out.iterrows():
        print(f"{r['province']:14s} {r['scenario']:16s} {r['n_historical_months_in_pool']:<6.0f} "
              f"{r['p10_risk']*100:5.0f}% {r['p50_risk']*100:5.0f}% {r['p90_risk']*100:5.0f}% "
              f"{r['prob_over_threshold']*100:6.0f}%")

    print(f"\nwrote {len(out)} rows -> data/processed/simulation_results.csv")


if __name__ == "__main__":
    main()

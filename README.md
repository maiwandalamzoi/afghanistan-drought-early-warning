# Afghanistan Drought Early-Warning Classifier

Predicts whether each of Afghanistan's 34 provinces will be in
meteorological drought **next month**, using this month's satellite radar,
vegetation, and temperature signals. A companion project to
[cereal-yield-satellite-forecast](https://github.com/maiwandalamzoi/cereal-yield-satellite-forecast)
— that one forecasts *how much* wheat a country will yield at harvest; this
one asks a different question, on a different timescale: *which province is
about to enter drought*, weeks before it does.

**What this is:** a real, trained classifier (LightGBM / Logistic
Regression) evaluated on a genuine one-month-ahead, time-based holdout,
benchmarked against a naive persistence baseline, with honestly reported
precision/recall — not just accuracy, which is misleading for an imbalanced
minority-class problem like drought detection.
**What this is not:** an operational early-warning system. See *Known
limitations* below.

## Method

1. **Label — SPI-3 (Standardized Precipitation Index)**, the actual WMO
   definition of meteorological drought, computed from real CHIRPS
   precipitation (2001–2023, `src/extract_precip.py`) via a proper
   zero-inflated Gamma distribution fit per province per calendar month
   (`src/compute_spi.py`) — not a proxy label. Sanity check: the computed
   drought rate (SPI-3 < -1.0) is 15.9% across the full record, almost
   exactly the ~15.87% a correctly-calibrated SPI implies by construction.
2. **Satellite radar** — Sentinel-1 SAR (VV+VH backscatter), monthly,
   2015–2023 (`src/extract_sar.py`). Radar penetrates cloud cover and
   responds to soil moisture more directly than optical vegetation
   indices — the feature this project has that the yield-forecasting
   companion project doesn't.
3. **Vegetation + temperature** — MODIS NDVI/EVI and land-surface
   temperature, monthly, 2015–2023 (`src/extract_satellite.py`,
   `src/extract_climate.py`).
4. **Geography** — all 34 Afghan provinces (FAO GAUL administrative
   boundaries), each source area-averaged over the real province polygon
   via Earth Engine's `reduceRegions()` — every province in one server-side
   call per month, not 34 separate calls.
5. **Panel + target** — merged into a province×month panel with 1-month
   lag features, and the actual prediction target: **next** month's drought
   status, from **this** month's conditions (`src/build_dataset.py`) — a
   real time gap between features and label, not same-month leakage.
6. **Train/evaluate** — time-based split (train 2015–2021, test 2022–2023,
   no random k-fold, same reasoning as the companion project: a random
   split would leak future conditions into training).
   `src/train.py`
7. **Composite Drought Index (CDI)** — a separate, human-readable 0–100
   score for the dashboard, built from two real published indices (VHI and
   SPI-3), not fed into the model above. See *Composite Drought Index*
   below. `src/compute_vhi.py`, `src/compute_composite_index.py`

## Results

Train: 2,065 rows (2015–2021), 11.2% drought-next.
Test: 782 rows (2022–2023), **30.8% drought-next** — a genuinely harder,
more drought-heavy period than the model trained on, not a cherry-picked
easy holdout. (2022 was Afghanistan's most severe drought year in this
record — see *Real-world check* below.)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| baseline (persistence) | 0.754 | 0.606 | 0.581 | 0.593 | — |
| logistic regression | 0.646 | 0.455 | **0.747** | 0.565 | 0.755 |
| random forest | 0.698 | 0.511 | 0.469 | 0.489 | 0.697 |
| xgboost | 0.721 | 0.555 | 0.485 | 0.518 | 0.724 |
| **lightgbm** | 0.752 | 0.598 | 0.598 | **0.598** | **0.766** |

**Honest read:** this is a hard problem, and the naive "next month = this
month" baseline is tough to beat — only LightGBM matches it on F1. Random
Forest and XGBoost actually do *worse* than just guessing persistence. The
one model that offers a real, different trade-off is logistic regression:
much lower precision, but recall of 0.747 vs. the baseline's 0.581 — it
catches noticeably more real droughts at the cost of more false alarms.
Whether that trade-off is worth it depends entirely on what a false alarm
costs vs. a missed drought in an actual early-warning deployment; this
report picks neither model as "the" winner, because that's a policy
decision, not a modeling one.

Feature importance (XGBoost):

| Feature | Importance |
|---|---|
| `spi3` (current month) | 0.224 |
| `precip_total_mm` | 0.103 |
| `precip_total_mm_lag1` | 0.087 |
| `lst_day_c` | 0.082 |
| `lst_day_c_lag1` | 0.068 |
| `ndvi_lag1` | 0.066 |
| `spi3_lag1` | 0.058 |
| `evi` | 0.055 |
| `vh_db` (SAR) | 0.048 |
| `vv_db` (SAR) | 0.047 |
| `evi_lag1` | 0.047 |
| `vv_db_lag1` | 0.046 |
| `vh_db_lag1` | 0.042 |
| `ndvi` | 0.028 |

Current drought state and precipitation dominate (as expected — drought is
autocorrelated month to month), but SAR contributes a real, non-trivial
~18% combined — the radar signal is doing genuine work, not just riding
along.

Reproduce with `python src/train.py`.

**Tested and deliberately not used as model features:** VCI/TCI/VHI (see
below) are deterministic functions of the same NDVI/LST columns already in
`FEATURES`. Adding them as extra model inputs was tried honestly, not
assumed to help — measured before/after on this exact test split:

| Model | F1 without VHI features | F1 with VHI features added |
|---|---|---|
| LightGBM | **0.598** | 0.482 |
| Random Forest | **0.489** | 0.419 |

Both got worse — most likely overfitting: near-duplicate, redundant
dimensionality against only ~2,065 training rows, not new information. The
leaner 14-feature set (`src/train.py`) is what's actually trained and
reported above. VHI is still computed and used, just for a different job —
see the next section.

## Composite Drought Index (CDI) — a human-readable score for the dashboard

The model above predicts one thing: the probability of drought *next*
month. It doesn't produce a simple "how bad is it right now" number for a
province, and the raw SPI-3 value it's trained on (a standard-normal
z-score, roughly -3 to +3) isn't intuitive to a non-technical reader. The
[live dashboard](https://maiwandalamzoi.github.io/afghanistan-drought-early-warning/)
needed a real, defensible score instead of an invented one, so it's built
from two independently-published methods, not from scratch:

1. **VHI (Vegetation Health Index)** — Kogan, F.N. (1995), *"Application of
   vegetation index and brightness temperature for drought detection,"*
   Advances in Space Research 15(11). Operational at NOAA/STAR and used in
   FAO's Agricultural Stress Index System. `src/compute_vhi.py`:
   - VCI = 100 × (NDVI − NDVI_min) / (NDVI_max − NDVI_min)
   - TCI = 100 × (LST_max − LST) / (LST_max − LST_min)
   - VHI = 0.5 × VCI + 0.5 × TCI (Kogan's standard weighting)
   - min/max computed per province, per calendar month, across the full
     2015–2023 record — each province judged against its own history for
     that time of year, the standard VHI convention.
   - Real 3,672-row distribution: mean 49.6, median 50.0. 30.8% of
     province-months are at or below NOAA/STAR's stress threshold (≤40);
     15.9% at or below the severe threshold (≤26).
2. **SPI-3**, rescaled 0–100 via its own normal CDF (it's already a
   standard-normal value by construction — no separate fit needed), so it
   points the same "higher = healthier" direction as VHI.
3. **CDI = 0.5 × VHI + 0.5 × SPI-3(rescaled)** (`src/compute_composite_index.py`).
   Equal weighting is a disclosed choice — it treats a satellite-vegetation
   signal and a rainfall signal as equally informative absent a
   province-specific reason to prefer one — not a tuned or hidden
   parameter. A genuine simplification of the published Multivariate
   Standardized Drought Index approach (Hao & AghaKouchak, 2013, which
   fits a joint distribution rather than averaging) — named and built
   independently here, not presented as that exact method.

Categories carry over VHI's own NOAA/STAR thresholds: **≥60 Healthy, 40–59
Normal, 26–39 Moderate drought, <26 Severe drought.** Across the full
3,672-row record: 1,362 Healthy, 1,218 Normal, 661 Moderate drought, 431
Severe drought.

**Cross-check:** VHI and SPI-3 come from genuinely independent sources —
one from MODIS vegetation greenness + land-surface temperature, the other
from CHIRPS rainfall — and still correlate at **r = 0.27**. Two independent
signals agreeing at all is a real, if modest, sanity check; a much higher
correlation would actually be suspicious (it would suggest they aren't
really independent). CDI is a display index, not a model input — see
*Tested and deliberately not used as model features* above for why it's
kept out of `train.py`.

## Real-world check: does this line up with actual events?

**2022 spring drought.** National-average SPI-3 (mean across all 34
provinces) for April–June 2022: **-1.61, -1.18, -1.25** — moderate-to-severe
drought, matching widely reported conditions from that period.

**FEWS NET cross-check (not the training label — see below for why).**
Across 23 months where FEWS NET separately classified Afghanistan at IPC
Crisis level (phase 3) or worse, mean national SPI-3 was **-0.18**, vs.
**+0.02** across the full record — a real but modest shift, not a strong
correlation. That's expected, not a weak result: Afghanistan's food-security
crisis since 2021 has been driven mostly by conflict and economic collapse,
not meteorological drought alone, which is exactly why this project used
SPI-3 (an actual, disclosed, single-cause meteorological definition) as the
label instead of IPC phase (which mixes drought with everything else). A
strong correlation here would have been suspicious, not reassuring.

## Known limitations

Read this before citing the results anywhere:

- **Test period is unusually severe.** 2022–2023's 30.8% drought-next rate
  is roughly triple the training period's 11.2% — real distributional
  shift, not an easy holdout, but it also means these metrics may not
  represent performance in a more typical (less drought-heavy) year.
- **SAR/NDVI/LST only go back to 2015** (Sentinel-1's real coverage start
  for this region — verified, not assumed: 0 images in 2014, 55–184/month
  from January 2015). Only ~9 years of monthly data, 3,604 rows after
  dropping the first/last month per province — modest for 14 features.
  Precipitation alone goes back to 2001 for the SPI climatology fit.
- **11 of 108 months have no Sentinel-1 dual-polarization coverage at all**
  (mostly in early 2015, right at the edge of the operational archive) —
  written as real nulls, dropped at training time, not imputed or hidden.
- **LST is land-surface temperature, not air temperature** — daytime values
  in bare, sun-heated terrain can exceed 45°C, which is correct for LST,
  not a data error, but not what a thermometer would read.
- **A single dominant label source (CHIRPS) drives both the target and a
  feature** (`precip_total_mm` informs both SPI-3 and is itself a model
  input) — some of the model's apparent skill is definitionally close to
  the label, not independent signal, though the lag structure and the SAR/
  NDVI/LST features are genuinely independent of precipitation.
- **Province-level area averaging** blends non-agricultural land (bare
  mountains, urban areas) into every feature, same caveat as the companion
  project's country-level averaging, just one administrative level down.
- **No model here should inform an actual response decision.** This is a
  research artifact evaluating whether the signal exists, not a validated
  early-warning product.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # only needed to re-extract satellite/SAR/climate data
```

`requirements.txt` is pinned to the exact versions used to train the
committed `models/*.joblib` — the same reproducibility discipline as the
companion project, verified the same way (fresh venv, fresh clone).

**You do not need a GEE account or any API key to use the trained models.**
`models/*.joblib` and `data/processed/panel.csv` are committed —
`python src/train.py` runs immediately after `pip install`. A GEE service
account is only needed to re-run the `extract_*.py` scripts and pull fresh
satellite/SAR/climate data.

## Reproduce

```bash
python src/extract_precip.py       # -> data/raw/precip_chirps.csv (2001-2023, CHIRPS)
python src/compute_spi.py          # -> data/processed/spi3.csv (SPI-3 + drought label)
python src/extract_satellite.py    # -> data/raw/satellite_modis.csv (NDVI/EVI, 2015-2023)
python src/extract_climate.py      # -> data/raw/lst_modis.csv (LST, 2015-2023)
python src/extract_sar.py          # -> data/raw/sar_sentinel1.csv (Sentinel-1, 2015-2023, resumable)
python src/compute_vhi.py          # -> data/processed/vhi.csv (VCI/TCI/VHI, Kogan 1995)
python src/compute_composite_index.py  # -> data/processed/composite_index.csv (CDI, dashboard-only)
python src/build_dataset.py        # -> data/processed/panel.csv
python src/train.py                # -> models/*.joblib, prints metrics
python src/fetch_fews_net.py       # -> data/raw/fews_net_ipc_national.csv (validation only, not a training input)
```

All four `extract_*.py` scripts use `reduceRegions()` to pull all 34
provinces in one Earth Engine call per month (not 34 separate calls) and
write incrementally with resume support — a stopped or interrupted run
picks up from whichever province-months are already on disk. `extract_sar.py`
is the slowest (~50-60s/month, compositing 100+ Sentinel-1 scenes across the
whole country) and also handles the one real bug found while building
this: an empty dual-polarization collection produces a bandless composite
image, which crashes `reduceRegions()` outright rather than returning nulls
— checked for and handled explicitly, not caught-and-ignored.

## License

MIT

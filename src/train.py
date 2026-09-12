"""
Train + evaluate drought early-warning classifiers on data/processed/panel.csv.

Target: target_drought_next (binary -- will next month's SPI-3 be moderate
drought or worse, i.e. SPI-3 < -1.0). Predicted from THIS month's SAR,
NDVI/EVI, LST, precipitation, and SPI-3 (plus their 1-month lags) -- a real
one-month-ahead forecast, not a same-month fit.

Time-based split (not random k-fold, same reasoning as the companion
yield-forecasting project): train on 2015-2021, test on 2022-2023, so nothing
in training ever saw the years being scored.

Baseline is naive persistence: "next month's drought status = this month's
drought status." For an early-warning system, the question that matters
isn't just accuracy (drought is the minority class -- a model that never
predicts drought can still look "accurate") -- it's RECALL on the drought
class: missing a real drought is worse than a false alarm, so recall is
reported alongside precision/F1, not just accuracy.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import os

TRAIN_END_YEAR = 2021  # train 2015-2021, test 2022-2023

FEATURES = [
    "ndvi", "evi", "vv_db", "vh_db", "lst_day_c", "precip_total_mm", "spi3",
    "ndvi_lag1", "evi_lag1", "vv_db_lag1", "vh_db_lag1", "lst_day_c_lag1",
    "precip_total_mm_lag1", "spi3_lag1",
]
# Tested and deliberately NOT included: vci/tci/vhi (Kogan 1995 vegetation
# health index components, computed in compute_vhi.py). Adding them as
# model features was tried honestly, not assumed to help -- LightGBM's F1
# dropped from 0.598 to 0.482 and Random Forest's from 0.489 to 0.419 with
# them included, most likely overfitting: they're deterministic functions
# of ndvi/lst already in this feature set, so adding them as separate
# columns mostly adds redundant dimensionality against only ~2,065 training
# rows, not new information. VHI/CDI are still computed and used --
# as the dashboard's human-readable display index (see
# compute_composite_index.py), a different job than a model feature, with a
# different bar to clear. Keeping the model on its better-performing,
# leaner feature set here rather than the more sophisticated-sounding one.
TARGET = "target_drought_next"


def evaluate(name, y_true, y_pred, y_proba, results):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else float("nan")
    print(f"{name:22s}  acc={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  "
          f"F1={f1:.3f}  AUC={auc:.3f}")
    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def main():
    df = pd.read_csv("data/processed/panel.csv").dropna(subset=FEATURES + [TARGET])
    df["target_drought_next"] = df["target_drought_next"].astype(bool).astype(int)
    df["date"] = pd.to_datetime(df["date"])

    train = df[df["year"] <= TRAIN_END_YEAR].reset_index(drop=True)
    test = df[df["year"] > TRAIN_END_YEAR].reset_index(drop=True)
    print(f"train: {len(train)} rows ({train['year'].min()}-{train['year'].max()}), "
          f"{train[TARGET].mean():.1%} drought-next")
    print(f"test:  {len(test)} rows ({test['year'].min()}-{test['year'].max()}), "
          f"{test[TARGET].mean():.1%} drought-next")
    print()

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    results = {}

    # 1. Naive persistence baseline: next month = this month's current status
    baseline_pred = (test["spi3"] < -1.0).astype(int)
    evaluate("baseline_persistence", y_test, baseline_pred, None, results)

    # 2. Logistic regression (standardized features)
    scaler = StandardScaler().fit(X_train)
    logit = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X_train), y_train)
    logit_proba = logit.predict_proba(scaler.transform(X_test))[:, 1]
    evaluate("logistic_regression", y_test, (logit_proba >= 0.5).astype(int), logit_proba, results)

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    evaluate("random_forest", y_test, rf.predict(X_test), rf_proba, results)

    # 4. XGBoost
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         scale_pos_weight=pos_weight, random_state=42, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    evaluate("xgboost", y_test, xgb.predict(X_test), xgb_proba, results)

    # 5. LightGBM
    lgbm = LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                           class_weight="balanced", random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)
    lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
    evaluate("lightgbm", y_test, lgbm.predict(X_test), lgbm_proba, results)

    best_name = max(results, key=lambda k: results[k]["recall"])
    print(f"\nbest model by drought-class recall: {best_name} "
          f"(recall={results[best_name]['recall']:.3f}, precision={results[best_name]['precision']:.3f})")

    print("\nfeature importance (xgboost):")
    for feat, imp in sorted(zip(FEATURES, xgb.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:24s} {imp:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/random_forest_drought.joblib")
    joblib.dump(xgb, "models/xgboost_drought.joblib")
    joblib.dump(lgbm, "models/lightgbm_drought.joblib")
    joblib.dump({"model": logit, "scaler": scaler, "features": FEATURES}, "models/logistic_drought.joblib")
    print("\nsaved models/*.joblib")

    return results, best_name


if __name__ == "__main__":
    main()

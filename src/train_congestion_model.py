"""
train_congestion_model.py — Step 5A: Build ML dataset and train congestion model.

⚠ WEAK SUPERVISION — CLEARLY LABELLED
───────────────────────────────────────
No ground-truth congestion labels exist for individual road edges.
The target variable (congestion_score) is a PSEUDO-LABEL constructed from
a weighted combination of real feature scores plus synthetic temporal factors.

REAL FEATURES (from previous pipeline steps):
  • accident_score    — real BTP crash data (station-level)
  • violation_score   — real violation counts (city-level proxy)
  • traffic_density   — real area-level traffic data (area-level proxy)
  • enforcement_score — real enforcement data (city-level proxy)
  • centrality_score  — graph topology (real OSM structure)
  • road_type         — OSM highway classification (real)
  • road_length       — OSM geometry (real)
  • intersection_delay — SYNTHETIC (estimated from topology, Step 4)

SYNTHETIC FEATURES (clearly labelled):
  • hour           — randomly sampled from realistic Bangalore rush-hour distribution
                     ⚠ SYNTHETIC: No real time-series edge data available
  • weather_factor — sampled from Bangalore seasonal weather distribution
                     ⚠ SYNTHETIC: No real weather-to-congestion mapping available

PSEUDO-LABEL FORMULA (weak supervision):
  congestion_score = (
      0.35 × traffic_density +
      0.25 × accident_score +
      0.15 × violation_score +
      0.15 × (intersection_delay / max_delay) +
      0.10 × weather_factor_normalized
  ) × rush_hour_multiplier

  rush_hour_multiplier:
    08:00–10:00 → 1.5  (morning peak)
    17:00–20:00 → 1.6  (evening peak)
    otherwise   → 1.0

MODELS TRAINED:
  1. XGBoost Regressor (primary)
  2. Random Forest Regressor (comparison)
  3. Linear Regression (baseline)

EVALUATION:
  • MAE, RMSE, R² on 20% held-out test set
  • Feature importance (XGBoost + RF)
  • Residual analysis

OUTPUT:
  • output/models/congestion_model.pkl  (best model)
  • output/ml_dataset.csv
  • output/reports/model_evaluation.txt
"""

from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import osmnx as ox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import (
    GRAPH_SIGNAL_PATH, MODEL_PATH, ML_DATASET_PATH,
    MODELS_DIR, REPORTS_DIR, OUTPUT_DIR,
    RANDOM_STATE, TEST_SIZE, XGB_PARAMS,
)
from utils import ensure_output_dirs, safe_float


# ─────────────────────────────────────────────────────────────────────────────
# LOAD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def load_graph():
    print("[Step 5A] Loading signalized graph ...")
    G = ox.load_graphml(str(GRAPH_SIGNAL_PATH))
    print(f"  → {G.number_of_edges():,} edges")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# BUILD ML DATASET
# ─────────────────────────────────────────────────────────────────────────────

# Bangalore hourly traffic distribution (approximate, based on urban studies)
# Higher probability during rush hours
_HOUR_WEIGHTS = np.array([
    0.5, 0.3, 0.2, 0.2, 0.3, 0.6,   # 00–05
    1.2, 1.8, 2.5, 2.0, 1.5, 1.4,   # 06–11
    1.6, 1.5, 1.4, 1.5, 1.8, 2.8,   # 12–17
    2.6, 2.0, 1.5, 1.2, 0.9, 0.7,   # 18–23
])
_HOUR_WEIGHTS /= _HOUR_WEIGHTS.sum()

# Bangalore weather factor distribution (dry season heavier traffic)
_WEATHER_MEAN = 1.1
_WEATHER_STD  = 0.2


def build_ml_dataset(G) -> pd.DataFrame:
    """
    Extract features from every edge and generate pseudo-labels.

    ⚠ WEAK SUPERVISION: The target 'congestion_score' is a formula-derived
    pseudo-label, not a measured ground truth.
    """
    print("  Building ML dataset (weak supervision) ...")
    rng = np.random.default_rng(RANDOM_STATE)

    rows = []
    max_delay = max(
        (safe_float(d.get("intersection_delay", 0.0))
         for _, _, _, d in G.edges(keys=True, data=True)),
        default=1.0,
    )
    max_delay = max(max_delay, 1.0)

    for u, v, k, data in G.edges(keys=True, data=True):
        length    = safe_float(data.get("road_length", data.get("length", 1.0)), 1.0)
        acc       = safe_float(data.get("accident_score", 0.0))
        viol      = safe_float(data.get("violation_score", 0.0))
        traf      = safe_float(data.get("traffic_density", 0.0))
        delay     = safe_float(data.get("intersection_delay", 0.0))
        comp      = safe_float(data.get("compliance_score", 0.0))
        cent      = safe_float(data.get("centrality_score", 0.0))
        road_type = safe_float(data.get("road_type", 1.0))
        base_w    = safe_float(data.get("weight", 1.0), 1.0)

        # ── SYNTHETIC temporal & weather features ─────────────────────────
        hour           = int(rng.choice(24, p=_HOUR_WEIGHTS))
        weather_factor = float(np.clip(rng.normal(_WEATHER_MEAN, _WEATHER_STD), 0.5, 2.0))

        # ── PSEUDO-LABEL (weak supervision) ──────────────────────────────
        delay_norm = delay / max_delay
        weather_norm = (weather_factor - 0.5) / 1.5   # normalize to [0,1]

        congestion = (
            0.35 * traf +
            0.25 * acc +
            0.15 * viol +
            0.15 * delay_norm +
            0.10 * weather_norm
        )

        # Rush-hour multiplier (SYNTHETIC temporal assumption)
        if 8 <= hour <= 10:
            congestion *= 1.5
        elif 17 <= hour <= 20:
            congestion *= 1.6

        # Clip to [0, 1]
        congestion = float(np.clip(congestion, 0.0, 1.0))

        rows.append({
            # Real features
            "road_length"       : length,
            "accident_score"    : acc,
            "violation_score"   : viol,
            "traffic_density"   : traf,
            "enforcement_score" : comp,
            "centrality_score"  : cent,
            "road_type"         : road_type,
            "intersection_delay": delay,
            "base_weight"       : base_w,
            # Synthetic features (labelled)
            "hour_synthetic"    : hour,
            "weather_synthetic" : weather_factor,
            # Pseudo-label target
            "congestion_score"  : congestion,
        })

    df = pd.DataFrame(rows)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Congestion score — mean: {df['congestion_score'].mean():.4f}  "
          f"std: {df['congestion_score'].std():.4f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "road_length", "accident_score", "violation_score", "traffic_density",
    "enforcement_score", "centrality_score", "road_type",
    "intersection_delay", "base_weight",
    "hour_synthetic", "weather_synthetic",
]
TARGET_COL = "congestion_score"


def train_models(df: pd.DataFrame) -> tuple:
    """
    Train XGBoost, Random Forest, and Linear Regression models.
    Returns (best_model, results_dict, X_test, y_test).
    """
    print("\n  Training models ...")
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Scaler for Linear Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "XGBoost": XGBRegressor(**XGB_PARAMS, verbosity=0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LinearRegression": LinearRegression(),
    }

    results = {}
    best_model = None
    best_mae   = float("inf")

    for name, model in models.items():
        if name == "LinearRegression":
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": model}
        print(f"    {name:20s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

        if mae < best_mae:
            best_mae   = mae
            best_model = model

    print(f"\n  Best model: {min(results, key=lambda k: results[k]['MAE'])}")
    return best_model, results, X_test, y_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, output_dir):
    """Save a feature importance bar chart."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        idx = np.argsort(importances)[::-1]
        ax.bar(range(len(importances)), importances[idx], color="#2563EB")
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
        ax.set_title("Feature Importance — Congestion Prediction Model\n"
                     "(⚠ Target is a pseudo-label — weak supervision)")
        ax.set_ylabel("Importance Score")
        plt.tight_layout()
        path = output_dir / "feature_importance.png"
        plt.savefig(str(path), dpi=150)
        plt.close()
        print(f"  Feature importance plot → {path}")
    except Exception as e:
        print(f"  Could not plot feature importance: {e}")


def plot_residuals(model, X_test, y_test, output_dir):
    """Save a residual scatter plot."""
    try:
        preds = model.predict(X_test)
        residuals = y_test - preds

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(preds, residuals, alpha=0.3, s=5, color="#2563EB")
        axes[0].axhline(0, color="red", linewidth=1)
        axes[0].set_xlabel("Predicted Congestion Score")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Predicted")

        axes[1].hist(residuals, bins=50, color="#2563EB", edgecolor="white")
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Residual Distribution")

        plt.suptitle("Model Residual Analysis\n(⚠ Pseudo-label target — weak supervision)",
                     fontsize=11)
        plt.tight_layout()
        path = output_dir / "residuals.png"
        plt.savefig(str(path), dpi=150)
        plt.close()
        print(f"  Residual plot → {path}")
    except Exception as e:
        print(f"  Could not plot residuals: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model):
    ensure_output_dirs(MODELS_DIR)
    joblib.dump(model, str(MODEL_PATH))
    print(f"  Model saved → {MODEL_PATH}")


def save_dataset(df: pd.DataFrame):
    df.to_csv(str(ML_DATASET_PATH), index=False)
    print(f"  ML dataset saved → {ML_DATASET_PATH}")


def save_evaluation_report(results: dict, output_dir):
    ensure_output_dirs(output_dir)
    lines = [
        "=" * 60,
        "CONGESTION MODEL EVALUATION REPORT",
        "=" * 60,
        "",
        "⚠ WEAK SUPERVISION DISCLAIMER",
        "The target variable 'congestion_score' is a PSEUDO-LABEL",
        "derived from a weighted formula of real feature scores plus",
        "synthetic temporal/weather factors. It is NOT a measured",
        "ground-truth congestion value.",
        "",
        "FEATURE COLUMNS:",
    ]
    for f in FEATURE_COLS:
        synthetic = "(SYNTHETIC)" if "synthetic" in f else "(real/proxy)"
        lines.append(f"  {f:30s} {synthetic}")
    lines += ["", "MODEL PERFORMANCE (20% held-out test set):", ""]

    for name, res in results.items():
        lines.append(f"  {name}")
        lines.append(f"    MAE  : {res['MAE']:.6f}")
        lines.append(f"    RMSE : {res['RMSE']:.6f}")
        lines.append(f"    R²   : {res['R2']:.6f}")
        lines.append("")

    lines += [
        "LIMITATIONS:",
        "  1. No real congestion ground truth — pseudo-labels only.",
        "  2. Traffic density is area-level, not edge-level.",
        "  3. Signal delays are synthetic topology estimates.",
        "  4. Temporal features (hour, weather) are randomly sampled.",
        "  5. Model generalizes to Bangalore topology only.",
        "",
        "FUTURE WORK:",
        "  • Integrate real-time GPS probe data for ground truth.",
        "  • Use actual signal timing data from BTP.",
        "  • Add weather API integration for real weather features.",
        "=" * 60,
    ]

    report_path = output_dir / "model_evaluation.txt"
    report_path.write_text("\n".join(lines))
    print(f"  Evaluation report → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs(MODELS_DIR, REPORTS_DIR)

    G  = load_graph()
    df = build_ml_dataset(G)
    save_dataset(df)

    best_model, results, X_test, y_test, scaler = train_models(df)
    save_model(best_model)
    save_evaluation_report(results, REPORTS_DIR)

    plot_feature_importance(best_model, FEATURE_COLS, REPORTS_DIR)
    plot_residuals(best_model, X_test, y_test, REPORTS_DIR)

    print("\n✓ Step 5A complete — congestion model trained.")


if __name__ == "__main__":
    main()

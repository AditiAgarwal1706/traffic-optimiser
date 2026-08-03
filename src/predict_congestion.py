"""
predict_congestion.py — Step 5B: Apply trained model to predict edge congestion.

DATA TRANSPARENCY:
  • Loads the trained XGBoost model (weak supervision — pseudo-label trained).
  • Predicts at peak-hour conditions (hour=9, weather=1.2) as a representative
    morning rush-hour scenario.
  • Adds 'ml_congestion_score' to each edge.
  • Updates 'weight' to reflect ML-predicted congestion.

⚠ NOTE: Predictions inherit all limitations of the pseudo-label training.
  They represent relative congestion ranking, not absolute measurements.

OUTPUT: bangalore_ml.graphml
"""

from __future__ import annotations

import joblib
import numpy as np
import osmnx as ox
import pandas as pd

from config import (
    GRAPH_SIGNAL_PATH, GRAPH_ML_PATH, MODEL_PATH,
    GRAPHS_DIR,
)
from utils import ensure_output_dirs, safe_float


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_graph():
    print("[Step 5B] Loading signalized graph ...")
    G = ox.load_graphml(str(GRAPH_SIGNAL_PATH))
    print(f"  → {G.number_of_edges():,} edges")
    return G


def load_model():
    model = joblib.load(str(MODEL_PATH))
    print(f"  Model loaded: {type(model).__name__}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────

# Feature order must match train_congestion_model.py FEATURE_COLS
FEATURE_COLS = [
    "road_length", "accident_score", "violation_score", "traffic_density",
    "enforcement_score", "centrality_score", "road_type",
    "intersection_delay", "base_weight",
    "hour_synthetic", "weather_synthetic",
]

# Representative prediction scenario: morning peak
PRED_HOUR    = 9     # 09:00 — morning rush
PRED_WEATHER = 1.2   # slightly above average (light cloud / humidity)


def predict_congestion(G, model) -> object:
    """
    Predict congestion score for every edge and update graph attributes.

    Adds:
      ml_congestion_score — predicted congestion [0, 1] (pseudo-label scale)
      weight              — updated: original_weight + ml_congestion_score × 1000
    """
    print(f"  Predicting congestion (hour={PRED_HOUR}, weather={PRED_WEATHER}) ...")

    rows = []
    edge_keys = []

    for u, v, k, data in G.edges(keys=True, data=True):
        rows.append({
            "road_length"       : safe_float(data.get("road_length", data.get("length", 1.0)), 1.0),
            "accident_score"    : safe_float(data.get("accident_score", 0.0)),
            "violation_score"   : safe_float(data.get("violation_score", 0.0)),
            "traffic_density"   : safe_float(data.get("traffic_density", 0.0)),
            "enforcement_score" : safe_float(data.get("enforcement_score", data.get("compliance_score", 0.0))),
            "centrality_score"  : safe_float(data.get("centrality_score", 0.0)),
            "road_type"         : safe_float(data.get("road_type", 1.0)),
            "intersection_delay": safe_float(data.get("intersection_delay", 0.0)),
            "base_weight"       : safe_float(data.get("weight", 1.0), 1.0),
            "hour_synthetic"    : PRED_HOUR,
            "weather_synthetic" : PRED_WEATHER,
        })
        edge_keys.append((u, v, k))

    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    predictions = model.predict(df.values)
    predictions = np.clip(predictions, 0.0, 1.0)

    for (u, v, k), pred in zip(edge_keys, predictions):
        data = G[u][v][k]
        data["ml_congestion_score"] = float(pred)
        # Scale prediction into weight space (×1000 to be comparable with length-based weights)
        data["weight"] = safe_float(data.get("weight", 1.0), 1.0) + float(pred) * 1000.0

    scores = predictions
    print(f"  Congestion score — mean: {scores.mean():.4f}  "
          f"max: {scores.max():.4f}  min: {scores.min():.4f}")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_graph(G):
    ensure_output_dirs(GRAPHS_DIR)
    ox.save_graphml(G, str(GRAPH_ML_PATH))
    print(f"  ML graph saved → {GRAPH_ML_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    G     = load_graph()
    model = load_model()
    G     = predict_congestion(G, model)
    save_graph(G)
    print("\n✓ Step 5B complete — ML congestion scores applied to graph.")


if __name__ == "__main__":
    main()

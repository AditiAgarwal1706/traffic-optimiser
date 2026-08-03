"""
feature_engineering.py — Step 3: Compute graph edge features and congestion weights.

DATA TRANSPARENCY
─────────────────
REAL DATA (from BTP / Karnataka datasets):
  • accident_score   — normalized crash counts from BTP station-wise CSVs (2018–2025)
  • violation_score  — normalized traffic violation counts by type (city-wide)
  • enforcement_score — normalized enforcement cases (M.V.Act, automation, etc.)
  • compliance_score  — derived from enforcement intensity (higher enforcement → higher compliance proxy)
  • road_type        — from OSM highway tag (real)
  • road_length      — from OSM geometry (real)

ESTIMATED / PROXY DATA (clearly labelled):
  • traffic_density  — derived from Bangalore_traffic_Dataset.csv (area-level, not edge-level)
                       Mapped by area/road name fuzzy matching. Where no match exists,
                       the city-wide mean is used as a fallback.
                       ⚠ NOTE: This is an area-level proxy, not a direct edge measurement.

SYNTHETIC / ASSUMED (clearly labelled):
  • None in this step. All scores come from real datasets.

CONGESTION WEIGHT FORMULA:
  weight = (0.35·traffic + 0.25·accident + 0.20·violation + 0.15·signal − 0.05·compliance) × length
  signal_delay is initialized to 0 here; it is filled in Step 4 (signal_intelligence.py).

OUTPUT: bangalore_weighted.graphml
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd

from config import (
    GRAPH_ENRICHED_PATH, GRAPH_WEIGHTED_PATH,
    ACCIDENTS_2025_PATH, ACCIDENTS_2024_PATH, ACCIDENTS_2023_PATH,
    ACCIDENTS_MULTI_PATH, ACCIDENTS_ZONE_PATH,
    VIOLATIONS_PATH, ENFORCEMENT_PATH, BANGALORE_TRAFFIC_PATH,
    WEIGHT_TRAFFIC, WEIGHT_ACCIDENT, WEIGHT_VIOLATION,
    WEIGHT_SIGNAL, WEIGHT_COMPLIANCE,
    GRAPHS_DIR,
)
from utils import (
    clean_columns, coerce_numeric, normalize_series,
    ensure_output_dirs, road_type_rank, safe_float,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def load_graph():
    path = GRAPH_ENRICHED_PATH if GRAPH_ENRICHED_PATH.exists() else GRAPH_WEIGHTED_PATH
    # Fall back to base graph if enriched not yet built
    from config import GRAPH_PATH
    if not path.exists():
        path = GRAPH_PATH
    print(f"[Step 3] Loading graph from {path.name} ...")
    G = ox.load_graphml(str(path))
    print(f"  → {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# ACCIDENT SCORE  (REAL DATA)
# ─────────────────────────────────────────────────────────────────────────────

def build_accident_lookup() -> dict[str, float]:
    """
    Merge all available BTP station-wise accident CSVs (2018–2025).
    Returns {station_name_lower: normalized_accident_score}.
    """
    frames = []
    for path in [
        ACCIDENTS_2025_PATH, ACCIDENTS_2024_PATH, ACCIDENTS_2023_PATH,
        ACCIDENTS_MULTI_PATH, ACCIDENTS_ZONE_PATH,
    ]:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            df = clean_columns(df)
        except Exception:
            continue

        # Find station column
        station_col = next(
            (c for c in df.columns if "station" in c.lower()), None
        )
        if station_col is None:
            continue

        # Find the most recent total-crash column
        total_cols = [c for c in df.columns if "total" in c.lower() and "crash" in c.lower()]
        if not total_cols:
            total_cols = [c for c in df.columns if "total" in c.lower()]
        if not total_cols:
            continue

        def _year_key(col: str) -> int:
            m = re.search(r"(\d{4})", col)
            return int(m.group(1)) if m else 0

        best_col = max(total_cols, key=_year_key)
        sub = df[[station_col, best_col]].copy()
        sub.columns = ["station", "total"]
        sub["total"] = coerce_numeric(sub["total"]).fillna(0)
        frames.append(sub)

    if not frames:
        print("  ⚠ No accident data loaded — accident_score will be 0.")
        return {}

    combined = pd.concat(frames, ignore_index=True)
    # Average across years for the same station
    combined = combined.groupby("station", as_index=False)["total"].mean()
    combined["score"] = normalize_series(combined["total"])
    return {
        str(row["station"]).strip().lower(): float(row["score"])
        for _, row in combined.iterrows()
    }


# ─────────────────────────────────────────────────────────────────────────────
# VIOLATION SCORE  (REAL DATA — city-wide, not station-wise)
# ─────────────────────────────────────────────────────────────────────────────

def build_violation_score() -> float:
    """
    The violations dataset is city-wide (not station-wise), so we compute a
    single normalized city-level score and apply it uniformly.
    Returns a scalar in [0, 1].

    ⚠ LIMITATION: This is a city-wide proxy, not road-specific.
    """
    if not VIOLATIONS_PATH.exists():
        return 0.0
    try:
        df = pd.read_csv(VIOLATIONS_PATH)
        df = clean_columns(df)
    except Exception:
        return 0.0

    year_cols = [c for c in df.columns if str(c).strip().isdigit()]
    if not year_cols:
        return 0.0

    latest = str(max(int(c) for c in year_cols))
    total = coerce_numeric(df[latest]).fillna(0).sum()
    # Normalize against a reference maximum (2025 total violations ~ 2.5M)
    reference_max = 2_500_000
    return float(min(total / reference_max, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# ENFORCEMENT / COMPLIANCE SCORE  (REAL DATA — annual, city-wide)
# ─────────────────────────────────────────────────────────────────────────────

def build_enforcement_score() -> float:
    """
    Annual enforcement data is city-wide.
    Returns a normalized compliance proxy in [0, 1].

    ⚠ LIMITATION: City-wide annual figure applied uniformly to all edges.
    """
    if not ENFORCEMENT_PATH.exists():
        return 0.0
    try:
        df = pd.read_csv(ENFORCEMENT_PATH)
        df = clean_columns(df)
    except Exception:
        return 0.0

    for candidate in ["Total Cases Booked", "Automation cases", "M.V.Act Cases"]:
        if candidate in df.columns:
            val = coerce_numeric(df[candidate]).fillna(0).max()
            # Normalize against reference (2M cases/year)
            return float(min(val / 2_000_000, 1.0))
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TRAFFIC DENSITY  (REAL DATA — area-level proxy)
# ─────────────────────────────────────────────────────────────────────────────

def build_traffic_density_lookup() -> dict[str, float]:
    """
    Build a lookup {area_name_lower: normalized_traffic_density} from the
    Bangalore_traffic_Dataset.csv.

    This dataset has area-level (not edge-level) traffic volume and congestion
    level columns. We average across all dates per area.

    ⚠ PROXY NOTE: Area-level data mapped to edges by name matching.
    """
    if not BANGALORE_TRAFFIC_PATH.exists():
        print("  ⚠ Bangalore traffic dataset not found — traffic_density will use fallback.")
        return {}
    try:
        df = pd.read_csv(BANGALORE_TRAFFIC_PATH)
        df = clean_columns(df)
    except Exception:
        return {}

    area_col = next((c for c in df.columns if "area" in c.lower()), None)
    road_col = next((c for c in df.columns if "road" in c.lower() or "intersection" in c.lower()), None)
    vol_col  = next((c for c in df.columns if "traffic volume" in c.lower() or "volume" in c.lower()), None)
    cong_col = next((c for c in df.columns if "congestion" in c.lower()), None)

    if not (area_col and vol_col):
        return {}

    df[vol_col] = coerce_numeric(df[vol_col]).fillna(0)
    if cong_col:
        df[cong_col] = coerce_numeric(df[cong_col]).fillna(0)

    # Build lookup by area name
    lookup: dict[str, float] = {}
    for area, grp in df.groupby(area_col):
        score = float(grp[vol_col].mean())
        if cong_col:
            # Blend volume and congestion level
            score = 0.6 * score + 0.4 * float(grp[cong_col].mean())
        lookup[str(area).strip().lower()] = score

    # Also build by road/intersection name if available
    if road_col:
        for road, grp in df.groupby(road_col):
            score = float(grp[vol_col].mean())
            if cong_col:
                score = 0.6 * score + 0.4 * float(grp[cong_col].mean())
            lookup[str(road).strip().lower()] = score

    # Normalize all values
    if lookup:
        vals = np.array(list(lookup.values()))
        mn, mx = vals.min(), vals.max()
        if mx - mn > 1e-9:
            lookup = {k: float((v - mn) / (mx - mn)) for k, v in lookup.items()}

    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# MATCH EDGE TO LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def _match_score(
    road_name: str,
    lookup: dict[str, float],
    default: float,
) -> float:
    """
    Try to match a road/area name to the lookup dictionary.
    Falls back to *default* if no match is found.
    """
    if not lookup:
        return default
    name_lower = road_name.strip().lower()
    if name_lower in lookup:
        return lookup[name_lower]
    # Partial match: check if any key is a substring of the road name
    for key, val in lookup.items():
        if key and (key in name_lower or name_lower in key):
            return val
    return default


# ─────────────────────────────────────────────────────────────────────────────
# ASSIGN FEATURES TO GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def assign_features(G, accident_lookup, traffic_lookup, violation_score, enforcement_score):
    """
    Assign all engineered features to every edge in the graph.

    Edge attributes added:
      accident_score    — [0,1] normalized crash intensity (real, station-level)
      violation_score   — [0,1] city-wide violation proxy (real, city-level)
      traffic_density   — [0,1] area-level traffic proxy (real, area-level)
      enforcement_score — [0,1] city-wide enforcement proxy (real, city-level)
      compliance_score  — same as enforcement_score (proxy for compliance)
      road_type         — integer rank of OSM highway type (real)
      road_length       — edge length in metres (real, from OSM)
      centrality_score  — placeholder 0.0 (filled in Step 4)
      signal_delay      — placeholder 0.0 (filled in Step 4)
      weight            — composite congestion weight (formula-based)
    """
    print("  Assigning features to edges ...")

    default_accident    = float(np.mean(list(accident_lookup.values()))) if accident_lookup else 0.0
    default_traffic     = float(np.mean(list(traffic_lookup.values()))) if traffic_lookup else 0.0

    for u, v, k, data in G.edges(keys=True, data=True):
        road_name = str(data.get("name", ""))
        highway   = str(data.get("highway", "unclassified"))
        length    = safe_float(data.get("length", 1.0), 1.0)

        # ── Accident score (station-level real data) ──────────────────────
        acc = _match_score(road_name, accident_lookup, default_accident)
        data["accident_score"] = acc

        # ── Traffic density (area-level proxy) ────────────────────────────
        traf = _match_score(road_name, traffic_lookup, default_traffic)
        data["traffic_density"] = traf

        # ── Violation & enforcement (city-wide scalars) ───────────────────
        data["violation_score"]   = violation_score
        data["enforcement_score"] = enforcement_score
        data["compliance_score"]  = enforcement_score   # proxy

        # ── Road type rank (real OSM data) ────────────────────────────────
        data["road_type"] = road_type_rank(highway)

        # ── Road length (real OSM data) ───────────────────────────────────
        data["road_length"] = length

        # ── Placeholders for Step 4 ───────────────────────────────────────
        data["centrality_score"] = 0.0
        data["signal_delay"]     = 0.0

        # ── Composite congestion weight ───────────────────────────────────
        weight = (
            WEIGHT_TRAFFIC    * traf +
            WEIGHT_ACCIDENT   * acc +
            WEIGHT_VIOLATION  * violation_score +
            WEIGHT_SIGNAL     * 0.0 +          # updated in Step 4
            WEIGHT_COMPLIANCE * enforcement_score
        ) * length

        data["weight"] = max(weight, 1e-6)   # avoid zero weights

    return G


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_graph(G):
    ensure_output_dirs(GRAPHS_DIR)
    ox.save_graphml(G, str(GRAPH_WEIGHTED_PATH))
    print(f"  Weighted graph saved → {GRAPH_WEIGHTED_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    G = load_graph()

    print("  Building feature lookups ...")
    accident_lookup   = build_accident_lookup()
    traffic_lookup    = build_traffic_density_lookup()
    violation_score   = build_violation_score()
    enforcement_score = build_enforcement_score()

    print(f"  Accident stations loaded : {len(accident_lookup)}")
    print(f"  Traffic areas loaded     : {len(traffic_lookup)}")
    print(f"  Violation score (city)   : {violation_score:.4f}")
    print(f"  Enforcement score (city) : {enforcement_score:.4f}")

    G = assign_features(G, accident_lookup, traffic_lookup, violation_score, enforcement_score)
    save_graph(G)

    print("\n✓ Step 3 complete — graph features engineered.")


if __name__ == "__main__":
    main()

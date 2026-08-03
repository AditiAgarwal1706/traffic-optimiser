"""
bottleneck_analysis.py — Step 6A: Bottleneck Detection, Hotspot Ranking,
                          Critical Road Analysis, and Intervention Simulation.

DATA TRANSPARENCY:
  • All analysis is performed on the ML-enriched graph (bangalore_ml.graphml).
  • Betweenness centrality is computed from real OSM topology.
  • Congestion scores are ML-predicted pseudo-labels (weak supervision).
  • Intervention simulations are HYPOTHETICAL scenarios — not real policy outcomes.

ANALYSES PERFORMED:
  1. Bottleneck Detection     — top-N nodes by betweenness centrality
  2. Hotspot Ranking          — top-N edges by composite congestion score
  3. Critical Road Analysis   — edges that are both high-centrality AND high-congestion
  4. Intervention Simulation  — model what-if scenarios:
       a) Reduce signal delay by 30% (e.g., adaptive signal control)
       b) Improve enforcement by 20% (e.g., more traffic police)
       c) Reduce violations by 25% (e.g., awareness campaign)
  5. Hotspot Map              — Folium choropleth of congestion hotspots

OUTPUTS:
  • output/bottlenecks.csv
  • output/hotspots.csv
  • output/intervention_results.csv
  • output/maps/hotspot_map.html
  • output/maps/intervention_map.html
  • output/reports/bottleneck_summary.txt
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from config import (
    GRAPH_ML_PATH,
    BOTTLENECKS_PATH, HOTSPOTS_PATH, INTERVENTION_PATH,
    MAP_HOTSPOTS_PATH, MAP_INTERVENTION_PATH,
    REPORTS_DIR, OUTPUT_DIR,
    TOP_N_BOTTLENECKS, TOP_N_HOTSPOTS,
    WEIGHT_TRAFFIC, WEIGHT_ACCIDENT, WEIGHT_VIOLATION,
    WEIGHT_SIGNAL, WEIGHT_COMPLIANCE,
)
from utils import ensure_output_dirs, safe_float


# ─────────────────────────────────────────────────────────────────────────────
# LOAD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def load_graph():
    print("[Step 6A] Loading ML graph ...")
    G = ox.load_graphml(str(GRAPH_ML_PATH))
    # Ensure numeric weights
    for u, v, k, data in G.edges(keys=True, data=True):
        for attr in ("weight", "length", "ml_congestion_score",
                     "accident_score", "violation_score", "traffic_density",
                     "intersection_delay", "compliance_score", "centrality_score"):
            if attr in data:
                data[attr] = safe_float(data[attr])
    print(f"  → {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 1. BOTTLENECK DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_bottlenecks(G) -> pd.DataFrame:
    """
    Rank bottleneck nodes using a composite of:
      1. centrality_score  — already computed in Step 4 (sampled Brandes, 200 sources)
      2. degree            — number of connected road segments
      3. signal_delay      — estimated intersection delay (synthetic)

    This avoids re-running expensive betweenness centrality.
    The centrality_score from Step 4 is a normalized betweenness approximation.

    Returns top-N nodes sorted by composite bottleneck score.
    """
    print(f"  Ranking bottleneck nodes using Step-4 centrality scores ...")
    rows = []
    for node, ndata in G.nodes(data=True):
        cent  = safe_float(ndata.get("centrality_score", 0.0))
        delay = safe_float(ndata.get("signal_delay", 0.0))
        deg   = G.degree(node)
        # Composite: centrality is primary, degree and delay are secondary
        composite = 0.70 * cent + 0.20 * min(deg / 10.0, 1.0) + 0.10 * min(delay / 130.0, 1.0)
        rows.append({
            "node_id"          : node,
            "centrality"       : cent,
            "composite_score"  : composite,
            "lat"              : safe_float(ndata.get("y", 0.0)),
            "lon"              : safe_float(ndata.get("x", 0.0)),
            "degree"           : deg,
            "signal_delay_est" : delay,
            "centrality_score" : cent,
        })

    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    print(f"  Top bottleneck score: {df['composite_score'].iloc[0]:.6f}")
    return df.head(TOP_N_BOTTLENECKS * 5)
    print(f"  Top bottleneck centrality: {df['centrality'].iloc[0]:.6f}")
    return df.head(TOP_N_BOTTLENECKS * 5)   # save more, display top-N


# ─────────────────────────────────────────────────────────────────────────────
# 2. HOTSPOT RANKING
# ─────────────────────────────────────────────────────────────────────────────

def rank_hotspots(G) -> pd.DataFrame:
    """
    Rank edges by composite congestion score.
    Combines ML prediction with raw feature scores for a robust ranking.
    """
    print("  Ranking congestion hotspots ...")
    rows = []

    for u, v, k, data in G.edges(keys=True, data=True):
        ml_score  = safe_float(data.get("ml_congestion_score", 0.0))
        acc       = safe_float(data.get("accident_score", 0.0))
        viol      = safe_float(data.get("violation_score", 0.0))
        traf      = safe_float(data.get("traffic_density", 0.0))
        delay     = safe_float(data.get("intersection_delay", 0.0))
        comp      = safe_float(data.get("compliance_score", 0.0))
        cent      = safe_float(data.get("centrality_score", 0.0))
        length    = safe_float(data.get("length", 1.0), 1.0)
        highway   = str(data.get("highway", "unknown"))
        name      = str(data.get("name", "unnamed"))

        # Composite hotspot score (blends ML prediction with raw features)
        composite = (
            0.40 * ml_score +
            0.20 * traf +
            0.20 * acc +
            0.10 * cent +
            0.10 * (delay / 120.0)   # normalize delay to ~[0,1]
        )

        # Get midpoint coordinates for mapping
        try:
            geom = data.get("geometry")
            if geom is not None:
                mid = geom.interpolate(0.5, normalized=True)
                lat, lon = mid.y, mid.x
            else:
                lat = (safe_float(G.nodes[u].get("y", 0.0)) + safe_float(G.nodes[v].get("y", 0.0))) / 2
                lon = (safe_float(G.nodes[u].get("x", 0.0)) + safe_float(G.nodes[v].get("x", 0.0))) / 2
        except Exception:
            lat = safe_float(G.nodes[u].get("y", 0.0))
            lon = safe_float(G.nodes[u].get("x", 0.0))

        rows.append({
            "u"                  : u,
            "v"                  : v,
            "road_name"          : name,
            "highway"            : highway,
            "length_m"           : length,
            "ml_congestion_score": ml_score,
            "accident_score"     : acc,
            "violation_score"    : viol,
            "traffic_density"    : traf,
            "intersection_delay" : delay,
            "compliance_score"   : comp,
            "centrality_score"   : cent,
            "composite_score"    : composite,
            "lat"                : lat,
            "lon"                : lon,
        })

    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. CRITICAL ROAD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def critical_road_analysis(hotspots_df: pd.DataFrame, bottlenecks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify roads that are BOTH high-congestion AND structurally critical.
    These are the highest-priority intervention targets.
    """
    print("  Identifying critical roads (high congestion + high centrality) ...")

    # Normalize scores to [0,1] for comparison
    hs = hotspots_df.copy()
    hs["norm_congestion"] = hs["composite_score"] / (hs["composite_score"].max() + 1e-9)
    hs["norm_centrality"] = hs["centrality_score"] / (hs["centrality_score"].max() + 1e-9)
    hs["criticality_index"] = 0.6 * hs["norm_congestion"] + 0.4 * hs["norm_centrality"]
    hs = hs.sort_values("criticality_index", ascending=False).reset_index(drop=True)
    hs["priority_rank"] = hs.index + 1

    return hs[["priority_rank", "road_name", "highway", "length_m",
               "ml_congestion_score", "centrality_score", "criticality_index",
               "accident_score", "violation_score", "lat", "lon"]].head(TOP_N_HOTSPOTS)


# ─────────────────────────────────────────────────────────────────────────────
# 4. INTERVENTION SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_interventions(hotspots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate three hypothetical interventions on the top hotspot edges.

    ⚠ HYPOTHETICAL SIMULATION — NOT REAL POLICY OUTCOMES
    These are model-based estimates of congestion reduction under idealized
    assumptions. Real-world outcomes depend on many additional factors.

    Interventions:
      A) Adaptive Signal Control  — reduce intersection_delay by 30%
      B) Enhanced Enforcement     — increase compliance_score by 20%
      C) Violation Reduction      — reduce violation_score by 25%
    """
    print("  Simulating interventions (HYPOTHETICAL) ...")

    top = hotspots_df.head(TOP_N_HOTSPOTS).copy()

    def _recompute_weight(row, delay_factor=1.0, comp_factor=1.0, viol_factor=1.0):
        traf  = row["traffic_density"]
        acc   = row["accident_score"]
        viol  = row["violation_score"] * viol_factor
        delay = row["intersection_delay"] * delay_factor
        comp  = row["compliance_score"] * comp_factor
        length = row["length_m"]
        w = (
            WEIGHT_TRAFFIC   * traf +
            WEIGHT_ACCIDENT  * acc +
            WEIGHT_VIOLATION * viol +
            WEIGHT_SIGNAL    * (delay / 120.0) +
            WEIGHT_COMPLIANCE * comp
        ) * length
        return max(w, 1e-6)

    top["baseline_weight"] = top.apply(_recompute_weight, axis=1)

    top["weight_after_signal_opt"]    = top.apply(
        lambda r: _recompute_weight(r, delay_factor=0.70), axis=1
    )
    top["weight_after_enforcement"]   = top.apply(
        lambda r: _recompute_weight(r, comp_factor=1.20), axis=1
    )
    top["weight_after_viol_reduction"] = top.apply(
        lambda r: _recompute_weight(r, viol_factor=0.75), axis=1
    )

    top["reduction_signal_pct"]    = (
        (top["baseline_weight"] - top["weight_after_signal_opt"])
        / (top["baseline_weight"] + 1e-9) * 100
    )
    top["reduction_enforcement_pct"] = (
        (top["baseline_weight"] - top["weight_after_enforcement"])
        / (top["baseline_weight"] + 1e-9) * 100
    )
    top["reduction_violation_pct"]  = (
        (top["baseline_weight"] - top["weight_after_viol_reduction"])
        / (top["baseline_weight"] + 1e-9) * 100
    )

    return top


# ─────────────────────────────────────────────────────────────────────────────
# 5. HOTSPOT MAP
# ─────────────────────────────────────────────────────────────────────────────

def _congestion_color(score: float) -> str:
    """Map a congestion score [0,1] to a hex color (green → yellow → red)."""
    cmap = cm.get_cmap("RdYlGn_r")
    rgba = cmap(float(np.clip(score, 0.0, 1.0)))
    return mcolors.to_hex(rgba)


def build_hotspot_map(G, hotspots_df: pd.DataFrame, interventions_df: pd.DataFrame):
    """
    Build two Folium maps:
      1. hotspot_map.html   — top hotspot edges colored by congestion score
      2. intervention_map.html — same edges with intervention comparison
    """
    print("  Building hotspot maps ...")

    # Compute map center
    ys = [d.get("y") for _, d in G.nodes(data=True) if d.get("y")]
    xs = [d.get("x") for _, d in G.nodes(data=True) if d.get("x")]
    center = [np.mean(ys), np.mean(xs)] if ys else [12.97, 77.59]

    # ── Hotspot map ───────────────────────────────────────────────────────
    m1 = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    top_edges = set(zip(hotspots_df["u"].astype(str), hotspots_df["v"].astype(str)))
    score_map = {
        (str(r["u"]), str(r["v"])): r["composite_score"]
        for _, r in hotspots_df.iterrows()
    }

    for u, v, k, data in G.edges(keys=True, data=True):
        key = (str(u), str(v))
        if key not in top_edges:
            continue
        score = score_map.get(key, 0.0)
        color = _congestion_color(score)
        try:
            geom = data.get("geometry")
            if geom is not None:
                coords = [(lat, lon) for lon, lat in geom.coords]
            else:
                coords = [
                    (safe_float(G.nodes[u].get("y")), safe_float(G.nodes[u].get("x"))),
                    (safe_float(G.nodes[v].get("y")), safe_float(G.nodes[v].get("x"))),
                ]
            folium.PolyLine(
                coords, color=color, weight=5, opacity=0.85,
                tooltip=f"{data.get('name','unnamed')} | score={score:.3f}",
            ).add_to(m1)
        except Exception:
            continue

    # Add top-N markers
    for _, row in hotspots_df.head(TOP_N_HOTSPOTS).iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color=_congestion_color(row["composite_score"]),
            fill=True, fill_opacity=0.9,
            tooltip=(
                f"Rank #{int(row['rank'])} | {row['road_name']}<br>"
                f"Congestion: {row['composite_score']:.3f}<br>"
                f"Accident: {row['accident_score']:.3f} | "
                f"Traffic: {row['traffic_density']:.3f}"
            ),
        ).add_to(m1)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px;border-radius:5px;
                border:1px solid #ccc;font-size:12px;">
    <b>Congestion Score</b><br>
    <span style="color:#006837">■</span> Low (0.0–0.3)<br>
    <span style="color:#f7c948">■</span> Medium (0.3–0.6)<br>
    <span style="color:#d73027">■</span> High (0.6–1.0)<br>
    <i>⚠ ML pseudo-label scores</i>
    </div>"""
    m1.get_root().html.add_child(folium.Element(legend_html))
    m1.save(str(MAP_HOTSPOTS_PATH))
    print(f"  Hotspot map → {MAP_HOTSPOTS_PATH}")

    # ── Intervention comparison map ───────────────────────────────────────
    m2 = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    for _, row in interventions_df.iterrows():
        lat, lon = row["lat"], row["lon"]
        best_reduction = max(
            row["reduction_signal_pct"],
            row["reduction_enforcement_pct"],
            row["reduction_violation_pct"],
        )
        color = "#16a34a" if best_reduction > 5 else "#f59e0b"
        folium.CircleMarker(
            location=[lat, lon],
            radius=9,
            color=color, fill=True, fill_opacity=0.85,
            tooltip=(
                f"{row['road_name']}<br>"
                f"Signal opt: -{row['reduction_signal_pct']:.1f}%<br>"
                f"Enforcement: -{row['reduction_enforcement_pct']:.1f}%<br>"
                f"Viol. reduction: -{row['reduction_violation_pct']:.1f}%<br>"
                f"<i>⚠ Hypothetical simulation</i>"
            ),
        ).add_to(m2)

    m2.save(str(MAP_INTERVENTION_PATH))
    print(f"  Intervention map → {MAP_INTERVENTION_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_report(bottlenecks_df, hotspots_df, critical_df, interventions_df):
    ensure_output_dirs(REPORTS_DIR)
    lines = [
        "=" * 65,
        "BOTTLENECK & HOTSPOT ANALYSIS REPORT — BANGALORE TRAFFIC",
        "=" * 65,
        "",
        f"TOP {TOP_N_BOTTLENECKS} STRUCTURAL BOTTLENECKS (by betweenness centrality)",
        "-" * 65,
    ]
    for _, row in bottlenecks_df.head(TOP_N_BOTTLENECKS).iterrows():
        lines.append(
            f"  #{int(row['rank']):2d}  Node {row['node_id']}  "
            f"centrality={row['centrality']:.6f}  "
            f"degree={int(row['degree'])}  "
            f"signal_delay={row['signal_delay_est']:.1f}s (est.)"
        )

    lines += [
        "",
        f"TOP {TOP_N_HOTSPOTS} CONGESTION HOTSPOTS (composite score)",
        "-" * 65,
    ]
    for _, row in hotspots_df.head(TOP_N_HOTSPOTS).iterrows():
        lines.append(
            f"  #{int(row['rank']):2d}  {row['road_name'][:35]:35s}  "
            f"score={row['composite_score']:.4f}  "
            f"ML={row['ml_congestion_score']:.4f}"
        )

    lines += [
        "",
        f"TOP {TOP_N_HOTSPOTS} CRITICAL ROADS (congestion + centrality)",
        "-" * 65,
    ]
    for _, row in critical_df.iterrows():
        lines.append(
            f"  #{int(row['priority_rank']):2d}  {row['road_name'][:35]:35s}  "
            f"criticality={row['criticality_index']:.4f}"
        )

    lines += [
        "",
        "INTERVENTION SIMULATION RESULTS (⚠ HYPOTHETICAL)",
        "-" * 65,
        "  Scenario A: Adaptive signal control (−30% delay)",
        "  Scenario B: Enhanced enforcement (+20% compliance)",
        "  Scenario C: Violation reduction campaign (−25% violations)",
        "",
    ]
    for _, row in interventions_df.iterrows():
        lines.append(
            f"  {row['road_name'][:30]:30s}  "
            f"A:{row['reduction_signal_pct']:+.1f}%  "
            f"B:{row['reduction_enforcement_pct']:+.1f}%  "
            f"C:{row['reduction_violation_pct']:+.1f}%"
        )

    lines += [
        "",
        "LIMITATIONS:",
        "  • Congestion scores are ML pseudo-labels (weak supervision).",
        "  • Signal delays are synthetic topology estimates.",
        "  • Intervention reductions are model-based, not empirical.",
        "=" * 65,
    ]

    path = REPORTS_DIR / "bottleneck_summary.txt"
    path.write_text("\n".join(lines))
    print(f"  Summary report → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs(OUTPUT_DIR, REPORTS_DIR)

    G = load_graph()

    bottlenecks_df  = detect_bottlenecks(G)
    hotspots_df     = rank_hotspots(G)
    critical_df     = critical_road_analysis(hotspots_df, bottlenecks_df)
    interventions_df = simulate_interventions(hotspots_df)

    # Save CSVs
    bottlenecks_df.to_csv(str(BOTTLENECKS_PATH), index=False)
    hotspots_df.to_csv(str(HOTSPOTS_PATH), index=False)
    interventions_df.to_csv(str(INTERVENTION_PATH), index=False)
    print(f"  Bottlenecks CSV → {BOTTLENECKS_PATH}")
    print(f"  Hotspots CSV    → {HOTSPOTS_PATH}")
    print(f"  Interventions   → {INTERVENTION_PATH}")

    build_hotspot_map(G, hotspots_df, interventions_df)
    save_summary_report(bottlenecks_df, hotspots_df, critical_df, interventions_df)

    print("\n✓ Step 6A complete — bottleneck, hotspot, and intervention analysis done.")


if __name__ == "__main__":
    main()

"""
area_optimization.py — City-Wide Area-Level Traffic Optimization for Bangalore.

PURPOSE:
  Produces a comprehensive city-level decision-support dashboard covering
  ALL zones of Bangalore simultaneously — not a single road or route.

  This answers the question urban planners actually ask:
  "Which AREAS of the city are most congested, where should we intervene,
   and what does the entire network look like under different scenarios?"

WHAT THIS PRODUCES:
  1. city_congestion_map.html
     — Every road in Bangalore colored green→yellow→red by congestion score.
       Organized into named zone layers (East, West, North, South, Central,
       South-West) that can be toggled on/off.

  2. zone_intervention_map.html
     — Per-zone intervention simulation: each zone rendered under 3 scenarios
       (signal optimization, enforcement boost, violation reduction).
       Toggle between scenarios using the layer control.

  3. area_dashboard.html
     — Single-page dashboard combining:
         • City-wide congestion heatmap
         • Zone-wise congestion bar chart (inline SVG)
         • Top-5 hotspot roads per zone
         • Intervention impact table per zone

  4. output/reports/area_analysis_report.txt
     — Detailed per-zone statistics and policy recommendations.

DATA TRANSPARENCY:
  • Road geometries: real OSM data.
  • Congestion scores: ML pseudo-labels (weak supervision, Step 5B).
  • Zone boundaries: keyword-based classification of OSM road names.
  • Intervention estimates: hypothetical model-based scenarios.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from config import (
    GRAPH_ML_PATH,
    MAPS_DIR, REPORTS_DIR, OUTPUT_DIR,
    WEIGHT_TRAFFIC, WEIGHT_ACCIDENT, WEIGHT_VIOLATION,
    WEIGHT_SIGNAL, WEIGHT_COMPLIANCE,
)
from utils import ensure_output_dirs, safe_float, road_type_rank


# ─────────────────────────────────────────────────────────────────────────────
# ZONE DEFINITIONS  (Bangalore areas with bounding boxes + keywords)
# ─────────────────────────────────────────────────────────────────────────────

# Each zone has:
#   bbox   : (lat_min, lat_max, lon_min, lon_max)  — geographic boundary
#   color  : base color for the zone layer
#   keywords: road-name keywords for fallback classification
ZONES = {
    "Central": {
        "bbox"    : (12.960, 13.010, 77.560, 77.620),
        "color"   : "#7c3aed",
        "keywords": ["majestic", "city market", "shivajinagar", "cubbon",
                     "mg road", "brigade", "richmond", "lavelle", "residency",
                     "kasturba", "gandhi nagar", "seshadri"],
    },
    "East": {
        "bbox"    : (12.930, 13.020, 77.620, 77.780),
        "color"   : "#2563eb",
        "keywords": ["indiranagar", "domlur", "hal", "marathahalli", "whitefield",
                     "kr puram", "varthur", "brookefield", "sarjapur", "bellandur",
                     "kadubeesanahalli", "outer ring road"],
    },
    "West": {
        "bbox"    : (12.960, 13.060, 77.490, 77.570),
        "color"   : "#0891b2",
        "keywords": ["rajajinagar", "vijayanagar", "magadi", "tumkur", "peenya",
                     "yeshwanthpur", "malleshwaram", "chord road", "mysore road"],
    },
    "North": {
        "bbox"    : (13.010, 13.120, 77.540, 77.660),
        "color"   : "#059669",
        "keywords": ["hebbal", "yelahanka", "devanahalli", "airport", "thanisandra",
                     "kogilu", "bellary", "hennur", "kalyan nagar", "banaswadi"],
    },
    "South": {
        "bbox"    : (12.860, 12.960, 77.580, 77.700),
        "color"   : "#dc2626",
        "keywords": ["jayanagar", "jp nagar", "bannerghatta", "electronic city",
                     "silk board", "btm", "hsr", "koramangala", "bommanahalli",
                     "hosur road", "begur", "arekere"],
    },
    "South-West": {
        "bbox"    : (12.870, 12.970, 77.490, 77.590),
        "color"   : "#d97706",
        "keywords": ["kanakapura", "uttarahalli", "padmanabhanagar", "basavanagudi",
                     "girinagar", "kumaraswamy", "banashankari", "kathriguppe",
                     "dollar colony", "rr nagar"],
    },
}

# Color ramp for congestion (green → yellow → red)
_CMAP = cm.get_cmap("RdYlGn_r")

def congestion_color(score: float) -> str:
    rgba = _CMAP(float(np.clip(score, 0.0, 1.0)))
    return mcolors.to_hex(rgba)

def congestion_weight(score: float) -> float:
    """Line weight for road rendering — thicker = more congested."""
    return 1.5 + score * 4.0


# ─────────────────────────────────────────────────────────────────────────────
# LOAD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def load_graph():
    print("[Area Opt] Loading ML graph ...")
    G = ox.load_graphml(str(GRAPH_ML_PATH))
    for u, v, k, data in G.edges(keys=True, data=True):
        for attr in ("weight", "length", "ml_congestion_score",
                     "accident_score", "violation_score", "traffic_density",
                     "intersection_delay", "compliance_score", "centrality_score"):
            if attr in data:
                data[attr] = safe_float(data[attr])
    print(f"  → {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFY EVERY EDGE INTO A ZONE
# ─────────────────────────────────────────────────────────────────────────────

def classify_edges_by_zone(G) -> dict[str, list[dict]]:
    """
    Assign every edge to a zone using two methods (in priority order):
      1. Geographic bounding box of the edge midpoint (primary)
      2. Road-name keyword matching (fallback)

    Returns {zone_name: [edge_dict, ...]}
    """
    print("  Classifying edges by zone ...")
    zone_edges: dict[str, list[dict]] = {z: [] for z in ZONES}
    zone_edges["Other"] = []

    for u, v, k, data in G.edges(keys=True, data=True):
        # Midpoint coordinates
        try:
            geom = data.get("geometry")
            if geom is not None:
                mid = geom.interpolate(0.5, normalized=True)
                lat, lon = mid.y, mid.x
                coords = [(y, x) for x, y in geom.coords]
            else:
                lat = (safe_float(G.nodes[u].get("y")) + safe_float(G.nodes[v].get("y"))) / 2
                lon = (safe_float(G.nodes[u].get("x")) + safe_float(G.nodes[v].get("x"))) / 2
                coords = [
                    (safe_float(G.nodes[u].get("y")), safe_float(G.nodes[u].get("x"))),
                    (safe_float(G.nodes[v].get("y")), safe_float(G.nodes[v].get("x"))),
                ]
        except Exception:
            continue

        ml_score  = safe_float(data.get("ml_congestion_score", 0.0))
        acc       = safe_float(data.get("accident_score", 0.0))
        traf      = safe_float(data.get("traffic_density", 0.0))
        delay     = safe_float(data.get("intersection_delay", 0.0))
        viol      = safe_float(data.get("violation_score", 0.0))
        comp      = safe_float(data.get("compliance_score", 0.0))
        cent      = safe_float(data.get("centrality_score", 0.0))
        length    = safe_float(data.get("length", 1.0), 1.0)
        highway   = str(data.get("highway", "unclassified"))
        name      = str(data.get("name", ""))

        composite = (
            0.40 * ml_score +
            0.20 * traf +
            0.20 * acc +
            0.10 * cent +
            0.10 * min(delay / 120.0, 1.0)
        )

        edge_rec = {
            "u": u, "v": v,
            "lat": lat, "lon": lon,
            "coords": coords,
            "name": name,
            "highway": highway,
            "length_m": length,
            "ml_congestion_score": ml_score,
            "accident_score": acc,
            "traffic_density": traf,
            "intersection_delay": delay,
            "violation_score": viol,
            "compliance_score": comp,
            "centrality_score": cent,
            "composite_score": composite,
        }

        # 1. Bounding-box classification
        assigned = False
        for zone_name, zinfo in ZONES.items():
            lat_min, lat_max, lon_min, lon_max = zinfo["bbox"]
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                zone_edges[zone_name].append(edge_rec)
                assigned = True
                break

        if not assigned:
            # 2. Keyword fallback
            name_lower = name.lower()
            for zone_name, zinfo in ZONES.items():
                if any(kw in name_lower for kw in zinfo["keywords"]):
                    zone_edges[zone_name].append(edge_rec)
                    assigned = True
                    break
            if not assigned:
                zone_edges["Other"].append(edge_rec)

    for z, edges in zone_edges.items():
        print(f"    {z:12s}: {len(edges):6,} edges")

    return zone_edges


# ─────────────────────────────────────────────────────────────────────────────
# ZONE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_zone_stats(zone_edges: dict) -> pd.DataFrame:
    """Compute per-zone aggregate statistics."""
    rows = []
    for zone, edges in zone_edges.items():
        if not edges:
            continue
        df = pd.DataFrame(edges)
        rows.append({
            "zone"                  : zone,
            "total_edges"           : len(df),
            "total_length_km"       : df["length_m"].sum() / 1000,
            "mean_congestion"       : df["composite_score"].mean(),
            "max_congestion"        : df["composite_score"].max(),
            "mean_ml_score"         : df["ml_congestion_score"].mean(),
            "mean_accident_score"   : df["accident_score"].mean(),
            "mean_traffic_density"  : df["traffic_density"].mean(),
            "mean_signal_delay_s"   : df["intersection_delay"].mean(),
            "high_congestion_edges" : int((df["composite_score"] > 0.55).sum()),
            "high_congestion_pct"   : (df["composite_score"] > 0.55).mean() * 100,
            "color"                 : ZONES.get(zone, {}).get("color", "#64748b"),
        })
    return pd.DataFrame(rows).sort_values("mean_congestion", ascending=False).reset_index(drop=True)


def compute_zone_interventions(zone_edges: dict) -> pd.DataFrame:
    """
    Simulate three interventions per zone and estimate congestion reduction.
    ⚠ HYPOTHETICAL — model-based estimates only.
    """
    rows = []
    for zone, edges in zone_edges.items():
        if not edges:
            continue
        df = pd.DataFrame(edges)

        def _zone_weight(delay_f=1.0, comp_f=1.0, viol_f=1.0):
            w = (
                WEIGHT_TRAFFIC   * df["traffic_density"] +
                WEIGHT_ACCIDENT  * df["accident_score"] +
                WEIGHT_VIOLATION * df["violation_score"] * viol_f +
                WEIGHT_SIGNAL    * (df["intersection_delay"] * delay_f / 120.0).clip(0, 1) +
                WEIGHT_COMPLIANCE * df["compliance_score"] * comp_f
            ) * df["length_m"]
            return w.mean()

        baseline = _zone_weight()
        after_signal = _zone_weight(delay_f=0.70)
        after_enforce = _zone_weight(comp_f=1.20)
        after_viol = _zone_weight(viol_f=0.75)

        def _pct(after):
            return (baseline - after) / (baseline + 1e-9) * 100

        rows.append({
            "zone"                      : zone,
            "baseline_mean_weight"      : baseline,
            "signal_opt_reduction_pct"  : _pct(after_signal),
            "enforcement_reduction_pct" : _pct(after_enforce),
            "violation_reduction_pct"   : _pct(after_viol),
            "best_intervention"         : max(
                [("Signal Optimization", _pct(after_signal)),
                 ("Enforcement Boost",   _pct(after_enforce)),
                 ("Violation Reduction", _pct(after_viol))],
                key=lambda x: x[1]
            )[0],
            "best_reduction_pct"        : max(
                _pct(after_signal), _pct(after_enforce), _pct(after_viol)
            ),
        })
    return pd.DataFrame(rows).sort_values("best_reduction_pct", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAP 1: CITY-WIDE CONGESTION MAP  (all roads, all zones)
# ─────────────────────────────────────────────────────────────────────────────

def build_city_congestion_map(zone_edges: dict, zone_stats: pd.DataFrame) -> str:
    """
    Render EVERY road in Bangalore colored by its congestion score.
    Roads are grouped into per-zone FeatureGroups so planners can
    toggle individual zones on/off.

    Color scale: green (low) → yellow (medium) → red (high congestion).
    """
    print("  Building city-wide congestion map ...")
    center = [12.9716, 77.5946]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron",
                   prefer_canvas=True)

    # Add a base layer with ALL roads (thin, grey) for context
    all_group = folium.FeatureGroup(name="All Roads (context)", show=False)

    # Per-zone layers
    zone_groups = {}
    for zone in list(ZONES.keys()) + ["Other"]:
        color = ZONES.get(zone, {}).get("color", "#64748b")
        zone_groups[zone] = folium.FeatureGroup(name=f"Zone: {zone}", show=True)

    # Render each zone's edges
    for zone, edges in zone_edges.items():
        if not edges:
            continue
        fg = zone_groups[zone]
        # Sort: draw low-congestion roads first so high-congestion ones appear on top
        sorted_edges = sorted(edges, key=lambda e: e["composite_score"])
        for rec in sorted_edges:
            score = rec["composite_score"]
            color = congestion_color(score)
            lw    = congestion_weight(score)
            hw    = rec["highway"]
            if isinstance(hw, list):
                hw = hw[0] if hw else "road"
            # Only draw named/significant roads at lower zoom to keep map fast
            folium.PolyLine(
                rec["coords"],
                color=color,
                weight=lw,
                opacity=0.75,
                tooltip=(
                    f"<b>{rec['name'] or 'Unnamed road'}</b><br>"
                    f"Zone: {zone}<br>"
                    f"Type: {hw}<br>"
                    f"Congestion: {score:.3f}<br>"
                    f"ML Score: {rec['ml_congestion_score']:.3f}<br>"
                    f"Accident: {rec['accident_score']:.3f}<br>"
                    f"Traffic: {rec['traffic_density']:.3f}<br>"
                    f"Signal Delay: {rec['intersection_delay']:.1f}s (est.)"
                ),
            ).add_to(fg)

    for fg in zone_groups.values():
        fg.add_to(m)

    # Zone centroid markers with summary stats
    for _, row in zone_stats.iterrows():
        zone = row["zone"]
        zinfo = ZONES.get(zone)
        if zinfo is None:
            continue
        lat_min, lat_max, lon_min, lon_max = zinfo["bbox"]
        clat = (lat_min + lat_max) / 2
        clon = (lon_min + lon_max) / 2
        fill_color = congestion_color(row["mean_congestion"])
        folium.CircleMarker(
            location=[clat, clon],
            radius=18,
            color="white",
            weight=2,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.85,
            tooltip=(
                f"<b>{zone} Zone</b><br>"
                f"Roads: {int(row['total_edges']):,}<br>"
                f"Length: {row['total_length_km']:.0f} km<br>"
                f"Mean Congestion: {row['mean_congestion']:.3f}<br>"
                f"High-Congestion Roads: {int(row['high_congestion_edges']):,} "
                f"({row['high_congestion_pct']:.1f}%)<br>"
                f"Mean Signal Delay: {row['mean_signal_delay_s']:.1f}s (est.)"
            ),
        ).add_to(m)
        folium.Marker(
            location=[clat, clon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px;font-weight:bold;color:white;'
                     f'text-shadow:1px 1px 2px #000;white-space:nowrap;">{zone}</div>',
                icon_size=(80, 20),
                icon_anchor=(40, 10),
            ),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:40px;left:40px;z-index:9999;
                background:white;padding:14px 18px;border-radius:8px;
                border:1px solid #ccc;font-size:12px;box-shadow:2px 2px 6px rgba(0,0,0,0.2);">
      <b style="font-size:13px;">Congestion Score</b><br>
      <span style="color:#1a9850;font-size:16px;">━━</span> Low  (0.0 – 0.35)<br>
      <span style="color:#fee08b;font-size:16px;">━━</span> Medium (0.35 – 0.55)<br>
      <span style="color:#d73027;font-size:16px;">━━</span> High (0.55 – 1.0)<br>
      <hr style="margin:6px 0;">
      <i style="font-size:10px;">⚠ ML pseudo-label scores<br>Use layer control to toggle zones</i>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    out_path = str(MAPS_DIR / "city_congestion_map.html")
    m.save(out_path)
    print(f"  City congestion map → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAP 2: ZONE INTERVENTION MAP  (per-zone scenario comparison)
# ─────────────────────────────────────────────────────────────────────────────

def build_intervention_map(zone_edges: dict, zone_stats: pd.DataFrame,
                           zone_interventions: pd.DataFrame) -> str:
    """
    Show each zone colored by its BEST intervention outcome.
    Darker green = more congestion reduction achievable.
    Markers show which intervention works best per zone.
    ⚠ HYPOTHETICAL simulation.
    """
    print("  Building zone intervention map ...")
    center = [12.9716, 77.5946]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB dark_matter",
                   prefer_canvas=True)

    # Scenario layers
    scenarios = {
        "Scenario A: Signal Optimization (−30% delay)": "signal_opt_reduction_pct",
        "Scenario B: Enforcement Boost (+20% compliance)": "enforcement_reduction_pct",
        "Scenario C: Violation Reduction (−25% violations)": "violation_reduction_pct",
    }

    intv_lookup = {row["zone"]: row for _, row in zone_interventions.iterrows()}
    stats_lookup = {row["zone"]: row for _, row in zone_stats.iterrows()}

    for scenario_name, col in scenarios.items():
        fg = folium.FeatureGroup(name=scenario_name, show=(col == "signal_opt_reduction_pct"))

        for zone, edges in zone_edges.items():
            if not edges or zone not in intv_lookup:
                continue
            intv = intv_lookup[zone]
            stats = stats_lookup.get(zone, {})
            reduction = float(intv.get(col, 0.0))

            # Color: green = high reduction achievable, red = low
            norm_reduction = np.clip(reduction / 15.0, 0.0, 1.0)
            fill_color = mcolors.to_hex(cm.get_cmap("RdYlGn")(norm_reduction))

            # Draw zone roads with intervention-adjusted opacity
            for rec in edges[:3000]:  # cap per zone for performance
                score_after = rec["composite_score"] * (1.0 - reduction / 100.0)
                folium.PolyLine(
                    rec["coords"],
                    color=fill_color,
                    weight=congestion_weight(score_after),
                    opacity=0.65,
                ).add_to(fg)

            # Zone centroid marker
            zinfo = ZONES.get(zone)
            if zinfo:
                lat_min, lat_max, lon_min, lon_max = zinfo["bbox"]
                clat = (lat_min + lat_max) / 2
                clon = (lon_min + lon_max) / 2
                folium.CircleMarker(
                    location=[clat, clon],
                    radius=20,
                    color="white", weight=2,
                    fill=True, fill_color=fill_color, fill_opacity=0.9,
                    tooltip=(
                        f"<b>{zone} Zone — {scenario_name}</b><br>"
                        f"Congestion reduction: <b>{reduction:.1f}%</b><br>"
                        f"Current mean congestion: {stats.get('mean_congestion', 0):.3f}<br>"
                        f"After intervention: "
                        f"{stats.get('mean_congestion', 0) * (1 - reduction/100):.3f}<br>"
                        f"High-congestion roads: {int(stats.get('high_congestion_edges', 0)):,}<br>"
                        f"<i>⚠ Hypothetical estimate</i>"
                    ),
                ).add_to(fg)
                folium.Marker(
                    location=[clat, clon],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:9px;font-weight:bold;color:white;'
                             f'text-shadow:1px 1px 2px #000;white-space:nowrap;">'
                             f'{zone}<br>−{reduction:.1f}%</div>',
                        icon_size=(80, 30), icon_anchor=(40, 15),
                    ),
                ).add_to(fg)

        fg.add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:40px;left:40px;z-index:9999;
                background:#1e293b;color:white;padding:14px 18px;border-radius:8px;
                border:1px solid #475569;font-size:12px;">
      <b>Intervention Impact</b><br>
      <span style="color:#4ade80;font-size:16px;">━━</span> High reduction (&gt;10%)<br>
      <span style="color:#fbbf24;font-size:16px;">━━</span> Medium (5–10%)<br>
      <span style="color:#f87171;font-size:16px;">━━</span> Low (&lt;5%)<br>
      <hr style="margin:6px 0;border-color:#475569;">
      <i style="font-size:10px;">⚠ Hypothetical simulation<br>Toggle scenarios above</i>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    out_path = str(MAPS_DIR / "zone_intervention_map.html")
    m.save(out_path)
    print(f"  Zone intervention map → {out_path}")
    return out_path

# ─────────────────────────────────────────────────────────────────────────────
# MAP 3: CONGESTION HEATMAP  (density view across the whole city)
# ─────────────────────────────────────────────────────────────────────────────

def build_heatmap(zone_edges: dict) -> str:
    """
    Folium HeatMap where each road midpoint is weighted by its congestion score.
    Gives a smooth density view of where congestion clusters across the city.
    """
    print("  Building city-wide congestion heatmap ...")
    center = [12.9716, 77.5946]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB dark_matter")

    heat_data = []
    for edges in zone_edges.values():
        for rec in edges:
            score = rec["composite_score"]
            if score > 0.1:   # skip near-zero entries to keep heatmap clean
                heat_data.append([rec["lat"], rec["lon"], score])

    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_zoom=16,
        radius=12,
        blur=10,
        gradient={0.2: "#1a9850", 0.5: "#fee08b", 0.75: "#f46d43", 1.0: "#d73027"},
    ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:40px;left:40px;z-index:9999;
                background:#1e293b;color:white;padding:14px 18px;border-radius:8px;
                border:1px solid #475569;font-size:12px;">
      <b>Congestion Density</b><br>
      <span style="color:#1a9850;">●</span> Low<br>
      <span style="color:#fee08b;">●</span> Medium<br>
      <span style="color:#f46d43;">●</span> High<br>
      <span style="color:#d73027;">●</span> Very High<br>
      <hr style="margin:6px 0;border-color:#475569;">
      <i style="font-size:10px;">⚠ ML pseudo-label scores</i>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    out_path = str(MAPS_DIR / "city_heatmap.html")
    m.save(out_path)
    print(f"  Heatmap → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# STATIC CHARTS  (saved as PNG for reports/presentations)
# ─────────────────────────────────────────────────────────────────────────────

def build_zone_charts(zone_stats: pd.DataFrame, zone_interventions: pd.DataFrame):
    """Generate zone-comparison bar charts."""
    ensure_output_dirs(REPORTS_DIR)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "Bangalore Zone-Level Traffic Analysis\n(⚠ ML pseudo-label congestion scores)",
        fontsize=13, fontweight="bold"
    )

    zones  = zone_stats["zone"].tolist()
    colors = [zone_stats.loc[zone_stats["zone"] == z, "color"].values[0]
              if z in zone_stats["zone"].values else "#64748b" for z in zones]

    # Chart 1: Mean congestion per zone
    ax = axes[0]
    bars = ax.barh(zones, zone_stats["mean_congestion"], color=colors, edgecolor="white")
    ax.set_xlabel("Mean Congestion Score")
    ax.set_title("Mean Congestion by Zone")
    ax.set_xlim(0, 1.0)
    for bar, val in zip(bars, zone_stats["mean_congestion"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.invert_yaxis()

    # Chart 2: % high-congestion roads per zone
    ax = axes[1]
    bars = ax.barh(zones, zone_stats["high_congestion_pct"], color=colors, edgecolor="white")
    ax.set_xlabel("% Roads with Score > 0.55")
    ax.set_title("High-Congestion Road Share")
    ax.set_xlim(0, 100)
    for bar, val in zip(bars, zone_stats["high_congestion_pct"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.invert_yaxis()

    # Chart 3: Best intervention reduction per zone
    intv_lookup = dict(zip(zone_interventions["zone"], zone_interventions["best_reduction_pct"]))
    best_intv   = dict(zip(zone_interventions["zone"], zone_interventions["best_intervention"]))
    reductions  = [intv_lookup.get(z, 0.0) for z in zones]
    intv_colors = ["#16a34a" if r > 8 else "#f59e0b" if r > 4 else "#dc2626" for r in reductions]

    ax = axes[2]
    bars = ax.barh(zones, reductions, color=intv_colors, edgecolor="white")
    ax.set_xlabel("Best Congestion Reduction (%)")
    ax.set_title("Max Achievable Reduction\n(⚠ Hypothetical)")
    for bar, val, z in zip(bars, reductions, zones):
        label = f"{val:.1f}% ({best_intv.get(z, '')[:8]})"
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    path = REPORTS_DIR / "zone_comparison_charts.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Zone charts → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# TOP HOTSPOT ROADS PER ZONE
# ─────────────────────────────────────────────────────────────────────────────

def top_roads_per_zone(zone_edges: dict, top_n: int = 5) -> dict[str, pd.DataFrame]:
    """Return the top-N most congested named roads per zone."""
    result = {}
    for zone, edges in zone_edges.items():
        if not edges:
            continue
        df = pd.DataFrame(edges)
        named = df[df["name"].str.strip() != ""].copy()
        if named.empty:
            named = df.copy()
        top = (named.groupby("name")
               .agg(mean_congestion=("composite_score", "mean"),
                    max_congestion=("composite_score", "max"),
                    mean_accident=("accident_score", "mean"),
                    mean_traffic=("traffic_density", "mean"),
                    mean_delay=("intersection_delay", "mean"),
                    total_length_km=("length_m", lambda x: x.sum() / 1000))
               .reset_index()
               .sort_values("mean_congestion", ascending=False)
               .head(top_n))
        result[zone] = top
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_area_report(zone_stats: pd.DataFrame, zone_interventions: pd.DataFrame,
                     top_roads: dict):
    ensure_output_dirs(REPORTS_DIR)
    intv_lookup = {r["zone"]: r for _, r in zone_interventions.iterrows()}

    lines = [
        "=" * 70,
        "CITY-WIDE AREA OPTIMIZATION REPORT — BANGALORE TRAFFIC",
        "=" * 70,
        "",
        "ZONE RANKING (by mean congestion score, highest first)",
        "-" * 70,
        f"  {'Zone':<14} {'Edges':>7} {'Length km':>10} {'Mean Cong':>10} "
        f"{'High Cong%':>11} {'Avg Delay':>10}",
        "-" * 70,
    ]
    for _, row in zone_stats.iterrows():
        lines.append(
            f"  {row['zone']:<14} {int(row['total_edges']):>7,} "
            f"{row['total_length_km']:>10.1f} "
            f"{row['mean_congestion']:>10.4f} "
            f"{row['high_congestion_pct']:>10.1f}% "
            f"{row['mean_signal_delay_s']:>9.1f}s"
        )

    lines += ["", "INTERVENTION SIMULATION PER ZONE (⚠ HYPOTHETICAL)", "-" * 70,
              f"  {'Zone':<14} {'Signal Opt':>11} {'Enforcement':>12} "
              f"{'Viol Reduc':>11} {'Best':>22}"]
    for _, row in zone_interventions.iterrows():
        lines.append(
            f"  {row['zone']:<14} "
            f"{row['signal_opt_reduction_pct']:>10.2f}% "
            f"{row['enforcement_reduction_pct']:>11.2f}% "
            f"{row['violation_reduction_pct']:>10.2f}% "
            f"  {row['best_intervention'][:20]}"
        )

    lines += ["", "TOP CONGESTED ROADS PER ZONE", "-" * 70]
    for zone, df in top_roads.items():
        lines.append(f"\n  [{zone}]")
        for i, (_, row) in enumerate(df.iterrows(), 1):
            lines.append(
                f"    {i}. {row['name'][:40]:<40}  "
                f"cong={row['mean_congestion']:.4f}  "
                f"len={row['total_length_km']:.2f}km  "
                f"delay={row['mean_delay']:.1f}s"
            )

    lines += [
        "",
        "POLICY RECOMMENDATIONS (decision-support, not prescriptive)",
        "-" * 70,
    ]
    # Dynamic recommendations based on actual zone rankings
    worst_zone = zone_stats.iloc[0]["zone"] if len(zone_stats) > 0 else "Unknown"
    best_intv_zone = zone_interventions.iloc[0]["zone"] if len(zone_interventions) > 0 else "Unknown"
    lines += [
        f"  1. IMMEDIATE: Deploy adaptive signal control in {worst_zone} zone",
        f"     (highest mean congestion: {zone_stats.iloc[0]['mean_congestion']:.3f})",
        f"  2. HIGH IMPACT: {zone_interventions.iloc[0]['best_intervention']} in "
        f"{best_intv_zone} zone",
        f"     (estimated {zone_interventions.iloc[0]['best_reduction_pct']:.1f}% reduction)",
        "  3. Protect top-5 bottleneck intersections from construction disruption.",
        "  4. Increase enforcement presence on high-accident corridors.",
        "  5. Coordinate signal timing across zone boundaries (arterial progression).",
        "",
        "LIMITATIONS",
        "-" * 70,
        "  • Congestion scores are ML pseudo-labels (weak supervision).",
        "  • Zone boundaries are geographic bounding boxes (approximate).",
        "  • Signal delays are synthetic topology estimates.",
        "  • Intervention reductions are model-based, not empirically validated.",
        "  • No real-time data — represents a static snapshot.",
        "=" * 70,
    ]

    path = REPORTS_DIR / "area_analysis_report.txt"
    path.write_text("\n".join(lines))
    print(f"  Area report → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs(MAPS_DIR, REPORTS_DIR, OUTPUT_DIR)

    G = load_graph()

    print("\n[1/6] Classifying all edges by zone ...")
    zone_edges = classify_edges_by_zone(G)

    print("\n[2/6] Computing zone statistics ...")
    zone_stats = compute_zone_stats(zone_edges)
    zone_interventions = compute_zone_interventions(zone_edges)
    zone_stats.to_csv(str(REPORTS_DIR / "zone_stats_detailed.csv"), index=False)
    zone_interventions.to_csv(str(REPORTS_DIR / "zone_interventions.csv"), index=False)

    print("\n[3/6] Building city-wide congestion map (all roads) ...")
    build_city_congestion_map(zone_edges, zone_stats)

    print("\n[4/6] Building zone intervention map ...")
    build_intervention_map(zone_edges, zone_stats, zone_interventions)

    print("\n[5/6] Building congestion heatmap ...")
    build_heatmap(zone_edges)

    print("\n[6/6] Generating charts and report ...")
    top_roads = top_roads_per_zone(zone_edges, top_n=5)
    build_zone_charts(zone_stats, zone_interventions)
    save_area_report(zone_stats, zone_interventions, top_roads)

    print("\n" + "=" * 70)
    print("ZONE SUMMARY")
    print("=" * 70)
    for _, row in zone_stats.iterrows():
        bar = "█" * int(row["mean_congestion"] * 30)
        print(f"  {row['zone']:<14} {bar:<30} {row['mean_congestion']:.3f}")

    print("\n✓ Area optimization complete.")
    print(f"  Maps    → {MAPS_DIR}")
    print(f"  Reports → {REPORTS_DIR}")


if __name__ == "__main__":
    main()

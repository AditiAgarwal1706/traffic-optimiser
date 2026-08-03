"""
emergency_routing.py — Step 6B: Emergency Vehicle Routing and Movement Analysis.

PURPOSE:
  This module is a DECISION-SUPPORT TOOL for traffic authorities, NOT a
  consumer navigation app. It answers the question:
  "Which corridors should be prioritized / cleared for emergency vehicles?"

DATA TRANSPARENCY:
  • Road network: real OSM data.
  • Congestion weights: ML pseudo-labels (weak supervision).
  • Emergency weight: derived by reducing weight on high-hierarchy roads
    (motorway, trunk, primary) — reflects real-world emergency protocols.
  • Route comparison: shortest-distance vs. least-congestion vs. emergency-priority.

ANALYSES:
  1. Emergency-priority route (Silk Board → Whitefield as default example)
  2. Comparison: distance route vs. congestion route vs. emergency route
  3. Corridor vulnerability: which emergency routes pass through bottlenecks?
  4. Alternative route analysis: top-3 emergency paths
  5. Folium map with all routes overlaid

OUTPUTS:
  • output/emergency_analysis.csv
  • output/maps/emergency_route.html
  • output/reports/emergency_summary.txt
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import folium

from config import (
    GRAPH_ML_PATH,
    EMERGENCY_SOURCE, EMERGENCY_TARGET,
    EMERGENCY_PATH, MAP_EMERGENCY_PATH,
    REPORTS_DIR, OUTPUT_DIR,
)
from utils import ensure_output_dirs, safe_float, road_type_rank


# ─────────────────────────────────────────────────────────────────────────────
# LOAD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def load_graph():
    print("[Step 6B] Loading ML graph ...")
    G = ox.load_graphml(str(GRAPH_ML_PATH))
    # Ensure all weights are numeric
    for u, v, k, data in G.edges(keys=True, data=True):
        for attr in ("weight", "length", "ml_congestion_score"):
            if attr in data:
                data[attr] = safe_float(data[attr], 1.0)
    print(f"  → {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# BUILD EMERGENCY WEIGHT
# ─────────────────────────────────────────────────────────────────────────────

def build_emergency_weights(G) -> object:
    """
    Assign 'emergency_weight' to each edge.

    Emergency vehicles prefer:
      • High-hierarchy roads (motorway, trunk, primary) — lower weight
      • Low-congestion roads — lower weight
      • Avoid residential/service roads — higher weight

    ⚠ ASSUMPTION: Emergency vehicles follow road hierarchy preferences.
    Real emergency routing also depends on real-time signal preemption,
    which is not modeled here.
    """
    print("  Building emergency weights ...")

    hierarchy_factor = {7: 0.2, 6: 0.3, 5: 0.4, 4: 0.6, 3: 0.8, 2: 1.2, 1: 1.5}

    for u, v, k, data in G.edges(keys=True, data=True):
        base_weight = safe_float(data.get("weight", 1.0), 1.0)
        highway     = str(data.get("highway", "unclassified"))
        rank        = road_type_rank(highway)
        factor      = hierarchy_factor.get(rank, 1.0)
        data["emergency_weight"] = base_weight * factor

    return G


# ─────────────────────────────────────────────────────────────────────────────
# FIND NEAREST NODES
# ─────────────────────────────────────────────────────────────────────────────

def get_source_target(G, source_latlon: tuple, target_latlon: tuple) -> tuple[int, int]:
    src_lat, src_lon = source_latlon
    tgt_lat, tgt_lon = target_latlon
    source = ox.distance.nearest_nodes(G, src_lon, src_lat)
    target = ox.distance.nearest_nodes(G, tgt_lon, tgt_lat)
    print(f"  Source node: {source} (near {src_lat:.4f}, {src_lon:.4f})")
    print(f"  Target node: {target} (near {tgt_lat:.4f}, {tgt_lon:.4f})")
    return source, target


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE ROUTES
# ─────────────────────────────────────────────────────────────────────────────

def compute_routes(G, source: int, target: int) -> dict:
    """
    Compute three routes:
      1. Shortest distance (weight=length)
      2. Least congestion (weight=weight)
      3. Emergency priority (weight=emergency_weight)
    """
    routes = {}

    for name, weight_key in [
        ("shortest_distance", "length"),
        ("least_congestion",  "weight"),
        ("emergency_priority","emergency_weight"),
    ]:
        try:
            route = nx.shortest_path(G, source=source, target=target, weight=weight_key)
            total_length = sum(
                safe_float(G[u][v][0].get("length", 0.0))
                for u, v in zip(route[:-1], route[1:])
            )
            total_weight = sum(
                safe_float(G[u][v][0].get(weight_key, 0.0))
                for u, v in zip(route[:-1], route[1:])
            )
            avg_congestion = np.mean([
                safe_float(G[u][v][0].get("ml_congestion_score", 0.0))
                for u, v in zip(route[:-1], route[1:])
            ])
            routes[name] = {
                "route"          : route,
                "nodes"          : len(route),
                "total_length_m" : total_length,
                "total_weight"   : total_weight,
                "avg_congestion" : avg_congestion,
            }
            print(f"  {name:25s}: {len(route)} nodes, "
                  f"{total_length/1000:.2f} km, "
                  f"avg_congestion={avg_congestion:.4f}")
        except nx.NetworkXNoPath:
            print(f"  {name}: No path found.")
            routes[name] = None

    return routes


# ─────────────────────────────────────────────────────────────────────────────
# CORRIDOR VULNERABILITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_corridor_vulnerability(G, routes: dict) -> pd.DataFrame:
    """
    For each route, identify edges that are high-congestion bottlenecks.
    These are the most vulnerable segments for emergency movement.
    """
    rows = []
    for route_name, route_info in routes.items():
        if route_info is None:
            continue
        route = route_info["route"]
        for u, v in zip(route[:-1], route[1:]):
            data = G[u][v][0]
            rows.append({
                "route"              : route_name,
                "u"                  : u,
                "v"                  : v,
                "road_name"          : str(data.get("name", "unnamed")),
                "highway"            : str(data.get("highway", "unknown")),
                "length_m"           : safe_float(data.get("length", 0.0)),
                "ml_congestion_score": safe_float(data.get("ml_congestion_score", 0.0)),
                "intersection_delay" : safe_float(data.get("intersection_delay", 0.0)),
                "centrality_score"   : safe_float(data.get("centrality_score", 0.0)),
                "emergency_weight"   : safe_float(data.get("emergency_weight", 0.0)),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("ml_congestion_score", ascending=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FOLIUM MAP
# ─────────────────────────────────────────────────────────────────────────────

def _draw_route(G, route: list, m: folium.Map, color: str, label: str, weight: int = 5):
    """Draw a route on a Folium map."""
    latlons = [(safe_float(G.nodes[n].get("y")), safe_float(G.nodes[n].get("x"))) for n in route]
    folium.PolyLine(latlons, color=color, weight=weight, opacity=0.85,
                    tooltip=label).add_to(m)


def build_emergency_map(G, routes: dict, source: int, target: int):
    ys = [d.get("y") for _, d in G.nodes(data=True) if d.get("y")]
    xs = [d.get("x") for _, d in G.nodes(data=True) if d.get("x")]
    center = [np.mean(ys), np.mean(xs)] if ys else [12.97, 77.59]

    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

    colors = {
        "shortest_distance" : "#2563EB",   # blue
        "least_congestion"  : "#16a34a",   # green
        "emergency_priority": "#dc2626",   # red
    }
    labels = {
        "shortest_distance" : "Shortest Distance",
        "least_congestion"  : "Least Congestion",
        "emergency_priority": "Emergency Priority",
    }

    for name, info in routes.items():
        if info is None:
            continue
        _draw_route(G, info["route"], m, colors[name], labels[name])

    # Source / target markers
    src_y = safe_float(G.nodes[source].get("y"))
    src_x = safe_float(G.nodes[source].get("x"))
    tgt_y = safe_float(G.nodes[target].get("y"))
    tgt_x = safe_float(G.nodes[target].get("x"))

    folium.Marker([src_y, src_x], tooltip="SOURCE: Silk Board",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([tgt_y, tgt_x], tooltip="TARGET: Whitefield",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px;border-radius:5px;
                border:1px solid #ccc;font-size:12px;">
    <b>Route Comparison</b><br>
    <span style="color:#2563EB">━━</span> Shortest Distance<br>
    <span style="color:#16a34a">━━</span> Least Congestion<br>
    <span style="color:#dc2626">━━</span> Emergency Priority<br>
    <i>⚠ ML pseudo-label weights</i>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(str(MAP_EMERGENCY_PATH))
    print(f"  Emergency map → {MAP_EMERGENCY_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_emergency_report(routes: dict, vulnerability_df: pd.DataFrame):
    ensure_output_dirs(REPORTS_DIR)
    lines = [
        "=" * 65,
        "EMERGENCY ROUTING ANALYSIS — BANGALORE TRAFFIC",
        "Corridor: Silk Board → Whitefield",
        "=" * 65,
        "",
        "ROUTE COMPARISON:",
        "-" * 65,
    ]
    for name, info in routes.items():
        if info is None:
            lines.append(f"  {name}: No path found.")
            continue
        lines.append(
            f"  {name:25s}: {info['nodes']} nodes | "
            f"{info['total_length_m']/1000:.2f} km | "
            f"avg_congestion={info['avg_congestion']:.4f}"
        )

    lines += [
        "",
        "TOP VULNERABLE SEGMENTS (emergency route, high congestion):",
        "-" * 65,
    ]
    em_vuln = vulnerability_df[vulnerability_df["route"] == "emergency_priority"]
    for _, row in em_vuln.head(10).iterrows():
        lines.append(
            f"  {row['road_name'][:35]:35s}  "
            f"congestion={row['ml_congestion_score']:.4f}  "
            f"delay={row['intersection_delay']:.1f}s (est.)"
        )

    lines += [
        "",
        "RECOMMENDATIONS (decision-support, not prescriptive):",
        "  1. Pre-clear high-congestion segments on emergency route.",
        "  2. Consider signal preemption at high-delay intersections.",
        "  3. Maintain emergency-priority corridors on primary roads.",
        "",
        "LIMITATIONS:",
        "  • Congestion weights are ML pseudo-labels.",
        "  • Signal delays are synthetic estimates.",
        "  • Real-time conditions not modeled.",
        "=" * 65,
    ]

    path = REPORTS_DIR / "emergency_summary.txt"
    path.write_text("\n".join(lines))
    print(f"  Emergency report → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs(OUTPUT_DIR, REPORTS_DIR)

    G = load_graph()
    G = build_emergency_weights(G)

    source, target = get_source_target(G, EMERGENCY_SOURCE, EMERGENCY_TARGET)
    routes = compute_routes(G, source, target)

    vulnerability_df = analyze_corridor_vulnerability(G, routes)
    vulnerability_df.to_csv(str(EMERGENCY_PATH), index=False)
    print(f"  Emergency analysis CSV → {EMERGENCY_PATH}")

    build_emergency_map(G, routes, source, target)
    save_emergency_report(routes, vulnerability_df)

    print("\n✓ Step 6B complete — emergency routing analysis done.")


if __name__ == "__main__":
    main()

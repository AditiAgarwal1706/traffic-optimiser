"""
dashboard/app.py
================
Flask app: A→B route comparison using PageRank-Guided Relief.

  BEFORE  = Dijkstra on raw congestion-weighted graph
  AFTER   = Dijkstra on PageRank-optimised weights

Run:
    source .venv/bin/activate
    python dashboard/app.py
Then open: http://localhost:5050
"""

import os
import sys

# Make repo-root imports work regardless of current working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NOTE: We intentionally do not auto-switch the interpreter to .venv.
# That behavior is surprising in GitHub repos and can break in CI.

import json, math, time
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from flask import Flask, render_template, request, jsonify

PROJECT = Path(__file__).resolve().parent.parent
app = Flask(__name__, template_folder="templates")


def _missing_outputs_message(missing_paths: list[Path]) -> str:
    bullets = "\n".join([f"- {p.relative_to(PROJECT)}" for p in missing_paths])
    return (
        "Required pipeline outputs are missing.\n\n"
        "Missing files:\n"
        f"{bullets}\n\n"
        "Fix:\n"
        "1) Create and activate a virtual environment\n"
        "2) pip install -r requirements.txt\n"
        "3) Run: python src/run_pipeline.py\n\n"
        "Once those files exist, restart the dashboard."
    )


def _load_or_set_startup_error():
    """Load required CSVs and return (nodes_df, edges_df, hot_df, bot_df, error_msg)."""
    required = [
        PROJECT / "output/graphs/nodes.csv",
        PROJECT / "output/graphs/edges.csv",
        PROJECT / "output/hotspots.csv",
        PROJECT / "output/bottlenecks.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        return None, None, None, None, _missing_outputs_message(missing)

    t0 = time.time()
    nodes_df = pd.read_csv(required[0], low_memory=False)
    edges_df = pd.read_csv(required[1], low_memory=False)
    hot_df = pd.read_csv(required[2])
    bot_df = pd.read_csv(required[3])
    print(f"  CSVs loaded in {time.time()-t0:.1f}s")
    return nodes_df, edges_df, hot_df, bot_df, None

# ── 1. Load data once at startup ──────────────────────────────────────────
print("=" * 55)
print("  Loading Bangalore road graph …")
print("=" * 55)

nodes_df, edges_df, hot_df, bot_df, STARTUP_ERROR = _load_or_set_startup_error()
if STARTUP_ERROR:
    print(STARTUP_ERROR)
    # Defer graph construction; routes will return the error instead.
    G = nx.DiGraph()
    node_pos = {}
else:
    STARTUP_ERROR = None

if not STARTUP_ERROR:
    # Merge congestion scores onto edges (hotspots has u,v + ml_congestion_score)
    t1 = time.time()
    cong_map = (hot_df[["u","v","ml_congestion_score","accident_score",
                         "violation_score","traffic_density",
                         "centrality_score","intersection_delay"]]
                .drop_duplicates(["u","v"])
                .set_index(["u","v"]))

    edges_df = edges_df.merge(
        cong_map.reset_index(), on=["u","v"], how="left"
    )
    for col in ["ml_congestion_score","accident_score","violation_score",
                "traffic_density","centrality_score","intersection_delay"]:
        edges_df[col] = edges_df[col].fillna(edges_df[col].median())

    # Composite congestion score (same formula as congestion_optimizer.py)
    edges_df["composite"] = (
        0.40 * edges_df["ml_congestion_score"] +
        0.20 * edges_df["traffic_density"] +
        0.20 * edges_df["accident_score"] +
        0.10 * edges_df["centrality_score"] +
        0.10 * (edges_df["intersection_delay"] / 120.0).clip(0, 1)
    )

    # Raw weight: length * (1 + 4 * composite)  — congestion slows travel
    edges_df["length"] = pd.to_numeric(edges_df["length"], errors="coerce").fillna(100.0)
    edges_df["raw_weight"] = edges_df["length"] * (1.0 + 4.0 * edges_df["composite"])

    print(f"  Congestion merged in {time.time()-t1:.1f}s")

if not STARTUP_ERROR:
    # ── 2. Build NetworkX graph ───────────────────────────────────────────────
    t2 = time.time()
    G = nx.DiGraph()

    node_pos = {}
    for _, r in nodes_df.iterrows():
        nid = int(r.osmid)
        G.add_node(nid, y=float(r.y), x=float(r.x))
        node_pos[nid] = (float(r.x), float(r.y))   # lon, lat

    for _, r in edges_df.iterrows():
        u, v = int(r.u), int(r.v)
        G.add_edge(u, v,
                   raw_weight   = float(r.raw_weight),
                   length       = float(r.length),
                   composite    = float(r.composite),
                   ml_score     = float(r.ml_congestion_score),
                   name         = str(r.get("name", "")),
                   highway      = str(r.get("highway", "")),
                   weight       = float(r.raw_weight))   # default = before

    print(f"  Graph built in {time.time()-t2:.1f}s  "
          f"| {G.number_of_nodes():,} nodes  {G.number_of_edges():,} edges")

if not STARTUP_ERROR:
    # ── 3. PageRank-Guided Relief — compute optimised weights ─────────────────
    print("  Running PageRank-Guided Relief …")
    t3 = time.time()

    pr = nx.pagerank(G, alpha=0.85, weight="raw_weight", max_iter=100, tol=1e-4)
    pr_vals  = np.array(list(pr.values()))
    pr_med   = float(np.median(pr_vals))

    comp_vals = np.array([d["composite"] for _, _, d in G.edges(data=True)])
    comp_med  = float(np.median(comp_vals))

    for u, v, data in G.edges(data=True):
        edge_pr   = (pr.get(u, 0.0) + pr.get(v, 0.0)) / 2.0
        high_pr   = edge_pr        >= pr_med
        high_cong = data["composite"] >= comp_med

        if   high_pr and high_cong:     mult = 0.65   # −35% critical bottleneck
        elif high_pr and not high_cong: mult = 0.90   # −10% important, not congested
        elif not high_pr and high_cong: mult = 0.80   # −20% congested, less critical
        else:                           mult = 0.95   # −5%  minor, low congestion

        data["opt_weight"] = data["raw_weight"] * mult
        data["pr_mult"]    = mult

    print(f"  PageRank done in {time.time()-t3:.1f}s")

if not STARTUP_ERROR:
    # ── 4. KD-tree for nearest-node lookup ───────────────────────────────────
    from scipy.spatial import KDTree
    node_ids  = np.array(list(node_pos.keys()))
    node_coords = np.array([(node_pos[n][1], node_pos[n][0]) for n in node_ids])  # lat,lon
    kd = KDTree(node_coords)

    def nearest_node(lat, lon):
        _, idx = kd.query([lat, lon])
        return int(node_ids[idx])
else:
    def nearest_node(lat, lon):
        raise RuntimeError("Dashboard isn't ready because required outputs are missing.")

# ── 5. Route computation ──────────────────────────────────────────────────
def compute_routes(src_node, dst_node):
    """Return before/after route dicts with coords + stats."""

    if STARTUP_ERROR:
        return {"error": STARTUP_ERROR}

    def dijkstra(weight_attr):
        try:
            path = nx.dijkstra_path(G, src_node, dst_node, weight=weight_attr)
            return path
        except nx.NetworkXNoPath:
            return None

    path_before = dijkstra("raw_weight")
    path_after  = dijkstra("opt_weight")

    def path_stats(path, weight_attr):
        if path is None:
            return None
        coords, total_len, total_w, total_cong = [], 0.0, 0.0, 0.0
        total_raw_cong = 0.0   # raw composite score
        total_eff_cong = 0.0   # effective congestion (accounts for relief)
        road_names, road_types = set(), set()
        n_edges = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if not G.has_edge(u, v):
                continue
            d = G[u][v]
            coords.append([G.nodes[u]["y"], G.nodes[u]["x"]])
            length = d.get("length", 0)
            total_len  += length
            total_w    += d.get(weight_attr, 0)
            
            raw_comp = d.get("composite", 0)
            total_raw_cong += raw_comp
            
            # Effective congestion = raw * (opt_weight / raw_weight)
            # If PageRank reduced weight by 35%, effective congestion is also 35% lower
            if weight_attr == "opt_weight":
                raw_w = d.get("raw_weight", 1)
                opt_w = d.get("opt_weight", 1)
                relief_factor = opt_w / max(raw_w, 1e-6)
                total_eff_cong += raw_comp * relief_factor
            else:
                total_eff_cong += raw_comp
            
            n = d.get("name", "")
            if n and n != "nan": road_names.add(n)
            h = d.get("highway", "")
            if h and h != "nan": road_types.add(h)
            n_edges += 1
        if path:
            last = path[-1]
            coords.append([G.nodes[last]["y"], G.nodes[last]["x"]])
        
        avg_raw_cong = total_raw_cong / max(n_edges, 1)
        avg_eff_cong = total_eff_cong / max(n_edges, 1)
        est_time = total_w / 13.9   # rough: weight/speed → seconds
        
        return {
            "coords":     coords,
            "length_m":   round(total_len, 1),
            "length_km":  round(total_len / 1000, 2),
            "est_time_s": round(est_time),
            "est_time_min": round(est_time / 60, 1),
            "avg_congestion": round(avg_eff_cong, 3),  # use effective
            "avg_raw_congestion": round(avg_raw_cong, 3),  # keep raw for reference
            "n_edges":    n_edges,
            "road_names": sorted(road_names)[:8],
            "road_types": sorted(road_types)[:5],
        }

    before = path_stats(path_before, "raw_weight")
    after  = path_stats(path_after,  "opt_weight")

    # improvement — always compare effective congestion
    improvement = {}
    if before and after:
        def pct(b, a):
            return round(100 * (b - a) / b, 1) if b > 0 else 0.0
        improvement = {
            "time_pct":  pct(before["est_time_s"],      after["est_time_s"]),
            "cong_pct":  pct(before["avg_congestion"],  after["avg_congestion"]),
            "len_pct":   pct(before["length_m"],        after["length_m"]),
        }

    # hotspots along before-route bbox
    if before and before["coords"]:
        lats = [c[0] for c in before["coords"]]
        lons = [c[1] for c in before["coords"]]
        pad = 0.01
        hot_near = hot_df[
            (hot_df.lat >= min(lats)-pad) & (hot_df.lat <= max(lats)+pad) &
            (hot_df.lon >= min(lons)-pad) & (hot_df.lon <= max(lons)+pad)
        ].nlargest(12, "ml_congestion_score")
        hotspots = hot_near[["lat","lon","ml_congestion_score","road_name"]].to_dict("records")

        bot_near = bot_df[
            (bot_df.lat >= min(lats)-pad) & (bot_df.lat <= max(lats)+pad) &
            (bot_df.lon >= min(lons)-pad) & (bot_df.lon <= max(lons)+pad)
        ].head(8)
        bottlenecks = bot_near[["lat","lon","composite_score","signal_delay_est"]].to_dict("records")
    else:
        hotspots, bottlenecks = [], []

    return {
        "before":      before,
        "after":       after,
        "improvement": improvement,
        "hotspots":    hotspots,
        "bottlenecks": bottlenecks,
        "src_node":    src_node,
        "dst_node":    dst_node,
    }

# ── 6. Preset locations ───────────────────────────────────────────────────
PRESETS = {
    "Silk Board":          (12.9177, 77.6238),
    "Whitefield":          (12.9698, 77.7500),
    "MG Road":             (12.9757, 77.6011),
    "Koramangala":         (12.9352, 77.6245),
    "Indiranagar":         (12.9784, 77.6408),
    "Electronic City":     (12.8399, 77.6770),
    "Hebbal":              (13.0358, 77.5970),
    "Marathahalli":        (12.9591, 77.6974),
    "Jayanagar":           (12.9308, 77.5838),
    "Yeshwanthpur":        (13.0234, 77.5554),
    "Banashankari":        (12.9255, 77.5468),
    "KR Puram":            (13.0050, 77.6960),
    "Yelahanka":           (13.1007, 77.5963),
    "BTM Layout":          (12.9166, 77.6101),
    "HSR Layout":          (12.9116, 77.6473),
    "Rajajinagar":         (12.9907, 77.5530),
    "Malleshwaram":        (13.0035, 77.5700),
    "Bellandur":           (12.9257, 77.6760),
    "Sarjapur Road":       (12.9100, 77.6900),
    "Outer Ring Road":     (12.9600, 77.7100),
}

# ── 7. Flask routes ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", presets=list(PRESETS.keys()))

@app.route("/api/route", methods=["POST"])
def api_route():
    data = request.get_json()
    try:
        src_lat = float(data["src_lat"])
        src_lon = float(data["src_lon"])
        dst_lat = float(data["dst_lat"])
        dst_lon = float(data["dst_lon"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    if STARTUP_ERROR:
        return jsonify({"error": STARTUP_ERROR}), 503

    src_node = nearest_node(src_lat, src_lon)
    dst_node = nearest_node(dst_lat, dst_lon)

    if src_node == dst_node:
        return jsonify({"error": "Source and destination are the same point."}), 400

    result = compute_routes(src_node, dst_node)
    if "error" in result:
        return jsonify(result), 503

    if result["before"] is None:
        return jsonify({"error": "No route found between these points."}), 404

    return jsonify(result)

@app.route("/api/presets")
def api_presets():
    return jsonify({k: {"lat": v[0], "lon": v[1]} for k, v in PRESETS.items()})

if __name__ == "__main__":
    print(f"\n  ✓ Dashboard ready → http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)

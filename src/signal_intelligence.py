"""
signal_intelligence.py — Step 4: Synthetic Signal & Intersection Delay Estimation.

⚠ SYNTHETIC / ESTIMATED DATA — CLEARLY LABELLED
────────────────────────────────────────────────
No real traffic-signal dataset is available for Bangalore.
All signal delays in this module are ESTIMATED from graph topology:

  1. Node degree  → proxy for intersection complexity
     (degree 3 = T-junction, 4 = 4-way, ≥5 = complex)

  2. Betweenness centrality → proxy for traffic load at a node
     (high centrality = more routes pass through → longer effective delay)

  3. Road hierarchy → arterials carry more traffic → higher delay multiplier

  4. Intersection density → local density of intersections within 500 m
     (dense areas like Silk Board / Majestic get higher delays)

These are SYNTHETIC ESTIMATES based on urban traffic engineering heuristics,
NOT measured signal timings. They are clearly labelled as such in all outputs.

REFERENCES (methodology basis):
  • Highway Capacity Manual (HCM) 6th Ed. — intersection delay models
  • Akçelik (1981) — delay formula for signalized intersections
  • OSMnx documentation — graph topology features

OUTPUT: bangalore_signalized.graphml
  New node attributes : signal_delay (seconds, estimated)
  New edge attributes : intersection_delay (seconds, estimated)
                        centrality_score (normalized betweenness)
                        weight (updated: += intersection_delay)
"""

from __future__ import annotations

import heapq

import numpy as np
import osmnx as ox

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from config import (
    GRAPH_WEIGHTED_PATH, GRAPH_SIGNAL_PATH,
    SIGNAL_DELAY_BY_DEGREE, SIGNAL_DELAY_DEFAULT,
    CENTRALITY_SAMPLE_SOURCES, CENTRALITY_RANDOM_SEED,
    GRAPHS_DIR,
)
from utils import ensure_output_dirs, road_type_rank, safe_float


# ─────────────────────────────────────────────────────────────────────────────
# LOAD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def load_graph():
    print("[Step 4] Loading weighted graph ...")
    G = ox.load_graphml(str(GRAPH_WEIGHTED_PATH))
    print(f"  → {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# DETECT INTERSECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def detect_intersections(G) -> list:
    """
    Return nodes with degree ≥ 3 (T-junctions and above).
    These are treated as signalized or controlled intersections.

    ⚠ SYNTHETIC ASSUMPTION: All degree-≥3 nodes are assumed to have
    traffic control. In reality, many may be uncontrolled.
    """
    intersections = [n for n in G.nodes() if G.degree(n) >= 3]
    print(f"  Detected {len(intersections):,} intersections (degree ≥ 3)")
    return intersections


# ─────────────────────────────────────────────────────────────────────────────
# BASE SIGNAL DELAY FROM DEGREE
# ─────────────────────────────────────────────────────────────────────────────

def assign_degree_based_delay(G, intersections: list) -> object:
    """
    Assign base signal delay (seconds) to each intersection node based on
    its degree (number of connected road segments).

    ⚠ SYNTHETIC: Delay values are heuristic estimates from HCM guidelines.
    """
    print("  Assigning degree-based signal delays (SYNTHETIC ESTIMATE) ...")
    for node in intersections:
        degree = G.degree(node)
        if degree >= 5:
            delay = SIGNAL_DELAY_BY_DEGREE.get(5, 50)
        else:
            delay = SIGNAL_DELAY_BY_DEGREE.get(degree, SIGNAL_DELAY_DEFAULT)
        G.nodes[node]["signal_delay"] = float(delay)
        G.nodes[node]["signal_estimated"] = True   # flag as synthetic
    return G


# ─────────────────────────────────────────────────────────────────────────────
# ROAD HIERARCHY MULTIPLIER
# ─────────────────────────────────────────────────────────────────────────────

def apply_road_hierarchy_multiplier(G) -> object:
    """
    Multiply signal delay by a road-hierarchy factor.
    Primary/trunk roads carry more traffic → longer effective delay.

    ⚠ SYNTHETIC: Multipliers are heuristic.
    """
    print("  Applying road-hierarchy delay multipliers (SYNTHETIC) ...")
    hierarchy_multiplier = {7: 1.8, 6: 1.6, 5: 1.4, 4: 1.2, 3: 1.0, 2: 0.8, 1: 0.6}

    for node in G.nodes():
        if "signal_delay" not in G.nodes[node]:
            continue
        # Find the maximum road rank among incident edges
        max_rank = 1
        for nbr in G.neighbors(node):
            for _, edata in G[node][nbr].items():
                rank = road_type_rank(str(edata.get("highway", "unclassified")))
                max_rank = max(max_rank, rank)
        mult = hierarchy_multiplier.get(max_rank, 1.0)
        G.nodes[node]["signal_delay"] = G.nodes[node]["signal_delay"] * mult

    return G


# ─────────────────────────────────────────────────────────────────────────────
# BETWEENNESS CENTRALITY (SAMPLED)
# ─────────────────────────────────────────────────────────────────────────────

def _brandes_sampled(G, sources: list, total_nodes: int) -> dict:
    """
    Sampled Brandes betweenness centrality (weighted by 'weight' attribute).
    Returns {node: centrality_score}.

    Uses a subset of source nodes for tractability on large graphs.
    Scales the result to approximate full-graph centrality.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    betweenness = dict.fromkeys(nodes, 0.0)

    iterator = tqdm(sources, desc="  Betweenness centrality", unit="src") if _TQDM else sources

    for s in iterator:
        S, P, sigma, dist = [], {v: [] for v in nodes}, dict.fromkeys(nodes, 0.0), {}
        sigma[s] = 1.0
        dist[s] = 0.0
        Q = [(0.0, s)]

        while Q:
            d_v, v = heapq.heappop(Q)
            if d_v != dist.get(v):
                continue
            S.append(v)
            for w, edata in G[v].items():
                # Handle MultiDiGraph edge data
                if isinstance(edata, dict) and any(isinstance(v2, dict) for v2 in edata.values()):
                    w_val = min(
                        safe_float(attrs.get("weight", 1.0), 1.0)
                        for attrs in edata.values()
                        if isinstance(attrs, dict)
                    )
                else:
                    w_val = safe_float(edata.get("weight", 1.0), 1.0)
                d_w = d_v + max(w_val, 1e-9)
                if w not in dist or d_w < dist[w]:
                    dist[w] = d_w
                    heapq.heappush(Q, (d_w, w))
                    sigma[w] = sigma[v]
                    P[w] = [v]
                elif abs(d_w - dist[w]) < 1e-9:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        delta = dict.fromkeys(nodes, 0.0)
        while S:
            w = S.pop()
            for v in P[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # Scale to approximate full-graph result
    if len(sources) < total_nodes:
        scale_up = total_nodes / float(len(sources))
        for v in betweenness:
            betweenness[v] *= scale_up

    # Normalize (NetworkX convention)
    if n > 2:
        scale = 1.0 / ((n - 1) * (n - 2))
        if not G.is_directed():
            scale *= 2.0
        for v in betweenness:
            betweenness[v] *= scale

    return betweenness


def apply_centrality_delay(G, intersections: list) -> object:
    """
    Compute sampled betweenness centrality and add an extra delay component
    proportional to centrality (high-centrality nodes = more traffic load).

    ⚠ SYNTHETIC: The centrality-to-delay mapping is a heuristic.
    """
    print(f"  Computing betweenness centrality "
          f"(sampling {CENTRALITY_SAMPLE_SOURCES} sources) ...")

    rng = np.random.default_rng(CENTRALITY_RANDOM_SEED)
    candidates = intersections if intersections else list(G.nodes())
    k = min(CENTRALITY_SAMPLE_SOURCES, len(candidates))
    sampled = rng.choice(np.array(candidates, dtype=object), size=k, replace=False).tolist()

    centrality = _brandes_sampled(G, sampled, total_nodes=len(candidates))

    # Normalize centrality to [0, 1]
    vals = np.array(list(centrality.values()))
    c_min, c_max = vals.min(), vals.max()
    if c_max - c_min > 1e-12:
        norm_centrality = {
            n: (centrality[n] - c_min) / (c_max - c_min)
            for n in centrality
        }
    else:
        norm_centrality = {n: 0.0 for n in centrality}

    # Store centrality and add extra delay
    for node, score in norm_centrality.items():
        G.nodes[node]["centrality_score"] = float(score)
        extra_delay = score * 60.0   # up to 60 extra seconds at max centrality
        current = safe_float(G.nodes[node].get("signal_delay", 0.0))
        G.nodes[node]["signal_delay"] = current + extra_delay

    return G


# ─────────────────────────────────────────────────────────────────────────────
# PROPAGATE NODE DELAY → EDGES
# ─────────────────────────────────────────────────────────────────────────────

def propagate_delay_to_edges(G) -> object:
    """
    Average the signal delays of the two endpoint nodes and assign to each edge.
    Also update the edge weight to include intersection delay.

    Edge attributes added:
      intersection_delay  — average of endpoint signal delays (seconds, SYNTHETIC)
      centrality_score    — average of endpoint centrality scores
      weight              — updated: original_weight + intersection_delay
    """
    print("  Propagating node delays to edges ...")
    for u, v, k, data in G.edges(keys=True, data=True):
        delay_u = safe_float(G.nodes[u].get("signal_delay", 0.0))
        delay_v = safe_float(G.nodes[v].get("signal_delay", 0.0))
        avg_delay = (delay_u + delay_v) / 2.0

        cent_u = safe_float(G.nodes[u].get("centrality_score", 0.0))
        cent_v = safe_float(G.nodes[v].get("centrality_score", 0.0))
        avg_cent = (cent_u + cent_v) / 2.0

        data["intersection_delay"] = avg_delay
        data["centrality_score"]   = avg_cent
        data["signal_delay"]       = avg_delay   # alias for ML step

        current_weight = safe_float(data.get("weight", 1.0), 1.0)
        data["weight"] = current_weight + avg_delay

    return G


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_graph(G):
    ensure_output_dirs(GRAPHS_DIR)
    ox.save_graphml(G, str(GRAPH_SIGNAL_PATH))
    print(f"  Signalized graph saved → {GRAPH_SIGNAL_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    G = load_graph()
    intersections = detect_intersections(G)
    G = assign_degree_based_delay(G, intersections)
    G = apply_road_hierarchy_multiplier(G)
    G = apply_centrality_delay(G, intersections)
    G = propagate_delay_to_edges(G)
    save_graph(G)

    # Summary statistics
    delays = [
        safe_float(G.nodes[n].get("signal_delay", 0.0))
        for n in G.nodes()
        if G.nodes[n].get("signal_delay", 0.0) > 0
    ]
    if delays:
        print(f"\n  Signal delay stats (SYNTHETIC ESTIMATES):")
        print(f"    Nodes with delay : {len(delays):,}")
        print(f"    Mean delay       : {np.mean(delays):.1f} s")
        print(f"    Max delay        : {np.max(delays):.1f} s")

    print("\n✓ Step 4 complete — synthetic signal intelligence applied.")


if __name__ == "__main__":
    main()

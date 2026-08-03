"""
build_graph.py — Step 1: Download and save the Bangalore road network.

DATA SOURCE: OpenStreetMap via OSMnx (real data).
OUTPUT    : bangalore_graph.graphml, nodes.csv, edges.csv, road_network.html
"""

import osmnx as ox
import geopandas as gpd
import folium
from shapely.ops import unary_union

from config import (
    PLACE_NAME, NETWORK_TYPE,
    GRAPH_PATH, NODES_PATH, EDGES_PATH, MAP_ROAD_NETWORK_PATH,
    GRAPHS_DIR, MAPS_DIR,
)
from utils import ensure_output_dirs


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_graph():
    print(f"[Step 1] Downloading OSM road network for '{PLACE_NAME}' ...")
    G = ox.graph_from_place(PLACE_NAME, network_type=NETWORK_TYPE)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"  → {n_nodes:,} nodes | {n_edges:,} edges")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# SAVE GRAPHML
# ─────────────────────────────────────────────────────────────────────────────

def save_graph(G):
    ox.save_graphml(G, str(GRAPH_PATH))
    print(f"  Graph saved → {GRAPH_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT NODE / EDGE CSVs
# ─────────────────────────────────────────────────────────────────────────────

def export_csvs(G):
    nodes, edges = ox.graph_to_gdfs(G)
    nodes.to_csv(NODES_PATH)
    edges.to_csv(EDGES_PATH)
    print(f"  Nodes CSV → {NODES_PATH}")
    print(f"  Edges CSV → {EDGES_PATH}")
    return nodes, edges


# ─────────────────────────────────────────────────────────────────────────────
# FOLIUM MAP
# ─────────────────────────────────────────────────────────────────────────────

def visualize_graph(G):
    print("  Building Folium road-network map ...")
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    # Simplify geometry for a lighter HTML file
    try:
        edges_proj = ox.projection.project_gdf(edges)
        edges_proj["geometry"] = edges_proj["geometry"].simplify(
            tolerance=30, preserve_topology=True
        )
        edges = edges_proj.to_crs(epsg=4326)
    except Exception:
        pass

    ys = [d.get("y") for _, d in G.nodes(data=True) if d.get("y") is not None]
    xs = [d.get("x") for _, d in G.nodes(data=True) if d.get("x") is not None]
    center = [sum(ys) / len(ys), sum(xs) / len(xs)] if ys else [12.97, 77.59]

    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    merged = unary_union(edges["geometry"].dropna().values)
    merged_gdf = gpd.GeoDataFrame(geometry=[merged], crs="EPSG:4326")
    folium.GeoJson(
        merged_gdf.to_json(),
        style_function=lambda _: {"color": "#2563EB", "weight": 1, "opacity": 0.6},
        name="Road Network",
    ).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(MAP_ROAD_NETWORK_PATH))
    print(f"  Map saved → {MAP_ROAD_NETWORK_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ensure_output_dirs(GRAPHS_DIR, MAPS_DIR)
    G = download_graph()
    save_graph(G)
    export_csvs(G)
    visualize_graph(G)
    print("\n✓ Step 1 complete — road network downloaded and saved.")


if __name__ == "__main__":
    main()

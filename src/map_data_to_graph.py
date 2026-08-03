"""
map_data_to_graph.py — Step 2: Enrich the OSM graph with spatial datasets.

DATA SOURCES (all real):
  • KML  : Bangalore ward/polling-booth boundaries with population & trip data
  • CSVs : Any dataset in data/ that contains lat/lon columns

DATA TRANSPARENCY:
  • No synthetic data is introduced in this step.
  • Datasets without coordinates are attached as graph-level metadata only.
  • All edge attributes added here are prefixed so their origin is traceable.

OUTPUT: bangalore_graph_enriched.graphml
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import osmnx as ox
import pandas as pd

from config import (
    GRAPH_PATH, GRAPH_ENRICHED_PATH,
    DATA_DIR, GRAPHS_DIR,
    KML_PATH,
)
from utils import (
    clean_columns, coerce_numeric, create_point_gdf,
    detect_lat_lon_columns, ensure_output_dirs, load_kml,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def load_graph():
    print("[Step 2] Loading base graph ...")
    G = ox.load_graphml(str(GRAPH_PATH))
    print(f"  → {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# MAP GEODATAFRAME → NEAREST EDGES
# ─────────────────────────────────────────────────────────────────────────────

def map_geodataframe_to_edges(
    G,
    gdf: gpd.GeoDataFrame,
    prefix: str,
    numeric_sum_cols: list[str] | None = None,
) -> object:
    """
    Map each geometry in *gdf* to its nearest road edge and accumulate counts
    and optional numeric sums.

    Uses vectorized nearest_edges for performance (batch all points at once).

    Adds edge attributes:
      {prefix}_count
      {prefix}_{col}_sum   for each column in numeric_sum_cols
    """
    if gdf.empty:
        return G

    # Reproject to WGS-84 if needed
    if gdf.crs is not None and str(gdf.crs).upper() not in {"EPSG:4326", "WGS84"}:
        try:
            gdf = gdf.to_crs(epsg=4326)
        except Exception:
            pass

    numeric_sum_cols = numeric_sum_cols or []
    available_sum_cols = [c for c in numeric_sum_cols if c in gdf.columns]
    if available_sum_cols:
        gdf = gdf.copy()
        for c in available_sum_cols:
            gdf[c] = coerce_numeric(gdf[c])

    count_key = f"{prefix}_count"

    # Extract representative points for all geometries at once
    lons, lats, valid_idx = [], [], []
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or getattr(geom, "is_empty", True):
            continue
        if geom.geom_type == "Point":
            lons.append(geom.x)
            lats.append(geom.y)
        else:
            try:
                rp = geom.representative_point()
                lons.append(rp.x)
                lats.append(rp.y)
            except Exception:
                continue
        valid_idx.append(i)

    if not lons:
        return G

    # Vectorized nearest-edge lookup (much faster than row-by-row)
    try:
        import numpy as np
        ne = ox.distance.nearest_edges(G, np.array(lons), np.array(lats))
        # ne is a list of (u, v, key) tuples
        for idx, (u, v, key) in zip(valid_idx, ne):
            try:
                edge = G[u][v][key]
            except (KeyError, TypeError):
                continue
            edge[count_key] = int(edge.get(count_key, 0)) + 1
            row = gdf.loc[idx]
            for c in available_sum_cols:
                val = row.get(c)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                sum_key = f"{prefix}_{c}_sum"
                edge[sum_key] = float(edge.get(sum_key, 0.0)) + float(val)
    except Exception as e:
        print(f"  Vectorized nearest_edges failed ({e}), falling back to row-by-row ...")
        for i, (lon, lat) in enumerate(zip(lons, lats)):
            try:
                u, v, key = ox.distance.nearest_edges(G, lon, lat)
                edge = G[u][v][key]
                edge[count_key] = int(edge.get(count_key, 0)) + 1
                row = gdf.loc[valid_idx[i]]
                for c in available_sum_cols:
                    val = row.get(c)
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        continue
                    sum_key = f"{prefix}_{c}_sum"
                    edge[sum_key] = float(edge.get(sum_key, 0.0)) + float(val)
            except Exception:
                continue

    return G


# ─────────────────────────────────────────────────────────────────────────────
# MAP KML
# ─────────────────────────────────────────────────────────────────────────────

def map_kml(G):
    """
    Map the ward/polling-booth KML to edges.
    Numeric fields: people, employees, trips, access_percentage
    These represent real ward-level demand data from the KML.
    """
    if not KML_PATH.exists():
        print(f"  KML not found at {KML_PATH} — skipping.")
        return G

    print(f"  Mapping KML: {KML_PATH.name} ...")
    try:
        gdf = load_kml(KML_PATH)
    except Exception as e:
        print(f"  Failed to load KML: {e}")
        return G

    G = map_geodataframe_to_edges(
        G, gdf,
        prefix="kml_ward",
        numeric_sum_cols=["people", "employees", "trips", "access_percentage"],
    )
    print(f"  KML mapped → {G.number_of_edges():,} edges enriched")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# MAP GEOCODED CSVs
# ─────────────────────────────────────────────────────────────────────────────

# Files that are graph-topology outputs, not raw data
_SKIP_FILES = {"edges.csv", "nodes.csv", "ml_dataset.csv"}


def map_geocoded_csvs(G):
    """
    Scan every CSV in data/ for lat/lon columns.
    If found, map each row to its nearest edge and increment a count attribute.
    Only real coordinate data is used here — no synthetic positions.
    """
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    for csv_path in csv_files:
        if csv_path.name in _SKIP_FILES:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        df = clean_columns(df)
        lat_col, lon_col = detect_lat_lon_columns(df)
        if not lat_col or not lon_col:
            continue

        print(f"  Mapping geocoded CSV: {csv_path.name} ({lat_col}/{lon_col}) ...")
        gdf = create_point_gdf(df, lat_col=lat_col, lon_col=lon_col)
        prefix = f"csv_{csv_path.stem[:20]}"
        G = map_geodataframe_to_edges(G, gdf, prefix=prefix)

    return G


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_graph(G):
    ensure_output_dirs(GRAPHS_DIR)
    ox.save_graphml(G, str(GRAPH_ENRICHED_PATH))
    print(f"  Enriched graph saved → {GRAPH_ENRICHED_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    G = load_graph()
    G = map_kml(G)
    G = map_geocoded_csvs(G)
    save_graph(G)
    print("\n✓ Step 2 complete — graph enriched with spatial datasets.")


if __name__ == "__main__":
    main()

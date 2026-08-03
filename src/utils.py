"""
utils.py — Shared helper utilities.

DATA TRANSPARENCY NOTE
----------------------
Functions in this module are pure data-processing helpers.
No synthetic data is generated here; all assumptions are documented
at the call site in the pipeline scripts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


# ─────────────────────────────────────────────────────────────────────────────
# DATAFRAME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip whitespace and embedded newlines."""
    df = df.copy()
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    return df


def coerce_numeric(s: pd.Series) -> pd.Series:
    """Convert a Series to numeric, coercing errors to NaN."""
    return pd.to_numeric(s, errors="coerce")


def normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize a numeric Series to [0, 1]."""
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float, returning *default* on failure."""
    try:
        f = float(value)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_lat_lon_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Best-effort detection of latitude / longitude columns.

    Returns (lat_col, lon_col) or (None, None) if not found.
    """
    cols = list(df.columns)
    lowered = {c: str(c).strip().lower() for c in cols}

    blacklist = {"year", "month", "day", "hour", "id", "osmid", "type of offences"}

    lat_candidates: list[str] = []
    lon_candidates: list[str] = []

    for c in cols:
        lc = lowered[c]
        if lc in blacklist:
            continue
        if lc in {"lat", "latitude"} or "latitude" in lc:
            lat_candidates.append(c)
        if lc in {"lon", "lng", "long", "longitude"} or "longitude" in lc:
            lon_candidates.append(c)
        if lc == "x":
            lon_candidates.append(c)
        if lc == "y":
            lat_candidates.append(c)

    def _pick_numeric(cands: list[str]) -> str | None:
        for c in cands:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return c
        return None

    return _pick_numeric(lat_candidates), _pick_numeric(lon_candidates)


# ─────────────────────────────────────────────────────────────────────────────
# I/O HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    return clean_columns(df)


def load_kml(path: str | Path) -> gpd.GeoDataFrame:
    """Load a KML file using geopandas / fiona."""
    import fiona
    fiona.drvsupport.supported_drivers["KML"] = "rw"
    fiona.drvsupport.supported_drivers["LIBKML"] = "rw"
    return gpd.read_file(str(path), driver="KML")


def create_point_gdf(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame of Points from lat/lon columns."""
    geometry = [
        Point(lon, lat)
        for lon, lat in zip(
            pd.to_numeric(df[lon_col], errors="coerce"),
            pd.to_numeric(df[lat_col], errors="coerce"),
        )
    ]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ATTRIBUTE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ensure_output_dirs(*paths: Path) -> None:
    """Create all required output directories."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def get_edge_attr(data: dict, key: str, default: float = 0.0) -> float:
    """Safely retrieve a numeric edge attribute."""
    return safe_float(data.get(key, default), default)


def road_type_rank(highway: str) -> int:
    """
    Return a numeric rank for road hierarchy (higher = more important).
    Used in signal delay estimation and emergency routing.
    """
    hierarchy = {
        "motorway": 7, "motorway_link": 6,
        "trunk": 6,    "trunk_link": 5,
        "primary": 5,  "primary_link": 4,
        "secondary": 4, "secondary_link": 3,
        "tertiary": 3,  "tertiary_link": 2,
        "residential": 2, "living_street": 1,
        "unclassified": 1, "service": 1,
    }
    hw = str(highway).lower().strip()
    # Handle list-type highway values stored as strings
    if hw.startswith("["):
        hw = hw.strip("[]'\" ").split(",")[0].strip().strip("'\"")
    return hierarchy.get(hw, 1)


def station_name_variants(name: str) -> list[str]:
    """
    Generate lookup variants for a police station / area name.
    Helps fuzzy-match dataset station names to OSM road names.
    """
    name = str(name).strip().lower()
    variants = [name]
    # Remove common suffixes
    for suffix in [" traffic police station", " police station", " ps", " tp"]:
        if name.endswith(suffix):
            variants.append(name[: -len(suffix)].strip())
    return variants

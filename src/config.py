"""
config.py — Central configuration for the Bangalore Traffic Congestion project.

All paths, constants, and tunable hyper-parameters live here so every other
module stays clean and easy to modify.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "output"
SRC_DIR      = PROJECT_ROOT / "src"

# Sub-directories (created on demand by each script)
GRAPHS_DIR = OUTPUT_DIR / "graphs"
MAPS_DIR   = OUTPUT_DIR / "maps"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

# ─────────────────────────────────────────────────────────────────────────────
# SUMO INTEGRATION (simulation + visualization layer)
# ─────────────────────────────────────────────────────────────────────────────
# These are intentionally optional. Your core pipeline doesn't depend on SUMO.
SUMO_DIR = PROJECT_ROOT / "sumo"
SUMO_NET_DIR = SUMO_DIR / "net"
SUMO_DEMAND_DIR = SUMO_DIR / "demand"
SUMO_CFG_DIR = SUMO_DIR / "cfg"
SUMO_OUTPUT_DIR = SUMO_DIR / "output"
SUMO_LOG_DIR = SUMO_DIR / "logs"
SUMO_VIEW_DIR = SUMO_DIR / "view"

# Default SUMO artifacts (generated)
SUMO_NET_PATH = SUMO_NET_DIR / "bangalore.net.xml"
SUMO_EDGE_MAP_PATH = SUMO_NET_DIR / "edge_mapping.csv"  # NetworkX edge -> SUMO edge

SUMO_TRIPS_XML = SUMO_DEMAND_DIR / "trips.xml"
SUMO_ROUTES_BEFORE_XML = SUMO_DEMAND_DIR / "routes_before.rou.xml"
SUMO_ROUTES_AFTER_XML = SUMO_DEMAND_DIR / "routes_after.rou.xml"

SUMO_BEFORE_CFG = SUMO_CFG_DIR / "before_simulation.sumocfg"
SUMO_AFTER_CFG = SUMO_CFG_DIR / "after_simulation.sumocfg"

SUMO_VIEW_SETTINGS_XML = SUMO_VIEW_DIR / "congestion_colors.xml"

# Pipeline toggle
ENABLE_SUMO_SIM = False

# ─────────────────────────────────────────────────────────────────────────────
# OSM DOWNLOAD SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
PLACE_NAME   = "Bangalore, India"
NETWORK_TYPE = "drive"

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH FILE PATHS  (pipeline stages)
# ─────────────────────────────────────────────────────────────────────────────
GRAPH_PATH          = GRAPHS_DIR / "bangalore_graph.graphml"
GRAPH_ENRICHED_PATH = GRAPHS_DIR / "bangalore_graph_enriched.graphml"
GRAPH_WEIGHTED_PATH = GRAPHS_DIR / "bangalore_weighted.graphml"
GRAPH_SIGNAL_PATH   = GRAPHS_DIR / "bangalore_signalized.graphml"
GRAPH_ML_PATH       = GRAPHS_DIR / "bangalore_ml.graphml"

NODES_PATH = GRAPHS_DIR / "nodes.csv"
EDGES_PATH = GRAPHS_DIR / "edges.csv"

# ─────────────────────────────────────────────────────────────────────────────
# DATA FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────
# Real datasets (clearly labelled)
ACCIDENTS_2025_PATH   = DATA_DIR / "btp_2025_station_wise.csv"          # BTP station-wise 2025
ACCIDENTS_2024_PATH   = DATA_DIR / "74e645e3-85d2-4d81-a133-4f346f87fdd6.csv"
ACCIDENTS_2023_PATH   = DATA_DIR / "abc5af52-08a7-4435-8ba1-12b99f62ee28.csv"
ACCIDENTS_MULTI_PATH  = DATA_DIR / "aef42379-f1f7-4a3b-94f5-5f344e7120f2.csv"
ACCIDENTS_ZONE_PATH   = DATA_DIR / "492d3dc6-ffc3-4b0e-b7d9-176d0ef7f1ec.csv"
ACCIDENTS_ANNUAL_PATH = DATA_DIR / "a583a07c-731b-4e8d-b0cb-d06e76ccc00c.csv"

VIOLATIONS_PATH       = DATA_DIR / "09d692fa-5b74-44a2-89de-5f0940a43117.csv"
DRINK_DRIVE_PATH      = DATA_DIR / "f648e986-e04b-4865-9dc6-8962df21a7c6.csv"
ENFORCEMENT_PATH      = DATA_DIR / "169b0ea2-4092-4d78-b450-ae39ad2d8197.csv"
ROAD_DESC_PATH        = DATA_DIR / "53c68b2c-fc53-471e-9c61-dd093f7a4111.csv"
BANGALORE_TRAFFIC_PATH = DATA_DIR / "Banglore_traffic_Dataset.csv"       # richest local dataset

# US Metro traffic (used for temporal pattern reference only — clearly labelled)
METRO_TRAFFIC_PATH    = DATA_DIR / "Metro_Interstate_Traffic_Volume.csv"

KML_PATH = DATA_DIR / "0602631e-5ee4-48ae-bcdd-3ef99f02c428.kml"

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────────────────────────────────────
MAP_ROAD_NETWORK_PATH  = MAPS_DIR / "road_network.html"
MAP_OPTIMIZED_PATH     = MAPS_DIR / "optimized_route.html"
MAP_HOTSPOTS_PATH      = MAPS_DIR / "hotspot_map.html"
MAP_INTERVENTION_PATH  = MAPS_DIR / "intervention_map.html"
MAP_EMERGENCY_PATH     = MAPS_DIR / "emergency_route.html"

MODEL_PATH             = MODELS_DIR / "congestion_model.pkl"
ML_DATASET_PATH        = OUTPUT_DIR / "ml_dataset.csv"
BOTTLENECKS_PATH       = OUTPUT_DIR / "bottlenecks.csv"
HOTSPOTS_PATH          = OUTPUT_DIR / "hotspots.csv"
INTERVENTION_PATH      = OUTPUT_DIR / "intervention_results.csv"
EMERGENCY_PATH         = OUTPUT_DIR / "emergency_analysis.csv"

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
# Congestion weight formula coefficients (must sum to 1.0 for interpretability)
WEIGHT_TRAFFIC    = 0.35
WEIGHT_ACCIDENT   = 0.25
WEIGHT_VIOLATION  = 0.20
WEIGHT_SIGNAL     = 0.15
WEIGHT_COMPLIANCE = -0.05   # negative: higher enforcement → lower congestion

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL INTELLIGENCE (SYNTHETIC — clearly labelled)
# ─────────────────────────────────────────────────────────────────────────────
# Estimated signal delays (seconds) based on node degree (no real signal data)
SIGNAL_DELAY_BY_DEGREE = {
    3: 20,   # T-junction
    4: 35,   # 4-way intersection
    5: 50,   # complex intersection
}
SIGNAL_DELAY_DEFAULT = 10   # for degree < 3
CENTRALITY_SAMPLE_SOURCES = 200   # increase for accuracy, decrease for speed
CENTRALITY_RANDOM_SEED    = 42

# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
XGB_PARAMS = dict(
    n_estimators  = 200,
    max_depth     = 6,
    learning_rate = 0.05,
    subsample     = 0.8,
    colsample_bytree = 0.8,
    random_state  = RANDOM_STATE,
)

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
TOP_N_BOTTLENECKS  = 20
TOP_N_HOTSPOTS     = 15
EMERGENCY_SOURCE   = (12.9177, 77.6238)   # Silk Board
EMERGENCY_TARGET   = (12.9698, 77.7500)   # Whitefield

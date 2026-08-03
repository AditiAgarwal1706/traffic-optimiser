# Bangalore Traffic Congestion Modeling (Graph + ML + Optimization)

A graph-based urban traffic analysis pipeline for Bangalore that combines:

- **Real road network topology** from OpenStreetMap (via OSMnx)
- **Real civic/traffic datasets** (BTP accidents, violations, enforcement, and ward-level KML)
- **Proxy features** (area-level traffic density mapped onto roads)
- **Clearly-labelled synthetic components** where real measurements are not available
- **Weakly-supervised ML** to create a relative congestion score per road segment
- **Decision-support analyses**: bottlenecks, hotspots, emergency routing, and city-level optimization

This project produces **GraphML artifacts**, **CSVs**, **reports**, and **interactive Folium/Leaflet maps** in the `output/` folder.

---

## Key capabilities

- **Road network graph build** for Bangalore (OSM → GraphML + CSVs + map)
- **Graph enrichment** by mapping external spatial datasets onto nearest road edges
- **Feature engineering** into interpretable per-edge scores (accidents, violations, density, etc.)
- **Synthetic signal delay estimation** (topology-based) to approximate intersection delay effects
- **ML model training** using *pseudo-labels* (weak supervision) and applying predictions back onto the graph
- **Bottleneck + hotspot detection**, plus simple “what-if” intervention simulation
- **Emergency routing comparison** (distance vs congestion vs emergency-priority)
- **City-wide congestion optimization** (multiple algorithms) with before/after comparison maps
- **Optional: compact map generation** for very large Folium HTML files (browser-friendly rendering)

---

## Data transparency (important)

This repository intentionally labels each stage’s data sources:

- **Real data**
  - OSM road network topology (nodes/edges, road type tags, geometry)
  - Bengaluru Traffic Police (BTP) / Karnataka datasets included in `data/` (accidents, violations, enforcement)
  - Ward/polling boundaries and ward-level fields from the provided KML (if present)

- **Proxy / estimated data**
  - `traffic_density`: derived from `data/Banglore_traffic_Dataset.csv` (area-level), mapped to roads via fuzzy matching

- **Synthetic / assumed data**
  - Signal / intersection delays are estimated from topology (node degree, centrality, local density)
  - Temporal and weather-like ML features are *synthetic* (used only to train a model in the absence of real edge-level time series)

- **Weak supervision / pseudo-labels**
  - The ML target `congestion_score` is a *formula-derived pseudo-label*, not a measured ground truth
  - As a result, outputs should be interpreted as **relative ranking / decision-support indicators** rather than absolute congestion measurements

---

## Repository structure

- `src/` — all pipeline modules (each script can run standalone)
- `data/` — input datasets (CSVs + KML)
- `output/` — generated artifacts (graphs, maps, models, reports)
- `cache/` — intermediate caches (if used by scripts)

---

## Dashboard (interactive route comparison)

This repo includes a small Flask + Leaflet dashboard in `dashboard/` for comparing **before** vs **after** routing.

- It loads its data from **committed** CSV artifacts under `output/`.
- If you want to regenerate outputs, run the pipeline (see below).

Run from the project root:

```bash
source .venv/bin/activate
python dashboard/app.py
```

Then open:

- http://localhost:5050

If the dashboard reports missing files, run:

```bash
python src/run_pipeline.py
```

---

## Setup

### 1) Create and activate a virtual environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Notes on geospatial dependencies

This project depends on geospatial libraries such as `geopandas`, `shapely`, `pyproj`, and `fiona`.

- On macOS, wheels are usually available; if installation fails, you may need Xcode command line tools.
- If you see errors around GDAL/PROJ, installing via Conda can be a fallback.

---

## Quick start: run the full pipeline

The master runner is `src/run_pipeline.py`. Run from the project root:

```bash
python src/run_pipeline.py
```

### Resume from a step

```bash
python src/run_pipeline.py --from 4
```

### Run only one step

```bash
python src/run_pipeline.py --only 6
```

Tip: step numbers are defined in `src/run_pipeline.py` (e.g., `7=bottleneck_analysis`, `8=emergency_routing`).

---

## (Optional) SUMO simulation: run the GUI (macOS)

This repo can generate SUMO inputs under `sumo/` (net, routes, and `.sumocfg` configs).

On macOS, the `.pkg` SUMO build used here links `sumo-gui` against **X11**, so you must install and run **XQuartz** for the GUI to open.

1) Install XQuartz: https://www.xquartz.org

2) Launch the GUI using the helper script:

```bash
./scripts/run_sumo_gui.sh
```

By default this opens `sumo/cfg/before_simulation.sumocfg`. You can also pass a different config:

```bash
./scripts/run_sumo_gui.sh sumo/cfg/after_simulation.sumocfg
```

If you only need a headless run (no GUI), use:

```bash
/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/bin/sumo \
  -c sumo/cfg/before_simulation.sumocfg \
  --no-step-log true
```

---

## Output artifacts policy (GitHub)

Unlike many ML repos, this repo intentionally **commits `output/`** so that:

- the dashboard loads immediately after clone
- `output/maps/*.html` can be opened without first running a long pipeline

You can still regenerate everything locally by running `python src/run_pipeline.py`.

---

## Pipeline overview (what each step does)

The step runner orchestrates these modules:

1. **`build_graph`** — Download OSM road network for Bangalore
   - Outputs:
     - `output/graphs/bangalore_graph.graphml`
     - `output/graphs/nodes.csv`, `output/graphs/edges.csv`
     - `output/maps/road_network.html`

2. **`map_data_to_graph`** — Enrich graph edges by mapping external spatial data to nearest edges
   - Uses:
     - `data/*.csv` files that contain latitude/longitude columns
     - `data/*.kml` (ward/boundary file) if present
   - Output:
     - `output/graphs/bangalore_graph_enriched.graphml`

3. **`feature_engineering`** — Build interpretable feature scores and a congestion weight formula
   - Creates normalized scores (accidents, violations, density proxy, etc.)
   - Output:
     - `output/graphs/bangalore_weighted.graphml`

4. **`signal_intelligence`** — Synthetic intersection/signal delay estimation
   - Adds estimated node/edge delays and updates weights
   - Output:
     - `output/graphs/bangalore_signalized.graphml`

5. **`train_congestion_model`** — Train ML model (weak supervision)
   - Builds `output/ml_dataset.csv`
   - Trains multiple models and saves the best
   - Outputs:
     - `output/models/congestion_model.pkl`
     - `output/ml_dataset.csv`
     - `output/reports/model_evaluation.txt`

6. **`predict_congestion`** — Apply trained model back onto the graph
   - Adds `ml_congestion_score` to each edge
   - Output:
     - `output/graphs/bangalore_ml.graphml`

7. **`bottleneck_analysis`** — Bottlenecks, hotspots, and intervention simulation
   - Outputs:
     - `output/bottlenecks.csv`
     - `output/hotspots.csv`
     - `output/intervention_results.csv`
     - `output/maps/hotspot_map.html`
     - `output/maps/intervention_map.html`
     - `output/reports/bottleneck_summary.txt`

8. **`emergency_routing`** — Emergency routing analysis and route comparison
   - Outputs:
     - `output/emergency_analysis.csv`
     - `output/maps/emergency_route.html`
     - `output/reports/emergency_summary.txt`

9. **`urban_optimization`** — City-level planning/optimization reports (zone, road class, resilience)
   - Outputs include (examples):
     - `output/reports/urban_optimization_report.txt`
     - `output/reports/zone_congestion_summary.csv`
     - `output/reports/network_resilience.csv`
     - `output/reports/congestion_by_road_type.png`

---

## Configuration

Central configuration lives in `src/config.py`:

- `PLACE_NAME`, `NETWORK_TYPE` — OSM download scope
- All `DATA_DIR` and `OUTPUT_DIR` paths
- Pipeline artifact names (GraphML, map HTMLs, model paths)
- Modeling parameters (weights, random seeds, XGBoost params)
- Emergency routing endpoints (lat/lon for source/target)

If you want to run the pipeline for a different city or change file locations, update `src/config.py`.

---

## Viewing output maps

Folium maps are written to `output/maps/*.html`.

- Small maps can be opened directly.
- Very large maps (hundreds of thousands of polylines) can be slow or crash the browser.

### Recommended: serve via a local HTTP server

From the project root:

```bash
python -m http.server 8000
```

Then open (examples):

- http://localhost:8000/output/maps/road_network.html
- http://localhost:8000/output/maps/hotspot_map.html

---

## Compact (browser-friendly) versions of huge maps

Some maps (especially optimization comparison maps) can become extremely large because Folium embeds each polyline directly into the HTML.

This repo includes `src/compact_folium_polylines.py`, which:

- Extracts polyline coordinates + style + tooltip from the original map
- Writes them to a **separate JSON file**
- Rewrites the HTML to **fetch JSON and render via a loop** (much faster parsing)

### Example: compact the optimization comparison map

```bash
python src/compact_folium_polylines.py \
  output/maps/optimization_comparison.html \
  --max-polylines 20000 \
  --select sample \
  --seed 7 \
  --color-scheme bin3 \
  --html-out output/maps/optimization_comparison.compact.html \
  --data-out output/maps/optimization_comparison.compact.polylines.json
```

Notes:

- `--select sample` helps avoid a “top-only” bias (e.g., showing mostly high-congestion segments).
- `--color-scheme bin3` forces a strict **Low / Medium / High** palette.
- The compact HTML uses `fetch()`, so it **must** be opened via `http://localhost:8000/...` (not `file://`).

---

## Common troubleshooting

- **`ModuleNotFoundError` after installing**
  - Ensure VS Code/terminal is using the same interpreter as `.venv`.

- **Maps show blank / no polylines**
  - If using compact maps, confirm you are serving via `python -m http.server`.
  - Open DevTools → Console to check for `fetch()` errors.

- **Geospatial install errors (fiona/pyproj/shapely)**
  - On macOS, install Xcode command line tools (`xcode-select --install`).
  - If pip wheels fail, consider a Conda environment.

---

## Reproducibility

Randomness is controlled by `RANDOM_STATE` and other seeds in `src/config.py`.

Even with fixed seeds, results can vary slightly across platforms due to:

- floating point differences
- geospatial library versions
- OSM changes over time (if re-downloading the network)

---

## Disclaimer

This project is intended for educational and decision-support use.

- The ML outputs are trained on pseudo-labels and synthetic features.
- The optimization and intervention results are hypothetical scenario analyses.

Use the outputs as **relative signals for prioritization**, not as ground-truth congestion measurements.

---

## Standalone: congestion optimization algorithms

The pipeline focuses on graph building → ML scoring → analysis.

For the algorithmic “before/after” optimization maps and comparisons, run:

```bash
python src/congestion_optimizer.py
```

Key outputs include:

- `output/maps/optimization_comparison.html`
- `output/maps/algorithm_comparison.html`
- `output/reports/optimization_report.txt`
- `output/optimization_results.csv`

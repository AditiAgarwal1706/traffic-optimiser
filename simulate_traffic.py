"""
simulate_traffic.py  —  Bangalore Traffic Simulation (Before vs After)
Produces: output/traffic_simulation.mp4
"""
import sys, os
VENV_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python3")
if os.path.exists(VENV_PY) and os.path.realpath(sys.executable) != os.path.realpath(VENV_PY):
    os.execv(VENV_PY, [VENV_PY] + sys.argv)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import random, xml.etree.ElementTree as ET
from pathlib import Path

random.seed(42)
np.random.seed(42)

PROJECT   = Path(__file__).resolve().parent
OUT_VIDEO = PROJECT / "output" / "traffic_simulation.mp4"
OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  BANGALORE TRAFFIC SIMULATION  —  Before vs After")
print("=" * 60)

# ── 1. Load data ───────────────────────────────────────────────────────────
print("\n[1/4] Loading data …")

hot = pd.read_csv(PROJECT / "output/hotspots.csv")
bot = pd.read_csv(PROJECT / "output/bottlenecks.csv")
inter = pd.read_csv(PROJECT / "output/intervention_results.csv")

# Central Bangalore bbox
LAT_MIN, LAT_MAX = 12.90, 13.02
LON_MIN, LON_MAX = 77.54, 77.68

hot = hot[(hot.lat >= LAT_MIN) & (hot.lat <= LAT_MAX) &
          (hot.lon >= LON_MIN) & (hot.lon <= LON_MAX)].copy()
bot = bot[(bot.lat >= LAT_MIN) & (bot.lat <= LAT_MAX) &
          (bot.lon >= LON_MIN) & (bot.lon <= LON_MAX)].copy()

print(f"   Road segments: {len(hot):,}  |  Bottleneck nodes: {len(bot)}")

# ── 2. Build road segment geometry ────────────────────────────────────────
print("[2/4] Building road geometry …")

# hotspots has lat/lon of midpoint — we synthesise short segments around it
# using a small random offset to simulate road direction
rng = np.random.default_rng(42)

MAJOR_HW = {"primary", "trunk", "secondary", "motorway",
            "primary_link", "trunk_link", "secondary_link", "motorway_link"}
MEDIUM_HW = {"tertiary", "tertiary_link", "unclassified"}

def sample_segs(df, hw_set, n):
    sub = df[df.highway.isin(hw_set)] if hw_set else df
    if len(sub) > n:
        sub = sub.sample(n, random_state=42)
    return sub

major_segs  = sample_segs(hot, MAJOR_HW,  800)
medium_segs = sample_segs(hot, MEDIUM_HW, 1200)
minor_segs  = sample_segs(hot, None,      2000)   # all types, sampled

def segs_to_lines(df, length_deg=0.003):
    """Convert midpoint rows to (x0,y0,x1,y1,congestion) tuples."""
    out = []
    angles = rng.uniform(0, np.pi, len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        cx, cy = row.lon, row.lat
        ang = angles[i]
        dx = np.cos(ang) * length_deg * 0.5
        dy = np.sin(ang) * length_deg * 0.3
        out.append((cx - dx, cy - dy, cx + dx, cy + dy,
                    float(row.ml_congestion_score)))
    return out

lines_major  = segs_to_lines(major_segs,  0.006)
lines_medium = segs_to_lines(medium_segs, 0.004)
lines_minor  = segs_to_lines(minor_segs,  0.002)

all_route_segs = lines_major + lines_medium   # vehicles drive on these

print(f"   Major: {len(lines_major)}  Medium: {len(lines_medium)}  Minor: {len(lines_minor)}")

# ── 3. Vehicle simulation ──────────────────────────────────────────────────
print("[3/4] Setting up vehicles …")

N_VEH   = 120
N_FRAMES = 300
FPS      = 30

CMAP = LinearSegmentedColormap.from_list(
    "cong", ["#16a34a", "#eab308", "#dc2626"], N=256)

def make_vehicles(n, mode):
    vehs = []
    for _ in range(n):
        seg = random.choice(all_route_segs)
        x0, y0, x1, y1, cong = seg
        t = random.random()
        base = 0.006 + rng.normal(0, 0.001)
        sf = max(0.08, 1.0 - (0.82 * cong if mode == "before" else 0.38 * cong))
        vehs.append(dict(x0=x0, y0=y0, x1=x1, y1=y1,
                         t=t, speed=max(0.001, base * sf), cong=cong))
    return vehs

def step(vehs, mode):
    for v in vehs:
        v["t"] += v["speed"]
        if v["t"] >= 1.0:
            seg = random.choice(all_route_segs)
            v["x0"], v["y0"], v["x1"], v["y1"], v["cong"] = seg
            v["t"] = 0.0
            base = 0.006 + rng.normal(0, 0.001)
            sf = max(0.08, 1.0 - (0.82 * v["cong"] if mode == "before" else 0.38 * v["cong"]))
            v["speed"] = max(0.001, base * sf)

def positions(vehs):
    t = np.array([v["t"] for v in vehs])
    x0 = np.array([v["x0"] for v in vehs])
    y0 = np.array([v["y0"] for v in vehs])
    x1 = np.array([v["x1"] for v in vehs])
    y1 = np.array([v["y1"] for v in vehs])
    congs = np.array([v["cong"] for v in vehs])
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)
    return xs, ys, congs

vehs_before = make_vehicles(N_VEH, "before")
vehs_after  = make_vehicles(N_VEH, "after")

# ── 4. Build figure ────────────────────────────────────────────────────────
print("[4/4] Rendering …")

BG = "#0d1117"
fig, axes = plt.subplots(1, 2, figsize=(22, 11), facecolor=BG)
fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.06, wspace=0.03)

def build_lc(lines, lw, alpha, mode):
    segs, cols = [], []
    for x0, y0, x1, y1, cong in lines:
        segs.append([(x0, y0), (x1, y1)])
        c = cong if mode == "before" else max(0.0, cong * 0.42)
        cols.append(CMAP(c))
    return LineCollection(segs, colors=cols, linewidths=lw, alpha=alpha)

scats, stats_txt = [], []

for ax, mode, title_str, title_col in zip(
        axes,
        ["before", "after"],
        ["BEFORE  ·  Peak Congestion", "AFTER  ·  Optimised Routing"],
        ["#f87171", "#34d399"]):

    ax.set_facecolor(BG)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal")
    ax.axis("off")

    # road layers
    ax.add_collection(build_lc(lines_minor,  0.35, 0.30, mode))
    ax.add_collection(build_lc(lines_medium, 0.80, 0.55, mode))
    ax.add_collection(build_lc(lines_major,  2.00, 0.90, mode))

    # hotspot glows (top 15)
    if mode == "before":
        top_hot = hot.nlargest(15, "ml_congestion_score")
        for _, r in top_hot.iterrows():
            ax.plot(r.lon, r.lat, "o", color="#ff3333",
                    ms=14, alpha=0.12, zorder=3)
            ax.plot(r.lon, r.lat, "o", color="#ff3333",
                    ms=6,  alpha=0.80, zorder=4,
                    mec="#ff6666", mew=0.8)

    # bottleneck triangles
    for _, r in bot.head(10).iterrows():
        ax.plot(r.lon, r.lat, "^", color="#fb923c",
                ms=7, alpha=0.85, zorder=5,
                mec="#fdba74", mew=0.8)

    # intervention markers (after panel only)
    if mode == "after":
        inter_bbox = inter[(inter.lat >= LAT_MIN) & (inter.lat <= LAT_MAX) &
                           (inter.lon >= LON_MIN) & (inter.lon <= LON_MAX)].head(10)
        for _, r in inter_bbox.iterrows():
            ax.plot(r.lon, r.lat, "D", color="#22d3ee",
                    ms=6, alpha=0.85, zorder=5,
                    mec="#67e8f9", mew=0.8)

    # vehicles
    xs, ys, congs = positions(vehs_before if mode == "before" else vehs_after)
    cols = [CMAP(c) for c in congs]
    sc = ax.scatter(xs, ys, c=cols, s=22, zorder=10,
                    edgecolors="white", linewidths=0.25, alpha=0.95)
    scats.append(sc)

    ax.set_title(title_str, color=title_col, fontsize=17,
                 fontweight="bold", pad=8, fontfamily="monospace")

    st = ax.text(0.02, 0.03, "", transform=ax.transAxes,
                 color="#e2e8f0", fontsize=9.5, fontfamily="monospace",
                 va="bottom",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#1e293b",
                           ec="#334155", alpha=0.88))
    stats_txt.append(st)

# super-title
fig.text(0.5, 0.945, "🚦  Bangalore Road Traffic  —  Congestion Simulation",
         ha="center", fontsize=21, fontweight="bold",
         color="#f1f5f9", fontfamily="monospace")
fig.text(0.5, 0.912, "ML congestion scores · OSM road network · Central Bangalore (12.90–13.02°N)",
         ha="center", fontsize=10.5, color="#94a3b8", fontfamily="monospace")

# divider
fig.add_artist(plt.Line2D([0.5, 0.5], [0.05, 0.90],
               transform=fig.transFigure,
               color="#475569", lw=1.5, ls="--"))

# legend
patches = [
    mpatches.Patch(color="#16a34a", label="Low congestion"),
    mpatches.Patch(color="#eab308", label="Medium congestion"),
    mpatches.Patch(color="#dc2626", label="High congestion"),
    plt.Line2D([0],[0], marker="o",  ls="None", color="w",
               mfc="#ff3333", ms=8, label="Hotspot"),
    plt.Line2D([0],[0], marker="^",  ls="None", color="w",
               mfc="#fb923c", ms=8, label="Bottleneck"),
    plt.Line2D([0],[0], marker="D",  ls="None", color="w",
               mfc="#22d3ee", ms=7, label="Intervention (after)"),
]
fig.legend(handles=patches, loc="lower center", ncol=6,
           facecolor="#1e293b", edgecolor="#334155",
           labelcolor="#e2e8f0", fontsize=10,
           bbox_to_anchor=(0.5, 0.005))

timer_txt = fig.text(0.5, 0.875, "", ha="center",
                     color="#64748b", fontsize=9, fontfamily="monospace")

# ── animation ─────────────────────────────────────────────────────────────
def update(frame):
    step(vehs_before, "before")
    step(vehs_after,  "after")

    for i, (mode, veh, sc, st) in enumerate(
            zip(["before","after"], [vehs_before, vehs_after], scats, stats_txt)):
        xs, ys, congs = positions(veh)
        sc.set_offsets(np.column_stack([xs, ys]))
        sc.set_facecolor([CMAP(c) for c in congs])

        halting = sum(1 for v in veh if v["speed"] < 0.002)
        avg_cong = float(np.mean(congs))
        # rough speed: degrees/frame → m/s  (1 deg lat ≈ 111 km)
        avg_spd = float(np.mean([v["speed"] for v in veh])) * 111000 / (1/FPS)
        st.set_text(
            f"Vehicles : {len(veh)}\n"
            f"Halting  : {halting}\n"
            f"Avg speed: {avg_spd:.1f} m/s\n"
            f"Avg cong : {avg_cong:.2f}"
        )

    timer_txt.set_text(f"t = {frame/FPS:.1f}s  |  frame {frame+1}/{N_FRAMES}")
    return scats + stats_txt + [timer_txt]

ani = animation.FuncAnimation(
    fig, update, frames=N_FRAMES,
    interval=1000 // FPS, blit=False
)

writer = animation.FFMpegWriter(
    fps=FPS, bitrate=5000,
    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast"]
)

ani.save(str(OUT_VIDEO), writer=writer, dpi=130,
         progress_callback=lambda i, n: print(f"\r   Frame {i+1}/{n}  ", end="", flush=True))

print(f"\n\n✓  Saved → {OUT_VIDEO}")
print(f"   Size : {OUT_VIDEO.stat().st_size / 1e6:.1f} MB")
print(f"\n   Open : open \"{OUT_VIDEO}\"")

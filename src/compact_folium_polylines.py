#!/usr/bin/env python3
"""Compact huge Folium/Leaflet HTML files that contain hundreds of thousands of
`L.polyline(...)` blocks.

What it does
- Scans the input HTML and extracts per-polyline data (coords, style, tooltip).
- Keeps only the top N polylines by `Raw weight` (from tooltip) when available,
  otherwise falls back to Leaflet style `weight`.
- Writes a small JSON file with the kept polylines.
- Rewrites the HTML to load the JSON via `fetch()` and render polylines in a
  loop (optionally using a Canvas renderer).

Why this helps
- The original Folium HTML repeats lots of boilerplate per polyline; browsers
  struggle to parse/execute it at large scale.
- Externalizing data + looping reduces HTML size dramatically and improves
  responsiveness.

Notes
- The rewritten HTML expects to be served over HTTP (a local server). Opening
  it via `file://` will typically block `fetch()`.
- This is intentionally conservative: it only targets `L.polyline` blocks.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Iterable, Optional


_POLYLINE_START_RE = re.compile(rb"\bvar\s+(poly_line_[0-9a-f]+)\s*=\s*L\.polyline\(\s*$")
_ADD_TO_RE = re.compile(rb"\)\.addTo\((((?:feature_group|map)_[0-9a-f]+))\)\s*;\s*$")
_BIND_TOOLTIP_RE = re.compile(rb"\b(poly_line_[0-9a-f]+)\.bindTooltip\(\s*$")
_FEATURE_GROUP_ADD_TO_MAP_RE = re.compile(rb"\b(feature_group_[0-9a-f]+)\.addTo\((map_[0-9a-f]+)\)\s*;\s*$")
_TOOLTIP_SCORE_RE = re.compile(
    r"(?:Raw weight:\s*|Optimised weight:\s*|Optimized weight:\s*|\bWeight:\s*|\bscore\s*=\s*)([0-9]+(?:\.[0-9]+)?)"
)
_TOOLTIP_CONGESTION_RE = re.compile(r"\bCongestion:\s*([0-9]+(?:\.[0-9]+)?)")
_LAYER_CONTROL_RE = re.compile(rb"\b(?:var|let)\s+layer_control_[0-9a-f]+\b")


@dataclass
class Polyline:
    coords: list[list[float]]
    style: dict[str, Any]
    tooltip_html: str
    score: float
    target_var: str
    congestion: Optional[float]


def _bin3_color_label(value: float, t1: float, t2: float) -> str:
    # low/medium/high based on thresholds, lower is "better".
    if value < t1:
        return "low"
    if value < t2:
        return "medium"
    return "high"


def _read_balanced_js_literal(
    f: IO[bytes],
    start_line: bytes,
    open_ch: bytes,
    close_ch: bytes,
) -> bytes:
    """Read a JS literal that starts on `start_line` and ends when brackets balance."""

    if open_ch not in start_line:
        raise ValueError("start_line does not contain opening character")

    data = bytearray()

    def count_delta(chunk: bytes) -> int:
        return chunk.count(open_ch) - chunk.count(close_ch)

    depth = 0
    # Start at first opening char in the provided line.
    idx = start_line.index(open_ch)
    chunk = start_line[idx:]
    data.extend(chunk)
    depth += count_delta(chunk)

    while depth > 0:
        nxt = f.readline()
        if not nxt:
            break
        data.extend(nxt)
        depth += count_delta(nxt)

    return bytes(data)


def _json_loads_js(value: str) -> Any:
    """Parse a JSON-like JS literal emitted by folium (uses true/false/null)."""
    # Folium emits valid JSON for arrays/objects; we just need to strip trailing commas.
    value = value.strip().rstrip(",")
    return json.loads(value)


def _extract_tooltip_html(bind_block: bytes) -> str:
    """Extract the HTML inside a folium template literal (between backticks)."""
    text = bind_block.decode("utf-8", errors="replace")
    first = text.find("`")
    if first == -1:
        return ""
    second = text.find("`", first + 1)
    if second == -1:
        return ""
    return text[first + 1 : second]


def _score_from_tooltip_or_style(tooltip_html: str, style: dict[str, Any]) -> float:
    m = _TOOLTIP_SCORE_RE.search(tooltip_html)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    w = style.get("weight")
    try:
        return float(w) if w is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _congestion_from_tooltip(tooltip_html: str) -> Optional[float]:
    m = _TOOLTIP_CONGESTION_RE.search(tooltip_html)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _congestion_bin(congestion: Optional[float]) -> str:
    if congestion is None:
        return "unknown"
    if congestion < 0.33:
        return "low"
    if congestion < 0.66:
        return "medium"
    return "high"


def _parse_hex_color(value: Any) -> Optional[tuple[int, int, int]]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s.startswith("#"):
        return None
    s = s[1:]
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return r, g, b
    except ValueError:
        return None


def _color3_bin(style: dict[str, Any]) -> str:
    """Bin by closeness to the green/yellow/red palette used by these maps."""

    rgb = _parse_hex_color(style.get("color"))
    if rgb is None:
        return "unknown"

    # Approx anchors from legend / typical RdYlGn.
    low = (0x1A, 0x98, 0x50)  # green
    medium = (0xFE, 0xE0, 0x8B)  # yellow
    high = (0xD7, 0x30, 0x27)  # red

    def dist2(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2

    d_low = dist2(rgb, low)
    d_med = dist2(rgb, medium)
    d_high = dist2(rgb, high)
    if d_low <= d_med and d_low <= d_high:
        return "low"
    if d_med <= d_high:
        return "medium"
    return "high"


def extract_top_polylines(
    html_path: Path,
    max_polylines: int,
    stratify: str,
    select: str,
    seed: int,
) -> tuple[list[Polyline], dict[str, str]]:
    """Return (top_polylines, context) where context contains detected var names."""

    # target_var -> bin -> min-heap[(score, seq, polyline)]
    heaps: dict[str, dict[str, list[tuple[float, int, Polyline]]]] = {}
    # target_var -> reservoir list[Polyline]
    reservoirs: dict[str, list[Polyline]] = {}
    # target_var -> count seen
    seen_per_target: dict[str, int] = {}
    rng = random.Random(seed)
    seq = 0

    context: dict[str, str] = {}

    pending_poly_id: Optional[bytes] = None
    pending_coords: Optional[list[list[float]]] = None
    pending_style: Optional[dict[str, Any]] = None
    pending_target_var: str = ""

    total_polylines = 0
    detected_target_var: Optional[str] = None
    detected_targets: set[str] = set()
    saw_any_congestion = False
    saw_any_color = False

    with html_path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break

            m_start = _POLYLINE_START_RE.search(line)
            if m_start:
                poly_id = m_start.group(1)

                # coords array
                coords_line = f.readline()
                if not coords_line:
                    break
                coords_block = _read_balanced_js_literal(f, coords_line, b"[", b"]")
                coords = _json_loads_js(coords_block.decode("utf-8", errors="replace"))

                # style object (may be single-line or multi-line)
                style_line = f.readline()
                if not style_line:
                    break
                style_block = _read_balanced_js_literal(f, style_line, b"{", b"}")
                style = _json_loads_js(style_block.decode("utf-8", errors="replace"))

                # addTo(feature_group|map)
                add_line = f.readline()
                if not add_line:
                    break
                m_add = _ADD_TO_RE.search(add_line)
                target_var = m_add.group(1).decode("utf-8") if m_add else ""
                if target_var and not detected_target_var:
                    detected_target_var = target_var
                if target_var:
                    detected_targets.add(target_var)

                # Expect tooltip binding next; store pending polyline until we parse tooltip.
                pending_poly_id = poly_id
                pending_coords = coords
                pending_style = style
                pending_target_var = target_var

                total_polylines += 1
                continue

            m_bind = _BIND_TOOLTIP_RE.search(line)
            if m_bind and pending_poly_id and pending_coords is not None and pending_style is not None:
                bind_id = m_bind.group(1)
                if bind_id != pending_poly_id:
                    # Unexpected; skip this binding.
                    continue

                # Read until we reach the end of bindTooltip call.
                block = bytearray()
                while True:
                    nxt = f.readline()
                    if not nxt:
                        break
                    block.extend(nxt)
                    if b");" in nxt:
                        break

                tooltip_html = _extract_tooltip_html(bytes(block))
                score = _score_from_tooltip_or_style(tooltip_html, pending_style)
                congestion = _congestion_from_tooltip(tooltip_html)
                if congestion is not None:
                    saw_any_congestion = True
                if _parse_hex_color(pending_style.get("color")) is not None:
                    saw_any_color = True

                poly = Polyline(
                    coords=pending_coords,
                    style=pending_style,
                    tooltip_html=tooltip_html,
                    score=score,
                    target_var=pending_target_var,
                    congestion=congestion,
                )

                # Auto stratify: only use congestion bins if this file actually has congestion values.
                effective_stratify = stratify
                if stratify == "auto":
                    if saw_any_congestion:
                        effective_stratify = "congestion"
                    elif saw_any_color:
                        effective_stratify = "color3"
                    else:
                        effective_stratify = "none"

                if max_polylines > 0 and pending_target_var:
                    if select == "sample":
                        seen = seen_per_target.get(pending_target_var, 0) + 1
                        seen_per_target[pending_target_var] = seen
                        r = reservoirs.setdefault(pending_target_var, [])
                        if len(r) < max_polylines:
                            r.append(poly)
                        else:
                            j = rng.randrange(seen)
                            if j < max_polylines:
                                r[j] = poly
                        pending_poly_id = None
                        pending_coords = None
                        pending_style = None
                        pending_target_var = ""
                        continue

                    by_bin = heaps.setdefault(pending_target_var, {})

                    if effective_stratify == "congestion":
                        b = _congestion_bin(congestion)
                        quota = max_polylines // 3
                        if b == "unknown":
                            quota = max(1, max_polylines // 20)
                        h = by_bin.setdefault(b, [])
                        if len(h) < quota:
                            heapq.heappush(h, (score, seq, poly))
                            seq += 1
                        else:
                            if h and score > h[0][0]:
                                heapq.heapreplace(h, (score, seq, poly))
                                seq += 1
                    elif effective_stratify == "color3":
                        b = _color3_bin(pending_style)
                        quota = max_polylines // 3
                        if b == "unknown":
                            quota = max(1, max_polylines // 20)
                        h = by_bin.setdefault(b, [])
                        if len(h) < quota:
                            heapq.heappush(h, (score, seq, poly))
                            seq += 1
                        else:
                            if h and score > h[0][0]:
                                heapq.heapreplace(h, (score, seq, poly))
                                seq += 1
                    else:
                        h = by_bin.setdefault("all", [])
                        if len(h) < max_polylines:
                            heapq.heappush(h, (score, seq, poly))
                            seq += 1
                        else:
                            if h and score > h[0][0]:
                                heapq.heapreplace(h, (score, seq, poly))
                                seq += 1

                pending_poly_id = None
                pending_coords = None
                pending_style = None
                pending_target_var = ""
                continue

            m_fg_add = _FEATURE_GROUP_ADD_TO_MAP_RE.search(line)
            if m_fg_add:
                context["feature_group"] = m_fg_add.group(1).decode("utf-8")
                context["map_var"] = m_fg_add.group(2).decode("utf-8")

    polylines: list[Polyline] = []
    if select == "sample":
        for items in reservoirs.values():
            polylines.extend(items)
    else:
        for by_bin in heaps.values():
            for h in by_bin.values():
                polylines.extend([t[2] for t in h])

    # De-dup (rare, but safe) by (target, coords endpoints, tooltip)
    seen: set[tuple[str, tuple[float, float], tuple[float, float], str]] = set()
    uniq: list[Polyline] = []
    for p in polylines:
        if not p.coords:
            continue
        key = (p.target_var, tuple(p.coords[0]), tuple(p.coords[-1]), p.tooltip_html)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    uniq.sort(key=lambda p: p.score, reverse=True)

    if detected_target_var:
        context.setdefault("target_var", detected_target_var)
        context.setdefault("target_kind", "feature_group" if detected_target_var.startswith("feature_group_") else "map")
    if detected_targets:
        context["targets"] = ",".join(sorted(detected_targets))
    # Backwards-compat: keep feature_group when detected via feature_group.addTo(map)
    context.setdefault("total_polylines", str(total_polylines))
    return uniq, context


def rewrite_html(
    html_in: Path,
    html_out: Path,
    data_filename: str,
    target_var: str,
    target_kind: str,
    targets: list[str],
    color_scheme: str,
) -> None:
    """Rewrite HTML by removing inline polyline blocks and injecting a loader.

    This works for both patterns:
    - `var poly_line_x = L.polyline(...).addTo(feature_group_y);`
    - `var poly_line_x = L.polyline(...).addTo(map_z);`
    """

    injected = False
    pending_tooltips: set[bytes] = set()
    last_written_blank = False

    targets_js = "{\n" + ",\n".join([f"  {json.dumps(t)}: {t}" for t in targets]) + "\n}" if targets else "{}"

    loader = f"""

            // --- Compact loader injected by src/compact_folium_polylines.py ---
            (async function() {{
                const canvasRenderer = L.canvas({{ padding: 0.5 }});
                const res = await fetch({json.dumps(data_filename)});
                if (!res.ok) {{
                    throw new Error(`Failed to load polyline data: ${{res.status}} ${{res.statusText}}`);
                }}
                const data = await res.json();
                const defaultTarget = {target_var};
                const targets = {targets_js};

                const items = Array.isArray(data) ? data : (data.polylines || []);

                // Optional: enforce true Low/Medium/High colors.
                const COLOR_SCHEME = {json.dumps(color_scheme)};
                const lowColor = "#1a9850";
                const medColor = "#fee08b";
                const highColor = "#d73027";
                let t1 = null;
                let t2 = null;
                if (COLOR_SCHEME === "bin3") {{
                    const metrics = [];
                    for (const it of items) {{
                        // Prefer weight-like `score`; fallback to congestion.
                        const m = (typeof it.score === "number") ? it.score : ((typeof it.congestion === "number") ? it.congestion : null);
                        if (m !== null && Number.isFinite(m)) metrics.push(m);
                    }}
                    metrics.sort((a,b) => a - b);
                    const q = (p) => {{
                        if (!metrics.length) return null;
                        const idx = (metrics.length - 1) * p;
                        const lo = Math.floor(idx);
                        const hi = Math.ceil(idx);
                        if (lo === hi) return metrics[lo];
                        const w = idx - lo;
                        return metrics[lo] * (1 - w) + metrics[hi] * w;
                    }};
                    t1 = q(1/3);
                    t2 = q(2/3);
                }}
                for (const item of items) {{
                    const coords = item.coords || item[0];
                    const style = item.style || item[1] || {{}};
                    const tooltip = item.tooltip_html || item.tooltip || item[2] || "";
                    const targetKey = item.target || item.target_var || item.layer || "";
                    const target = (targetKey && targets[targetKey]) ? targets[targetKey] : defaultTarget;
                    const opts = Object.assign({{}}, style, {{ renderer: canvasRenderer }});

                    if (COLOR_SCHEME === "bin3" && t1 !== null && t2 !== null) {{
                        const m = (typeof item.score === "number") ? item.score : ((typeof item.congestion === "number") ? item.congestion : null);
                        if (m !== null && Number.isFinite(m)) {{
                            const color = (m < t1) ? lowColor : ((m < t2) ? medColor : highColor);
                            opts.color = color;
                            if (opts.fillColor) opts.fillColor = color;
                        }}
                    }}
                    const line = L.polyline(coords, opts).addTo(target);
                    if (tooltip) {{
                        line.bindTooltip(tooltip, {{ sticky: true }});
                    }}
                }}
            }})().catch((err) => {{
                console.error(err);
            }});
            // --- End compact loader ---

""".encode("utf-8")

    with html_in.open("rb") as src, html_out.open("wb") as dst:
        while True:
            line = src.readline()
            if not line:
                break

            # Prefer injecting after all layer targets exist (before layer_control), so multi-layer maps work.
            if (not injected) and _LAYER_CONTROL_RE.search(line):
                dst.write(loader)
                injected = True

            m_start = _POLYLINE_START_RE.search(line)
            if m_start:
                # Inject loader the first time we encounter polylines.
                # For multi-layer maps, wait until we reach layer_control (targets exist).
                if (not injected) and (len(targets) <= 1):
                    dst.write(loader)
                    injected = True

                poly_id = m_start.group(1)

                # Skip coords array
                coords_line = src.readline()
                if not coords_line:
                    break
                _read_balanced_js_literal(src, coords_line, b"[", b"]")

                # Skip style object
                style_line = src.readline()
                if not style_line:
                    break
                _read_balanced_js_literal(src, style_line, b"{", b"}")

                # Skip addTo(...) line
                add_line = src.readline()
                if not add_line:
                    break

                pending_tooltips.add(poly_id)
                continue

            m_bind = _BIND_TOOLTIP_RE.search(line)
            if m_bind:
                poly_id = m_bind.group(1)
                if poly_id in pending_tooltips:
                    # Skip until end of bindTooltip call.
                    while True:
                        nxt = src.readline()
                        if not nxt:
                            break
                        if b");" in nxt:
                            break
                    pending_tooltips.discard(poly_id)
                    continue

            # Collapse runs of blank/whitespace-only lines to keep HTML tiny.
            if line.strip() == b"":
                if last_written_blank:
                    continue
                last_written_blank = True
            else:
                last_written_blank = False

            dst.write(line)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compact Folium HTML with many L.polyline blocks")
    ap.add_argument("html", type=Path, help="Input HTML file")
    ap.add_argument(
        "--max-polylines",
        type=int,
        default=20000,
        help="Keep up to N polylines per layer target (default: 20000)",
    )
    ap.add_argument(
        "--select",
        choices=["top", "sample"],
        default="top",
        help="How to choose kept polylines: top by score, or random reservoir sample (default: top)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for --select sample (default: 7)",
    )
    ap.add_argument(
        "--stratify",
        choices=["auto", "none", "congestion", "color3"],
        default="auto",
        help="Selection strategy for kept polylines (default: auto)",
    )
    ap.add_argument(
        "--color-scheme",
        choices=["original", "bin3"],
        default="original",
        help="Render colors using original style or strict Low/Medium/High bins (default: original)",
    )
    ap.add_argument("--data-out", type=Path, default=None, help="Output JSON path (default: next to HTML)")
    ap.add_argument("--html-out", type=Path, default=None, help="Output HTML path (default: overwrite input)")

    args = ap.parse_args(list(argv) if argv is not None else None)

    html_in: Path = args.html
    if not html_in.exists():
        raise SystemExit(f"Input not found: {html_in}")

    html_out: Path = args.html_out or html_in
    data_out: Path = args.data_out or html_in.with_suffix(".polylines.json")

    polylines, ctx = extract_top_polylines(
        html_in,
        max_polylines=args.max_polylines,
        stratify=args.stratify,
        select=args.select,
        seed=args.seed,
    )
    target_var = (ctx.get("target_var") or ctx.get("feature_group") or "").strip()
    if not target_var:
        raise SystemExit("Could not detect where polylines are added (map_... or feature_group_...)")
    target_kind = (ctx.get("target_kind") or ("feature_group" if target_var.startswith("feature_group_") else "map")).strip()

    targets = [t for t in (ctx.get("targets", "").split(",") if ctx.get("targets") else []) if t]

    payload = {
        "meta": {
            "source": html_in.name,
            "total_polylines_detected": int(ctx.get("total_polylines", "0")),
            "kept": len(polylines),
            "kept_per_layer_target": int(args.max_polylines),
            "select": args.select,
            "seed": int(args.seed),
            "stratify": args.stratify,
            "color_scheme": args.color_scheme,
            "scoring": "Raw weight / score (tooltip) else Leaflet style weight",
        },
        "polylines": [
            {
                "coords": p.coords,
                "style": p.style,
                "tooltip_html": p.tooltip_html,
                "score": p.score,
                "target": p.target_var,
                "congestion": p.congestion,
            }
            for p in polylines
        ],
    }

    data_out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # Use a relative filename in fetch() so the HTML works when served from its folder.
    rewrite_html(
        html_in=html_in,
        html_out=html_out,
        data_filename=data_out.name,
        target_var=target_var,
        target_kind=target_kind,
        targets=targets,
        color_scheme=args.color_scheme,
    )

    print(f"Wrote JSON: {data_out} ({data_out.stat().st_size/1024/1024:.1f} MB)")
    print(f"Wrote HTML: {html_out} ({html_out.stat().st_size/1024/1024:.1f} MB)")
    print(f"Kept {len(polylines)} polylines out of {ctx.get('total_polylines','?')}")
    if targets:
        print(f"Targets: {', '.join(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build forecast_report.pdf - a printable report of the 10-model forecasts, in a soft dreamy style.

Reads the CSVs written by build_forecast_dashboard.py (no retraining), lays the report out
as HTML (output/forecast_report.html) and prints it to PDF with headless Chrome or Edge.
The desktop app calls configure() and build() instead, with the models it has trained.

Run:  python build_forecast_pdf.py
"""

import datetime
import html
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "output"
HTML_PATH = OUT / "forecast_report.html"
PDF_PATH = ROOT / "forecast_report.pdf"
# Fonts saved by packaging/fetch_fonts.py, embedded so the report looks the same offline.
FONT_CSS = Path(getattr(sys, "_MEIPASS", ROOT)) / "assets" / "fonts" / "fonts.css"

# The workbook the report describes; configure() points these at another one.
SOURCE = "antibiotic trend (2022-2025).xlsx"
HISTORY_YEARS = [2022, 2023, 2024, 2025]
FORECAST_YEARS = [2026, 2027]
EXCLUDED_NOTE = "Streptococcus spp. has no forecasts: no antibiotics showed an increase from 2022 to 2025."
SHORT_NAMES = {"Exponential Smoothing": "Exp. smoothing"}
NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()
BROWSERS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

# Data colours (checked with the dataviz palette validator against the card surface).
INK, INK_2 = "#3a2e5c", "#5b5080"
VIOLET, ROSE, LILAC = "#6c4fe0", "#e0508f", "#cfc2f6"
UP, DOWN = "#c2255c", "#1f9d74"  # rising resistance is bad, falling is good
# Sequential ramp for the heatmap; every step stays light enough for ink text.
RAMP = [(0, (251, 248, 255)), (50, (219, 205, 250)), (100, (180, 156, 241))]

# One soft sky per page: (top, middle, bottom) gradient stops.
SKY_COVER = ("#ffd6ea", "#e9dcff", "#cfe2ff")
SKY_GLANCE = ("#efe5ff", "#ffe4f1", "#fff0e2")
SKY_OUTRO = ("#fff0df", "#ffdcea", "#e6dcff")
SKY_ORGANISM = [
    ("#ffe1ef", "#f1e4ff", "#e3ecff"),
    ("#e6e1ff", "#dcecff", "#f4e6ff"),
    ("#dff3ff", "#e9e4ff", "#ffe9f4"),
    ("#fff0e1", "#ffe3ef", "#efe3ff"),
    ("#e0f7ee", "#e4ecff", "#f1e4ff"),
    ("#f5e1ff", "#ffe4f0", "#fff1e6"),
    ("#e3e6ff", "#f6e2ff", "#ffe6ee"),
]

FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Fraunces:ital,opsz,wght,SOFT@0,9..144,300..700,100;1,9..144,300..700,100"
    "&family=Nunito:ital,wght@0,400..800;1,400..800&display=block"
)

CSS = """
@page { size: 297mm 210mm; margin: 0; }
* { box-sizing: border-box; margin: 0; padding: 0; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Nunito", "Segoe UI", system-ui, sans-serif; color: #3a2e5c; font-size: 9.5pt; line-height: 1.45; }

.page { position: relative; width: 297mm; height: 210mm; overflow: hidden; break-after: page; }
.page:last-child { break-after: auto; }
.sky { position: absolute; inset: 0; }
.sky svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.content { position: relative; height: 100%; padding: 13mm 16mm 14mm; display: flex; flex-direction: column; gap: 4.5mm; }
.footer { position: absolute; left: 16mm; right: 16mm; bottom: 6.5mm; display: flex; justify-content: space-between;
          font-size: 7pt; letter-spacing: 0.04em; color: #5b5080; }

.kicker { font-size: 7pt; font-weight: 800; letter-spacing: 0.22em; text-transform: uppercase; color: #5b5080; }
h1, h2, .serif { font-family: "Fraunces", Georgia, serif; font-variation-settings: "SOFT" 100, "WONK" 0; }
h1 { font-size: 24pt; font-weight: 400; font-style: italic; line-height: 1.1; margin-top: 1.2mm; }
h1 .tail { font-style: normal; font-size: 14pt; color: #5b5080; }
h2 { font-size: 13pt; font-weight: 450; font-style: italic; line-height: 1.2; }
.shimmer { width: 36mm; height: 0.9mm; border-radius: 1mm; margin: 2.8mm 0 2.4mm;
           background: linear-gradient(90deg, #8f7cf0, #f08bb8 55%, #ffc39b); }
.lede { color: #4d4273; font-size: 9pt; max-width: 235mm; }
.sub { font-size: 8pt; color: #5b5080; margin: 0.6mm 0 2.5mm; }
.card { background: rgba(255, 255, 255, 0.66); border: 0.35mm solid rgba(255, 255, 255, 0.95); border-radius: 4.5mm;
        padding: 4mm 5mm; box-shadow: 0 1.2mm 4mm rgba(108, 79, 224, 0.08); }
.icon { display: inline-block; vertical-align: -0.2mm; }

/* Cover */
.cover { justify-content: center; padding-left: 24mm; gap: 0; }
.cover .kicker { font-size: 8pt; }
.cover-title { font-size: 46pt; font-weight: 350; font-style: normal; line-height: 1.04; letter-spacing: -0.01em; margin-top: 5mm; }
.cover-title em { display: block; font-style: italic; padding-bottom: 1.5mm; color: #6c4fe0;
                  background: linear-gradient(90deg, #6c4fe0, #c93b7d); -webkit-background-clip: text; background-clip: text;
                  -webkit-text-fill-color: transparent; }
.cover .shimmer { width: 62mm; height: 1.1mm; margin: 6mm 0 5mm; }
.cover-lede { font-size: 12pt; color: #4d4273; max-width: 150mm; line-height: 1.5; }
.pills { display: flex; flex-wrap: wrap; gap: 2.5mm; margin-top: 9mm; max-width: 175mm; }
.pill { background: rgba(255, 255, 255, 0.62); border: 0.3mm solid rgba(255, 255, 255, 0.95); border-radius: 10mm;
        padding: 1.6mm 4.2mm; font-size: 8.5pt; color: #4d4273; }
.pill b { font-weight: 800; color: #3a2e5c; }
.cover-foot { position: absolute; left: 24mm; bottom: 12mm; font-size: 7.5pt; color: #5b5080; }

/* At a glance */
.glance { display: grid; grid-template-columns: 152mm 1fr; gap: 6mm; align-items: start; }
.keys { display: flex; flex-wrap: wrap; gap: 2mm 6mm; font-size: 7.5pt; color: #5b5080; margin-top: 1.5mm; }
.keys span { display: inline-flex; align-items: center; gap: 1.6mm; }
.swatch { display: inline-block; width: 3.2mm; height: 3.2mm; border-radius: 0.8mm; }
.stack { display: flex; flex-direction: column; gap: 3.5mm; }
.mini { padding: 3.4mm 4.5mm; }
.mini h2 { font-size: 11.5pt; }
.mini .sub { margin-bottom: 1mm; }
.rank { list-style: none; }
.rank li { display: flex; justify-content: space-between; align-items: center; gap: 3mm; padding: 1.1mm 0;
           border-top: 0.2mm solid rgba(58, 46, 92, 0.10); }
.rank li:first-child { border-top: 0; }
.who { display: flex; flex-direction: column; line-height: 1.25; }
.who b { font-weight: 700; font-size: 8.8pt; }
.who i { font-size: 7.4pt; color: #5b5080; }
.val { font-weight: 800; font-size: 10pt; font-variant-numeric: tabular-nums; white-space: nowrap; }

/* Organism pages */
.head-row { display: flex; justify-content: space-between; align-items: flex-end; gap: 8mm; }
.legend { list-style: none; display: flex; flex-direction: column; gap: 1.3mm; font-size: 7.5pt; color: #4d4273; padding-bottom: 2.4mm; }
.legend li { display: flex; align-items: center; gap: 2mm; }
table { width: 100%; border-collapse: collapse; table-layout: fixed; }
th { font-size: 6.8pt; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #5b5080; text-align: left;
     padding: 0 2mm 1.8mm; border-bottom: 0.3mm solid rgba(58, 46, 92, 0.18); vertical-align: bottom; }
td { padding: 0 2mm; border-bottom: 0.2mm solid rgba(58, 46, 92, 0.09); vertical-align: middle; font-size: 8.8pt; white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.abx { font-weight: 700; }
.strong { font-weight: 800; }
td.spark { padding: 0 1mm; }
td.spark svg { display: block; }

table.heat { border-collapse: separate; border-spacing: 0.6mm; }
table.heat th { border-bottom: 0; padding: 0 0.8mm 0.8mm; text-align: center; white-space: normal; line-height: 1.2; }
table.heat th.left { text-align: left; }
table.heat tr.groups th { font-family: "Fraunces", Georgia, serif; font-style: italic; text-transform: none; letter-spacing: 0;
                          font-size: 9pt; font-weight: 450; color: #4d4273; border-bottom: 0.35mm solid rgba(108, 79, 224, 0.35); }
table.heat tr.groups th.blank { border-bottom: 0; }
table.heat td { border: 0; border-radius: 1.2mm; text-align: center; padding: 0 0.6mm; font-variant-numeric: tabular-nums; }
table.heat td.abx { text-align: left; padding-left: 1mm; }
td.cell b { font-weight: 800; font-size: 8.4pt; }
td.cell span { font-size: 6.6pt; margin-left: 1.1mm; }
.scale { display: flex; align-items: center; gap: 2.5mm; font-size: 7pt; color: #5b5080; margin-top: 2.6mm; }
.scale .bar { width: 60mm; height: 2.6mm; border-radius: 1.3mm; border: 0.2mm solid rgba(58, 46, 92, 0.12);
              background: linear-gradient(90deg, rgb(251, 248, 255), rgb(219, 205, 250), rgb(180, 156, 241)); }

/* Closing page */
.outro-grid { display: grid; grid-template-columns: 1.5fr 1fr; gap: 6mm; align-items: start; }
table.models td { font-size: 8pt; padding: 1.35mm 2mm; vertical-align: top; }
table.models td.desc { color: #4d4273; white-space: normal; }
.notes { list-style: none; display: flex; flex-direction: column; gap: 2.1mm; font-size: 8.3pt; color: #4d4273; margin-top: 1.5mm; }
.notes li { position: relative; padding-left: 4.5mm; }
.notes li::before { content: ""; position: absolute; left: 0; top: 1.5mm; width: 1.9mm; height: 1.9mm; border-radius: 50%;
                    background: linear-gradient(135deg, #8f7cf0, #f08bb8); }
"""


def excluded_note(excluded: dict) -> str:
    """'Streptococcus spp. has no forecasts: no antibiotics showed ...' for each organism left out."""
    return " ".join(f"{org} has no forecasts: {why[:1].lower()}{why[1:]}" for org, why in excluded.items())


def configure(history_years, source: str, excluded: dict) -> None:
    """Point the report at another workbook: its recorded years, its file name and the organisms left out.

    The forecasts shown are the two years after the last recorded one.
    """
    global HISTORY_YEARS, FORECAST_YEARS, SOURCE, EXCLUDED_NOTE
    HISTORY_YEARS = list(history_years)
    FORECAST_YEARS = [HISTORY_YEARS[-1] + 1, HISTORY_YEARS[-1] + 2]
    SOURCE = source
    EXCLUDED_NOTE = excluded_note(excluded)


# --- Small helpers ---------------------------------------------------------------------

def count_word(n: int) -> str:
    return NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)


def short_year(year: int) -> str:
    return str(year)[2:]


def all_rising(series) -> bool:
    return all(s["history"][-1] > s["history"][0] for s in series)


def has_early_zeros(series) -> bool:
    return any(0 in s["history"][:-1] for s in series)


def esc(text) -> str:
    return html.escape(str(text))


def pct(v: float) -> str:
    return f"{v:.1f}%"


def signed(d: float) -> str:
    if abs(d) < 0.05:
        return "0.0"
    return f"+{d:.1f}" if d > 0 else f"−{abs(d):.1f}"


def ramp(v: float) -> str:
    for (v0, c0), (v1, c1) in zip(RAMP, RAMP[1:]):
        if v <= v1:
            t = (v - v0) / (v1 - v0)
            return "rgb({}, {}, {})".format(*(round(a + (b - a) * t) for a, b in zip(c0, c1)))
    return "rgb({}, {}, {})".format(*RAMP[-1][1])


def sparkle_path(x, y, r) -> str:
    k = r * 0.16
    return (f"M{x:.2f},{y - r:.2f} Q{x + k:.2f},{y - k:.2f} {x + r:.2f},{y:.2f} "
            f"Q{x + k:.2f},{y + k:.2f} {x:.2f},{y + r:.2f} Q{x - k:.2f},{y + k:.2f} {x - r:.2f},{y:.2f} "
            f"Q{x - k:.2f},{y - k:.2f} {x:.2f},{y - r:.2f}Z")


def sparkle_icon(size=2.8, color=ROSE) -> str:
    return (f'<svg class="icon" width="{size}mm" height="{size}mm" viewBox="-1 -1 2 2" aria-hidden="true">'
            f'<path d="{sparkle_path(0, 0, 1)}" fill="{color}"/></svg>')


def arrow(d: float) -> str:
    if abs(d) < 0.05:
        return ""
    up = d > 0
    path = "M0,-0.9 L0.95,0.75 L-0.95,0.75Z" if up else "M0,0.9 L0.95,-0.75 L-0.95,-0.75Z"
    return (f'<svg class="icon" width="2mm" height="2mm" viewBox="-1 -1 2 2" aria-hidden="true">'
            f'<path d="{path}" fill="{UP if up else DOWN}"/></svg> ')


def key(kind: str) -> str:
    body = {
        "recorded": f'<line x1="0.5" y1="1.5" x2="8.5" y2="1.5" stroke="{VIOLET}" stroke-width="0.5" stroke-linecap="round"/>',
        "forecast": f'<line x1="0.5" y1="1.5" x2="8.5" y2="1.5" stroke="{ROSE}" stroke-width="0.5" stroke-dasharray="1.1 0.8"/>',
        "band": f'<rect x="0.5" y="0" width="8" height="3" rx="0.8" fill="{ROSE}" fill-opacity="0.22"/>',
    }[kind]
    return f'<svg class="icon" width="9mm" height="3mm" viewBox="0 0 9 3" aria-hidden="true">{body}</svg>'


# --- Dreamy backgrounds ----------------------------------------------------------------

SKY_DEFS = """<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
<radialGradient id="puff"><stop offset="0" stop-color="#fff" stop-opacity="0.95"/>
<stop offset="0.55" stop-color="#fff" stop-opacity="0.7"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>
<radialGradient id="orb"><stop offset="0" stop-color="#fffbe8" stop-opacity="1"/>
<stop offset="0.35" stop-color="#fff3f8" stop-opacity="0.75"/><stop offset="1" stop-color="#fff3f8" stop-opacity="0"/></radialGradient>
</defs></svg>"""


def cloud(cx, cy, s, rng) -> str:
    puffs = []
    for k in range(-3, 4):
        r = (15 - abs(k) * 2.6) * s * rng.uniform(0.85, 1.15)
        x = cx + k * 8.5 * s + rng.uniform(-2, 2) * s
        y = cy - r * 0.45 + rng.uniform(-1.5, 1.5) * s
        puffs.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="url(#puff)"/>')
    puffs.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{34 * s:.1f}" ry="{7 * s:.1f}" fill="url(#puff)"/>')
    return "".join(puffs)


def sky(colors, seed, clouds, sparkles=16, orb=None) -> str:
    """Gradient sky with soft clouds, an optional glowing orb, and scattered sparkles."""
    rng = random.Random(seed)
    top, mid, bottom = colors
    shapes = []
    if orb:
        cx, cy, r = orb
        shapes.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#orb)"/>')
    shapes += [cloud(cx, cy, s, rng) for cx, cy, s in clouds]
    for _ in range(sparkles):
        x, y = rng.uniform(8, 289), rng.uniform(6, 204)
        color = rng.choice(["#ffffff", "#ffffff", "#fff4c7", "#ffd3e8"])
        shapes.append(f'<path d="{sparkle_path(x, y, rng.uniform(0.9, 2.4))}" fill="{color}" '
                      f'opacity="{rng.uniform(0.55, 0.95):.2f}"/>')
        shapes.append(f'<circle cx="{rng.uniform(8, 289):.1f}" cy="{rng.uniform(6, 204):.1f}" '
                      f'r="{rng.uniform(0.25, 0.6):.2f}" fill="#fff" opacity="{rng.uniform(0.5, 0.9):.2f}"/>')
    return (f'<div class="sky" style="background: linear-gradient(165deg, {top} 0%, {mid} 52%, {bottom} 100%)">'
            f'<svg viewBox="0 0 297 210" preserveAspectRatio="none" aria-hidden="true">{"".join(shapes)}</svg></div>')


# --- Charts ----------------------------------------------------------------------------

def sparkline(s, model, w, h) -> str:
    """Recorded line, the chosen model's forecast, and the band spanned by all models (0-100% scale)."""
    hist, last = s["history"], len(s["history"]) - 1
    n = last + 3
    px, py = 1.5, 0.6
    x = lambda i: px + i * (w - 2 * px) / (n - 1)
    y = lambda v: h - py - v / 100 * (h - 2 * py)
    pts = lambda pairs: " ".join(f"{a:.2f},{b:.2f}" for a, b in pairs)
    f = s["forecast"][model]
    lo = [min(p[j] for p in s["forecast"].values()) for j in (0, 1)]
    hi = [max(p[j] for p in s["forecast"].values()) for j in (0, 1)]
    zone = (x(last) + x(last + 1)) / 2
    band = [(x(last), y(hist[-1])), (x(last + 1), y(hi[0])), (x(last + 2), y(hi[1])),
            (x(last + 2), y(lo[1])), (x(last + 1), y(lo[0]))]
    return (
        f'<svg width="{w}mm" height="{h:.2f}mm" viewBox="0 0 {w} {h:.2f}" aria-hidden="true">'
        f'<rect x="{zone:.2f}" y="0" width="{w - zone:.2f}" height="{h:.2f}" rx="0.8" fill="{ROSE}" fill-opacity="0.06"/>'
        f'<line x1="{px}" x2="{w - px}" y1="{y(0):.2f}" y2="{y(0):.2f}" stroke="{INK}" stroke-opacity="0.15" stroke-width="0.2"/>'
        f'<polygon points="{pts(band)}" fill="{ROSE}" fill-opacity="0.2"/>'
        f'<polyline points="{pts([(x(i), y(v)) for i, v in enumerate(hist)])}" fill="none" stroke="{VIOLET}" '
        f'stroke-width="0.45" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<polyline points="{pts([(x(last), y(hist[-1])), (x(last + 1), y(f[0])), (x(last + 2), y(f[1]))])}" '
        f'fill="none" stroke="{ROSE}" stroke-width="0.45" stroke-dasharray="1.1 0.8" stroke-linecap="round"/>'
        f'<circle cx="{x(last):.2f}" cy="{y(hist[-1]):.2f}" r="0.75" fill="{VIOLET}" stroke="#fff" stroke-width="0.3"/>'
        f'<circle cx="{x(last + 2):.2f}" cy="{y(f[1]):.2f}" r="0.75" fill="{ROSE}" stroke="#fff" stroke-width="0.3"/>'
        "</svg>"
    )


def hbar(x, y, w, h, r, fill) -> str:
    """Horizontal bar, square at the baseline and rounded at the data end."""
    r = min(r, w / 2, h / 2)
    return (f'<path d="M{x:.2f},{y:.2f} H{x + w - r:.2f} Q{x + w:.2f},{y:.2f} {x + w:.2f},{y + r:.2f} '
            f'V{y + h - r:.2f} Q{x + w:.2f},{y + h:.2f} {x + w - r:.2f},{y + h:.2f} H{x:.2f} Z" fill="{fill}"/>')


def leaderboard(models, baseline, w=141.0) -> str:
    ranked = sorted(models, key=lambda m: m["mae"])
    row_h, top, label_w, scale_max = 9.6, 9.0, 44.0, 20
    bar_w = w - label_w - 12
    xv = lambda v: label_w + v / scale_max * bar_w
    h = top + row_h * len(ranked) + 7
    bx = xv(baseline)
    parts = []
    for t in range(0, scale_max + 1, 5):
        parts.append(f'<line x1="{xv(t):.2f}" x2="{xv(t):.2f}" y1="{top - 1}" y2="{h - 6}" '
                     f'stroke="{INK}" stroke-opacity="0.08" stroke-width="0.2"/>')
        parts.append(f'<text x="{xv(t):.2f}" y="{h - 2}" font-size="2.5" fill="{INK_2}" text-anchor="middle">{t} pts</text>')
    # Baseline goes under the bars; value labels get a halo so the line never cuts through them.
    parts.append(f'<line x1="{bx:.2f}" x2="{bx:.2f}" y1="{top - 2.2}" y2="{h - 6}" stroke="{UP}" '
                 f'stroke-width="0.35" stroke-dasharray="1.2 0.9"/>')
    parts.append(f'<text x="{bx:.2f}" y="{top - 3.6}" font-size="2.6" font-weight="700" fill="{INK}" '
                 f'text-anchor="middle">Repeat last year · {baseline:.1f}</text>')
    for i, m in enumerate(ranked):
        y0 = top + i * row_h
        fill = VIOLET if m["mae"] < baseline else LILAC
        parts.append(f'<text x="0" y="{y0 + 4.1:.2f}" font-size="3.1" font-weight="700" fill="{INK}">{esc(m["name"])}</text>')
        parts.append(f'<text x="0" y="{y0 + 7.2:.2f}" font-size="2.4" fill="{INK_2}">{esc(m["family"])}</text>')
        parts.append(hbar(xv(0), y0 + 2.0, xv(m["mae"]) - xv(0), 4.4, 1.0, fill))
        parts.append(f'<text x="{xv(m["mae"]) + 1.6:.2f}" y="{y0 + 5.35:.2f}" font-size="3" font-weight="800" '
                     f'fill="{INK}" stroke="#fdf9ff" stroke-width="1.3" stroke-linejoin="round" '
                     f'paint-order="stroke">{m["mae"]:.1f}</text>')
    return f'<svg width="{w}mm" height="{h:.1f}mm" viewBox="0 0 {w} {h:.1f}" role="img">{"".join(parts)}</svg>'


# --- Pages -----------------------------------------------------------------------------

def head(kicker, title, lede, tail="") -> str:
    tail_html = f' <span class="tail">{esc(tail)}</span>' if tail else ""
    return (f'<div><div class="kicker">{esc(kicker)}</div><h1>{esc(title)}{tail_html}</h1>'
            f'<div class="shimmer"></div><p class="lede">{lede}</p></div>')


def cover_page(series, models, best, organisms) -> tuple:
    today = datetime.date.today()
    best_mae = next(m["mae"] for m in models if m["name"] == best)
    body = f"""
<div class="content cover">
  <div class="kicker">Forecast report · hospital isolates {HISTORY_YEARS[0]}–{HISTORY_YEARS[-1]}</div>
  <h1 class="cover-title">Antibiotic Resistance<em>Forecast {FORECAST_YEARS[0]}–{FORECAST_YEARS[-1]}</em></h1>
  <div class="shimmer"></div>
  <p class="cover-lede">What ten forecasting models, from ARIMA and Prophet to XGBoost and LSTM, predict for
  the next two years of resistance across {len(organisms)} organisms.</p>
  <div class="pills">
    <span class="pill"><b>{len(series)}</b> organism–antibiotic pairs</span>
    <span class="pill"><b>{len(organisms)}</b> organisms</span>
    <span class="pill"><b>{len(models)}</b> models</span>
    <span class="pill">{sparkle_icon(2.6)} Most accurate: <b>{esc(best)}</b> · avg miss {best_mae:.1f} pts</span>
  </div>
  <div class="cover-foot">Prepared {today.day} {today:%B %Y} · Source: {esc(SOURCE)}</div>
</div>"""
    clouds = [(40, 200, 1.5), (150, 207, 1.25), (262, 198, 1.6), (120, 40, 0.55)]
    return sky(SKY_COVER, 1, clouds, sparkles=28, orb=(236, 46, 62)), body


def glance_page(series, models, best, baseline) -> tuple:
    beat = [m["name"] for m in sorted(models, key=lambda m: m["mae"]) if m["mae"] < baseline]
    last_year, (y1, _) = HISTORY_YEARS[-1], FORECAST_YEARS
    rows = []
    for s in series:
        f26 = s["forecast"][best][0]
        all26 = [p[0] for p in s["forecast"].values()]
        rows.append({"s": s, "f26": f26, "jump": f26 - s["history"][-1], "lo": min(all26), "hi": max(all26)})

    def ranked(title, sub, key_fn, fmt):
        items = "".join(
            f'<li><span class="who"><b>{esc(r["s"]["antibiotic"])}</b><i>{esc(r["s"]["organism"])}</i></span>'
            f'<span class="val">{fmt(r)}</span></li>'
            for r in sorted(rows, key=key_fn, reverse=True)[:3]
        )
        return f'<div class="card mini"><h2>{title}</h2><div class="sub">{sub}</div><ol class="rank">{items}</ol></div>'

    lede = (f"Each model was trained on {HISTORY_YEARS[0]}–{HISTORY_YEARS[-2]} and asked to predict {HISTORY_YEARS[-1]}. "
            f"Only {len(beat)} of {len(models)} beat simply repeating last year's value ({baseline:.1f} pts): "
            f"{esc(', '.join(beat))}. The rest of this report uses {esc(best)}, the most accurate.")
    body = f"""
<div class="content">
  {head("At a glance", "Which model to trust", lede)}
  <div class="glance">
    <div class="card">
      <h2>Average miss when predicting {HISTORY_YEARS[-1]}</h2>
      <div class="sub">Percentage points; shorter is better</div>
      {leaderboard(models, baseline)}
      <div class="keys">
        <span><i class="swatch" style="background:{VIOLET}"></i>Beat repeating last year</span>
        <span><i class="swatch" style="background:{LILAC}"></i>Did not</span>
      </div>
    </div>
    <div class="stack">
      {ranked(f"Highest forecast for {y1}", f"{esc(best)} · % resistant", lambda r: r["f26"], lambda r: pct(r["f26"]))}
      {ranked(f"Biggest rise from {last_year} to {y1}", f"{esc(best)} · percentage points", lambda r: r["jump"],
              lambda r: arrow(r["jump"]) + signed(r["jump"]))}
      {ranked("Where the models disagree most", f"Lowest to highest {y1} forecast, all {len(models)} models",
              lambda r: r["hi"] - r["lo"], lambda r: f'{r["lo"]:.0f}–{r["hi"]:.0f}%')}
    </div>
  </div>
</div>"""
    clouds = [(250, 206, 1.4), (30, 210, 1.0)]
    return sky(SKY_GLANCE, 2, clouds, sparkles=14), body


def organism_pages(organism, items, idx, total_orgs, models, best) -> list:
    colors = SKY_ORGANISM[idx % len(SKY_ORGANISM)]
    items = sorted(items, key=lambda s: s["forecast"][best][0], reverse=True)
    n = len(items)
    last_year, (y1, y2) = HISTORY_YEARS[-1], FORECAST_YEARS

    # Page 1: forecast summary with sparklines.
    row_h = min(16.0, 134.0 / n)
    rows = []
    for s in items:
        f26, f27 = s["forecast"][best]
        last = s["history"][-1]
        all26 = [p[0] for p in s["forecast"].values()]
        rows.append(
            f'<tr style="height:{row_h:.2f}mm"><td class="abx">{esc(s["antibiotic"])}</td>'
            f'<td class="spark">{sparkline(s, best, 66, row_h - 1.4)}</td>'
            f'<td class="num">{pct(last)}</td><td class="num strong">{pct(f26)}</td><td class="num">{pct(f27)}</td>'
            f'<td class="num">{min(all26):.0f}–{max(all26):.0f}%</td>'
            f'<td class="num">{arrow(f26 - last)}{signed(f26 - last)}</td></tr>'
        )
    legend = (f'<ul class="legend"><li>{key("recorded")}Recorded {HISTORY_YEARS[0]}–{HISTORY_YEARS[-1]}</li>'
              f'<li>{key("forecast")}{esc(best)} forecast</li><li>{key("band")}Range of all {len(models)} models</li></ul>')
    lede = (f"{n} antibiotics · {esc(best)} forecast with the spread of all {len(models)} models · "
            f"highest {y1} forecast first")
    summary = f"""
<div class="content">
  <div class="head-row">{head(f"Organism {idx + 1} of {total_orgs} · forecast", organism, lede)}{legend}</div>
  <div class="card">
    <table>
      <colgroup><col style="width:54mm"><col style="width:70mm"><col style="width:25mm"><col style="width:25mm">
      <col style="width:25mm"><col style="width:28mm"><col></colgroup>
      <thead><tr><th>Antibiotic</th><th>Trend {HISTORY_YEARS[0]} → {y2} (0–100%)</th><th class="num">Recorded {last_year}</th>
      <th class="num">Forecast {y1}</th><th class="num">Forecast {y2}</th><th class="num">All models {y1}</th>
      <th class="num">Change {last_year}→{short_year(y1)} (pts)</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
</div>"""

    # Page 2: every model's prediction as a soft heatmap. The 0.6 mm cell gaps count against the height too.
    heat_h = min(13.0, (124.0 - 0.6 * (n + 3)) / n)
    families = []
    for m in models:
        if families and families[-1][0] == m["family"]:
            families[-1][1] += 1
        else:
            families.append([m["family"], 1])
    group_row = '<th class="blank"></th><th class="blank"></th>' + "".join(
        f'<th colspan="{count}">{esc(fam)}</th>' for fam, count in families)
    name_row = f'<th class="left">Antibiotic</th><th>Recorded {last_year}</th>' + "".join(
        f'<th>{sparkle_icon(2.2) + " " if m["name"] == best else ""}{esc(SHORT_NAMES.get(m["name"], m["name"]))}</th>'
        for m in models)
    heat_rows = []
    for s in items:
        last = s["history"][-1]
        cells = "".join(
            f'<td class="cell" style="background:{ramp(s["forecast"][m["name"]][0])}">'
            f'<b>{s["forecast"][m["name"]][0]:.1f}</b><span>{s["forecast"][m["name"]][1]:.1f}</span></td>'
            for m in models)
        heat_rows.append(f'<tr style="height:{heat_h:.2f}mm"><td class="abx">{esc(s["antibiotic"])}</td>'
                         f'<td class="cell" style="background:{ramp(last)}"><b>{last:.1f}</b></td>{cells}</tr>')
    lede = (f"Bold: {y1} forecast · small: {y2} forecast · % resistant · shading follows the bold value · "
            f"{sparkle_icon(2.4)} most accurate model")
    heatmap = f"""
<div class="content">
  {head(f"Organism {idx + 1} of {total_orgs} · all models", organism, lede, tail="· every model's forecast")}
  <div class="card">
    <table class="heat">
      <colgroup><col style="width:44mm"><col style="width:17mm">{"<col>" * len(models)}</colgroup>
      <thead><tr class="groups">{group_row}</tr><tr>{name_row}</tr></thead>
      <tbody>{"".join(heat_rows)}</tbody>
    </table>
    <div class="scale"><span>0%</span><span class="bar"></span><span>100%</span>
      <span>· shading = forecast or recorded value</span></div>
  </div>
</div>"""

    clouds = [(28, 212, 1.1), (272, 210, 1.2)]
    return [
        (sky(colors, 10 + idx * 2, clouds, sparkles=12, orb=(262, 18, 34)), summary),
        (sky(colors, 11 + idx * 2, clouds, sparkles=12, orb=(262, 18, 34)), heatmap),
    ]


def outro_page(models, best, baseline, series) -> tuple:
    rows = "".join(
        f'<tr><td class="abx">{sparkle_icon(2.2) + " " if m["name"] == best else ""}{esc(m["name"])}</td>'
        f'<td>{esc(m["family"])}</td><td class="desc">{esc(m["about"])}</td>'
        f'<td class="num">{m["mae"]:.1f}</td><td class="num">{m["rmse"]:.1f}</td></tr>'
        for m in models)
    best_mae = next(m["mae"] for m in models if m["name"] == best)
    y1, y2 = FORECAST_YEARS
    notes = [
        f"<b>Backtest.</b> Every model saw only {HISTORY_YEARS[0]}–{HISTORY_YEARS[-2]} and predicted {HISTORY_YEARS[-1]}. "
        f"Avg miss is the average gap to the real value in percentage points; repeating the "
        f"{HISTORY_YEARS[-2]} value would have missed by {baseline:.1f}.",
        f"<b>Two kinds of model.</b> Statistical models were fitted to each series alone. Machine-learning and "
        f"deep-learning models learned the year-to-year change from all {len(series)} series at once.",
        f"<b>Very short history.</b> {count_word(len(HISTORY_YEARS)).capitalize()} yearly values per series is little "
        f"to learn from, so even the best model misses by about {best_mae:.0f} points on average. Read forecasts as a "
        f"direction, not a precise number.",
        f"<b>{y2} is less certain.</b> It is forecast from the {y1} forecast, so errors compound.",
    ]
    if all_rising(series):
        notes.append(f"<b>Upward lean.</b> The workbook lists only antibiotics whose resistance rose from "
                     f"{HISTORY_YEARS[0]} to {HISTORY_YEARS[-1]}.")
    if has_early_zeros(series):
        notes.append("<b>Zeros.</b> Some antibiotics show exactly 0% in early years, which may mean not tested rather "
                     "than fully susceptible.")
    if EXCLUDED_NOTE:
        notes.append(f"<b>Not included.</b> {EXCLUDED_NOTE}")
    body = f"""
<div class="content">
  {head("About this report", "The models & the fine print",
        f"Source: {esc(SOURCE)} · how each model works and how it scored in the backtest")}
  <div class="outro-grid">
    <div class="card">
      <h2>The {count_word(len(models))} models</h2>
      <div class="sub">Avg miss and RMSE from the {HISTORY_YEARS[-1]} backtest, in percentage points</div>
      <table class="models">
        <colgroup><col style="width:38mm"><col style="width:27mm"><col><col style="width:15mm"><col style="width:13mm"></colgroup>
        <thead><tr><th>Model</th><th>Type</th><th>How it works</th><th class="num">Avg miss</th><th class="num">RMSE</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div class="card">
      <h2>Good to know</h2>
      <ul class="notes">{"".join(f"<li>{n}</li>" for n in notes)}</ul>
    </div>
  </div>
</div>"""
    clouds = [(60, 208, 1.4), (190, 212, 1.2), (285, 204, 1.0)]
    return sky(SKY_OUTRO, 99, clouds, sparkles=18, orb=(150, 222, 70)), body


# --- Assemble & print ------------------------------------------------------------------

def load():
    wide = pd.read_csv(OUT / "resistance_wide.csv")
    forecasts = pd.read_csv(OUT / "model_forecasts.csv").set_index(["organism", "antibiotic", "model"]).sort_index()
    models = pd.read_csv(OUT / "model_backtest.csv").to_dict("records")
    best = min(models, key=lambda m: m["mae"])["name"]
    last, prev = HISTORY_YEARS[-1], HISTORY_YEARS[-2]
    baseline = float((wide[f"pct_{last}"] - wide[f"pct_{prev}"]).abs().mean())
    columns = [f"forecast_{y}" for y in FORECAST_YEARS]
    series = [
        {
            "organism": row.organism,
            "antibiotic": row.antibiotic,
            "history": [getattr(row, f"pct_{y}") for y in HISTORY_YEARS],
            "forecast": {
                m["name"]: tuple(forecasts.loc[(row.organism, row.antibiotic, m["name"]), columns])
                for m in models
            },
        }
        for row in wide.itertuples(index=False)
    ]
    return models, best, baseline, series


def find_browser() -> Path:
    for path in BROWSERS:
        if path.exists():
            return path
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        if found := shutil.which(name):
            return Path(found)
    raise RuntimeError("Printing the report needs Microsoft Edge or Google Chrome.")


def print_pdf(browser: Path, html_path: Path, pdf_path: Path, extra=()) -> None:
    """Print an HTML file to PDF, in a throwaway browser profile so an open browser window can't interfere."""
    pdf_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdf-profile-", ignore_cleanup_errors=True) as profile:
        subprocess.run(
            [str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
             *extra, f"--print-to-pdf={pdf_path}", html_path.as_uri()],
            check=True, capture_output=True, timeout=180,
        )
    if not pdf_path.exists():
        raise RuntimeError(f"The browser did not write {pdf_path.name}.")


def font_links() -> str:
    """The report's fonts: embedded once packaging/fetch_fonts.py has saved them, else from Google Fonts."""
    if FONT_CSS.is_file():
        return f"<style>{FONT_CSS.read_text(encoding='utf-8')}</style>"
    return ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            f'<link rel="stylesheet" href="{FONTS}">')


def report_html(models, best, baseline, series) -> tuple[str, int]:
    organisms = list(dict.fromkeys(s["organism"] for s in series))

    pages = [cover_page(series, models, best, organisms), glance_page(series, models, best, baseline)]
    for i, org in enumerate(organisms):
        pages += organism_pages(org, [s for s in series if s["organism"] == org], i, len(organisms), models, best)
    pages.append(outro_page(models, best, baseline, series))

    years = f"{FORECAST_YEARS[0]}–{FORECAST_YEARS[-1]}"
    total = len(pages)
    sections = []
    for number, (background, body) in enumerate(pages, start=1):
        footer = ("" if number == 1 else
                  f'<div class="footer"><span>Antibiotic resistance forecast · {years}</span>'
                  f'<span>{number} / {total}</span></div>')
        sections.append(f'<section class="page">{background}{body}{footer}</section>')

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Antibiotic Resistance Forecast {years}</title>
{font_links()}
<style>{CSS}</style></head>
<body>{SKY_DEFS}{"".join(sections)}</body></html>"""
    return document, total


def build(data, html_path: Path, pdf_path: Path) -> int:
    """Lay out the report for data = (models, best, baseline, series) and print it; returns the page count."""
    document, total = report_html(*data)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(document, encoding="utf-8")
    print_pdf(find_browser(), html_path, pdf_path, extra=["--virtual-time-budget=20000"])
    return total


def main() -> None:
    total = build(load(), HTML_PATH, PDF_PATH)
    print(f"Wrote {PDF_PATH.name} ({total} pages, {PDF_PATH.stat().st_size / 1024:.0f} KB) using {find_browser().name}")


if __name__ == "__main__":
    main()

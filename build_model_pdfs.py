"""Build one PDF report per forecasting model, in a plain light-blue and white style.

Reads the CSVs written by build_forecast_dashboard.py (no retraining). Each of the ten models gets
its own report in model_reports/: a cover, how accurate the model was, its key forecasts, a page
per organism and the method notes. The reports are laid out as HTML (output/model_reports/) and
printed to PDF with headless Chrome or Edge. The desktop app calls configure() and build() instead,
with the models it has trained.

Run:  python build_model_pdfs.py
"""

import datetime
import math
import re
import statistics
from pathlib import Path

import build_forecast_pdf
from build_forecast_pdf import (EXCLUDED_NOTE, FORECAST_YEARS, HISTORY_YEARS, SOURCE, all_rising, count_word, esc,
                                find_browser, has_early_zeros, hbar, load, pct, print_pdf, signed)

ROOT = Path(__file__).parent
HTML_DIR = ROOT / "output" / "model_reports"
PDF_DIR = ROOT / "model_reports"
TITLE = f"Antibiotic resistance forecast {FORECAST_YEARS[0]}–{FORECAST_YEARS[-1]}"


def configure(history_years, source: str, excluded: dict) -> None:
    """Point the reports at another workbook; see build_forecast_pdf.configure."""
    global HISTORY_YEARS, FORECAST_YEARS, SOURCE, EXCLUDED_NOTE, TITLE
    build_forecast_pdf.configure(history_years, source, excluded)
    HISTORY_YEARS, FORECAST_YEARS = build_forecast_pdf.HISTORY_YEARS, build_forecast_pdf.FORECAST_YEARS
    SOURCE, EXCLUDED_NOTE = build_forecast_pdf.SOURCE, build_forecast_pdf.EXCLUDED_NOTE
    TITLE = f"Antibiotic resistance forecast {FORECAST_YEARS[0]}–{FORECAST_YEARS[-1]}"

# Light blue and white: text tokens, then the one data hue and its pale companions.
INK, INK_2, MUTED = "#0f2747", "#3c5778", "#5d7593"
BAND, RULE, GRID = "#e8f1fb", "#d5e4f4", "#e4edf7"
BLUE, PALE = "#2a78d6", "#c3d6ec"  # this model's marks; the other models' bars
UP, DOWN = "#d03b3b", "#0ca30c"  # rising resistance is bad, falling is good

# How each model works, from its settings in build_forecast_dashboard.py.
DETAILS = {
    "ARIMA": "A random walk with drift, ARIMA(0,1,0): each series is carried forward from its latest value by its "
             "average year-on-year change. {Years} yearly values cannot support a richer ARIMA order.",
    "Exponential Smoothing": "Holt's linear trend with damping: it follows the latest level and trend of each series, "
                             "then flattens the trend a little in each year ahead. The smoothing weights are fixed "
                             "(level 0.8, trend 0.2, damping 0.9) because {years} values are too few to estimate them.",
    "Theta": "The Theta method: it blends a straight-line trend fitted through the whole series with simple "
             "exponential smoothing of the latest values. No seasonal adjustment is used for yearly data.",
    "Prophet": "Meta's Prophet: it fits a piecewise-linear trend that can bend at changepoints and extends it "
               "forward. Seasonal terms are switched off because the data are yearly.",
    "Random Forest": "500 decision trees, each grown on a resample of the data with at least three examples per "
                     "leaf. The forecast is the average of all the trees.",
    "XGBoost": "Gradient-boosted decision trees: 300 shallow trees (depth 2) added one after another, each "
               "correcting the errors of those before it, with a small learning rate (0.03) and row and column "
               "sampling.",
    "LightGBM": "Gradient-boosted trees grown leaf by leaf: 300 small trees of four leaves each, with a small "
                "learning rate (0.03) and row and column sampling.",
    "CatBoost": "Gradient boosting with symmetric trees: 500 trees of depth 3 with a small learning rate (0.03).",
    "LSTM": "A small long short-term memory network (16 hidden units) that reads the previous two years in order, "
            "with the organism given at each step. The forecast averages five networks trained from different "
            "random starts, since one network trained on so little data depends heavily on its starting weights.",
    "GRU": "A small gated recurrent unit network (16 hidden units) that reads the previous two years in order, "
           "with the organism given at each step. The forecast averages five networks trained from different "
           "random starts, since one network trained on so little data depends heavily on its starting weights.",
}
PER_SERIES = ("Fitted to each organism–antibiotic series on its own, using only that series' {years} yearly values. "
              "The {first} and {last} forecasts are one and two steps ahead.")
ACROSS_SERIES = ("{Years} values per series are too few to train on, so the model is trained once across all {n} "
                 "series: it learns next year's change from the previous two years and the organism, and adds that "
                 "change to the latest value. {last} is forecast from the {first} forecast.")

CSS = """
@page { size: 297mm 210mm; margin: 0; }
* { box-sizing: border-box; margin: 0; padding: 0; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: #0f2747; background: #fff;
       font-size: 9pt; line-height: 1.4; }

.page { position: relative; width: 297mm; height: 210mm; overflow: hidden; break-after: page; background: #fff; }
.page:last-child { break-after: auto; }
.band { position: absolute; top: 0; left: 0; right: 0; height: 36mm; padding: 0 14mm; background: #e8f1fb;
        border-bottom: 0.35mm solid #cfe0f3; display: flex; flex-direction: column; justify-content: center; }
.kicker { font-size: 7pt; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #2a5f9e; }
h1 { font-size: 19pt; font-weight: 600; line-height: 1.15; margin-top: 1mm; }
h1 .tail { font-size: 12pt; font-weight: 400; color: #3c5778; }
.lede { margin-top: 1.4mm; font-size: 8.8pt; color: #3c5778; max-width: 215mm; }
.body { position: absolute; top: 36mm; left: 14mm; right: 14mm; bottom: 13mm; padding-top: 6mm;
        display: flex; flex-direction: column; gap: 5mm; }
.footer { position: absolute; left: 14mm; right: 14mm; bottom: 6mm; display: flex; justify-content: space-between;
          padding-top: 1.8mm; border-top: 0.25mm solid #d5e4f4; font-size: 7pt; color: #5d7593; }

.card { border: 0.3mm solid #d5e4f4; border-radius: 2mm; padding: 4mm 5mm; background: #fff; }
.card-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 8mm; margin-bottom: 2.5mm; }
h2 { font-size: 11pt; font-weight: 600; line-height: 1.25; }
.sub { font-size: 7.8pt; color: #5d7593; margin: 0.6mm 0 3mm; }
.card-head .sub { margin-bottom: 0; }
.icon { display: inline-block; vertical-align: -0.2mm; }
.legend { list-style: none; display: flex; gap: 5mm; font-size: 7.5pt; color: #3c5778; }
.legend li { display: flex; align-items: center; gap: 1.8mm; white-space: nowrap; }

table { width: 100%; border-collapse: collapse; table-layout: fixed; }
th { background: #eef5fc; color: #3c5778; font-size: 6.8pt; font-weight: 700; letter-spacing: 0.05em;
     text-transform: uppercase; text-align: left; padding: 1.8mm 2.5mm; vertical-align: bottom; line-height: 1.25; }
td { padding: 0 2.5mm; border-bottom: 0.2mm solid #e4edf7; font-size: 8.8pt; white-space: nowrap; vertical-align: middle; }
tbody tr:last-child td { border-bottom: 0; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.abx { font-weight: 600; }
.strong { font-weight: 700; }
td.spark { padding: 0 1.5mm; }
td.spark svg { display: block; }

/* Cover */
.cover-panel { position: absolute; top: 0; left: 0; right: 0; height: 124mm; padding: 15mm 24mm 13mm; background: #e8f1fb;
               border-top: 1.6mm solid #2a78d6; display: flex; flex-direction: column; justify-content: space-between; }
.cover-top { display: flex; justify-content: space-between; align-items: baseline; }
.cover-top .series { font-size: 8pt; color: #3c5778; }
.cover-title { font-size: 44pt; font-weight: 600; line-height: 1.05; letter-spacing: -0.01em; }
.cover-sub { font-size: 14pt; color: #3c5778; margin-top: 2.5mm; }
.cover-lede { font-size: 10.5pt; color: #3c5778; margin-top: 6mm; max-width: 170mm; line-height: 1.5; }
.cover-tiles { position: absolute; top: 137mm; left: 24mm; right: 24mm; display: grid;
               grid-template-columns: repeat(4, 1fr); gap: 5mm; }
.tile { border: 0.3mm solid #d5e4f4; border-radius: 2mm; padding: 3.6mm 4.5mm; }
.tile .label { font-size: 7pt; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #3c5778; }
.tile .value { font-size: 20pt; font-weight: 600; line-height: 1.15; margin-top: 1.4mm; }
.tile .value small { font-size: 10pt; font-weight: 400; color: #3c5778; }
.tile .foot { font-size: 7.6pt; color: #5d7593; margin-top: 1mm; }
.cover-foot { position: absolute; left: 24mm; right: 24mm; bottom: 10mm; display: flex; justify-content: space-between;
              padding-top: 2mm; border-top: 0.25mm solid #d5e4f4; font-size: 7.5pt; color: #5d7593; }

/* Overview */
.overview { display: grid; grid-template-columns: 152mm 1fr; gap: 6mm; align-items: start; }
.keys { display: flex; flex-wrap: wrap; gap: 1.5mm 6mm; font-size: 7.5pt; color: #3c5778; margin-top: 2mm; }
.keys span { display: inline-flex; align-items: center; gap: 1.6mm; }
.swatch { display: inline-block; width: 3.2mm; height: 3.2mm; border-radius: 0.7mm; }
.stack { display: flex; flex-direction: column; gap: 5mm; }
.tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; }
.tiles .tile { background: #f2f7fd; border-color: #e1ecf8; }
.tiles .tile .foot { color: #3c5778; }
.about p { font-size: 8.6pt; color: #24405f; margin-top: 2mm; }
.facts { list-style: none; margin-top: 3mm; border-top: 0.2mm solid #e4edf7; }
.facts li { display: grid; grid-template-columns: 24mm 1fr; gap: 3mm; padding: 1.5mm 0;
            border-bottom: 0.2mm solid #e4edf7; font-size: 8pt; }
.facts b { font-weight: 600; color: #3c5778; }

/* Key forecasts */
table.orgs td { height: 6.6mm; }
.lists { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5mm; }
.rank { list-style: none; }
.rank li { display: flex; align-items: center; gap: 2.5mm; padding: 1mm 0; border-top: 0.2mm solid #e4edf7; }
.rank li:first-child { border-top: 0; }
.no { flex: none; width: 4.6mm; height: 4.6mm; border-radius: 50%; background: #e8f1fb; color: #2a5f9e;
      font-size: 7pt; font-weight: 700; display: grid; place-items: center; }
.who { flex: 1; display: flex; flex-direction: column; line-height: 1.25; min-width: 0; }
.who b { font-weight: 600; font-size: 8.6pt; }
.who i { font-style: normal; font-size: 7.2pt; color: #5d7593; }
.val { font-weight: 700; font-size: 9.5pt; font-variant-numeric: tabular-nums; white-space: nowrap; text-align: right;
       line-height: 1.25; }
.val small { display: block; font-size: 7pt; font-weight: 400; color: #5d7593; }

/* Method & notes */
.notes-grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 6mm; align-items: start; }
table.models td { height: 7.4mm; font-size: 8.4pt; }
table.models tr.mine td { background: #e8f1fb; font-weight: 700; }
table.models tr.base td { color: #5d7593; font-style: italic; border-top: 0.3mm solid #d5e4f4; }
.notes { list-style: none; display: flex; flex-direction: column; gap: 2.2mm; font-size: 8.2pt; color: #24405f; margin-top: 1mm; }
.notes li { position: relative; padding-left: 4.5mm; }
.notes li::before { content: ""; position: absolute; left: 0; top: 1.6mm; width: 1.8mm; height: 1.8mm;
                    border-radius: 0.4mm; background: #2a78d6; }
"""


# --- Small helpers ---------------------------------------------------------------------

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def arrow(d: float) -> str:
    if abs(d) < 0.05:
        return ""
    up = d > 0
    path = "M0,-0.9 L0.95,0.75 L-0.95,0.75Z" if up else "M0,0.9 L0.95,-0.75 L-0.95,-0.75Z"
    return (f'<svg class="icon" width="2mm" height="2mm" viewBox="-1 -1 2 2" aria-hidden="true">'
            f'<path d="{path}" fill="{UP if up else DOWN}"/></svg> ')


def key(kind: str) -> str:
    body = {
        "recorded": f'<line x1="0.5" y1="1.5" x2="8.5" y2="1.5" stroke="{BLUE}" stroke-width="0.5" stroke-linecap="round"/>',
        "forecast": f'<line x1="0.5" y1="1.5" x2="8.5" y2="1.5" stroke="{BLUE}" stroke-width="0.5" stroke-dasharray="1.2 0.8"/>',
        "zone": f'<rect x="0.5" y="0" width="8" height="3" rx="0.6" fill="{BAND}"/>',
        "baseline": f'<line x1="4.5" y1="0" x2="4.5" y2="3" stroke="{INK}" stroke-width="0.35" stroke-dasharray="0.8 0.6"/>',
    }[kind]
    return f'<svg class="icon" width="9mm" height="3mm" viewBox="0 0 9 3" aria-hidden="true">{body}</svg>'


def ranked_list(title, sub, rows) -> str:
    items = "".join(
        f'<li><span class="no">{i}</span><span class="who"><b>{who}</b><i>{where}</i></span>'
        f'<span class="val">{val}</span></li>'
        for i, (who, where, val) in enumerate(rows, start=1))
    return f'<div class="card"><h2>{title}</h2><div class="sub">{sub}</div><ol class="rank">{items}</ol></div>'


# --- Charts ----------------------------------------------------------------------------

def sparkline(history, forecast, w, h) -> str:
    """Recorded years as a solid line and the forecast dashed, on a fixed 0-100% scale."""
    last = len(history) - 1
    n = last + 1 + len(forecast)
    px, py = 1.6, 0.6
    x = lambda i: px + i * (w - 2 * px) / (n - 1)
    y = lambda v: h - py - v / 100 * (h - 2 * py)
    pts = lambda pairs: " ".join(f"{a:.2f},{b:.2f}" for a, b in pairs)
    zone = (x(last) + x(last + 1)) / 2
    ahead = [(x(last), y(history[-1]))] + [(x(last + 1 + j), y(v)) for j, v in enumerate(forecast)]
    return (
        f'<svg width="{w}mm" height="{h:.2f}mm" viewBox="0 0 {w} {h:.2f}" aria-hidden="true">'
        f'<rect x="{zone:.2f}" y="0" width="{w - zone:.2f}" height="{h:.2f}" rx="0.6" fill="{BAND}"/>'
        f'<line x1="{px}" x2="{w - px}" y1="{y(0):.2f}" y2="{y(0):.2f}" stroke="{RULE}" stroke-width="0.25"/>'
        f'<polyline points="{pts([(x(i), y(v)) for i, v in enumerate(history)])}" fill="none" stroke="{BLUE}" '
        f'stroke-width="0.5" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<polyline points="{pts(ahead)}" fill="none" stroke="{BLUE}" stroke-width="0.5" stroke-dasharray="1.2 0.8" '
        f'stroke-linecap="round"/>'
        f'<circle cx="{x(last):.2f}" cy="{y(history[-1]):.2f}" r="0.8" fill="{BLUE}" stroke="#fff" stroke-width="0.35"/>'
        f'<circle cx="{ahead[-1][0]:.2f}" cy="{ahead[-1][1]:.2f}" r="0.75" fill="#fff" stroke="{BLUE}" stroke-width="0.4"/>'
        "</svg>"
    )


def accuracy_chart(models, current, baseline, w=141.0) -> str:
    """Average backtest miss for every model, shortest first: this model in blue, the others pale."""
    ranked = sorted(models, key=lambda m: m["mae"])
    row_h, top, label_w, scale_max = 9.4, 8.5, 42.0, 20
    bar_w = w - label_w - 12
    xv = lambda v: label_w + v / scale_max * bar_w
    h = top + row_h * len(ranked) + 7
    bx = xv(baseline)
    parts = []
    mine = next(i for i, m in enumerate(ranked) if m["name"] == current)
    parts.append(f'<rect x="0" y="{top + mine * row_h:.2f}" width="{w}" height="{row_h:.2f}" rx="1.2" fill="#eef5fc"/>')
    for t in range(0, scale_max + 1, 5):
        parts.append(f'<line x1="{xv(t):.2f}" x2="{xv(t):.2f}" y1="{top - 1}" y2="{h - 6}" '
                     f'stroke="{GRID}" stroke-width="0.25"/>')
        parts.append(f'<text x="{xv(t):.2f}" y="{h - 2}" font-size="2.5" fill="{MUTED}" text-anchor="middle">{t} pts</text>')
    # Baseline goes under the bars; value labels sit on a backing box so the line never cuts through them.
    parts.append(f'<line x1="{bx:.2f}" x2="{bx:.2f}" y1="{top - 2.2}" y2="{h - 6}" stroke="{INK}" '
                 f'stroke-width="0.3" stroke-dasharray="1 0.8"/>')
    parts.append(f'<text x="{bx:.2f}" y="{top - 3.6}" font-size="2.6" font-weight="600" fill="{INK}" '
                 f'text-anchor="middle">Repeat {HISTORY_YEARS[-2]} value · {baseline:.1f}</text>')
    for i, m in enumerate(ranked):
        y0 = top + i * row_h
        weight = 700 if i == mine else 600
        label, lx = f'{m["mae"]:.1f}', xv(m["mae"]) + 1.6
        parts.append(f'<text x="2" y="{y0 + 4.2:.2f}" font-size="3.1" font-weight="{weight}" fill="{INK}">{esc(m["name"])}</text>')
        parts.append(f'<text x="2" y="{y0 + 7.3:.2f}" font-size="2.4" fill="{MUTED}">{esc(m["family"])}</text>')
        parts.append(hbar(xv(0), y0 + 2.5, xv(m["mae"]) - xv(0), 4.4, 1.0, BLUE if i == mine else PALE))
        parts.append(f'<rect x="{lx - 0.7:.2f}" y="{y0 + 2.6:.2f}" width="{1.75 * len(label) + 1.2:.2f}" height="4.2" '
                     f'fill="{"#eef5fc" if i == mine else "#fff"}"/>')
        parts.append(f'<text x="{lx:.2f}" y="{y0 + 5.85:.2f}" font-size="3" font-weight="{weight}" fill="{INK}">{label}</text>')
    return f'<svg width="{w}mm" height="{h:.1f}mm" viewBox="0 0 {w} {h:.1f}" role="img">{"".join(parts)}</svg>'


# --- Pages -----------------------------------------------------------------------------

def page(kicker, title, lede, body, tail="") -> str:
    tail_html = f' <span class="tail">{esc(tail)}</span>' if tail else ""
    return (f'<header class="band"><div class="kicker">{esc(kicker)}</div><h1>{esc(title)}{tail_html}</h1>'
            f'<p class="lede">{lede}</p></header><div class="body">{body}</div>')


def cover_page(model, rank, models, best, n_series, n_orgs) -> str:
    today = datetime.date.today()
    number = [m["name"] for m in models].index(model["name"]) + 1
    trained = ("fitted to each series on its own" if model["family"] == "Statistical"
               else f"trained across all {n_series} series")
    if model["name"] == best:
        rank_foot = f"Most accurate of the {len(models)} models"
    else:
        best_mae = next(m["mae"] for m in models if m["name"] == best)
        rank_foot = f"Most accurate: {esc(best)} ({best_mae:.1f} pts)"
    tiles = [
        ("Average miss", f'{model["mae"]:.1f}<small> pts</small>',
         f"Trained on {HISTORY_YEARS[0]}–{HISTORY_YEARS[-2]}, tested on {HISTORY_YEARS[-1]}"),
        ("Accuracy rank", f"{rank}<small> of {len(models)}</small>", rank_foot),
        ("Forecasts", f"{n_series}<small> pairs</small>", f"For {FORECAST_YEARS[0]} and {FORECAST_YEARS[-1]}"),
        ("Organisms", f"{n_orgs}", "One page each in this report"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="label">{label}</div><div class="value">{value}</div>'
                         f'<div class="foot">{foot}</div></div>' for label, value, foot in tiles)
    return f"""
<div class="cover-panel">
  <div class="cover-top"><span class="kicker">{esc(TITLE)}</span>
  <span class="series">Model report {number} of {len(models)}</span></div>
  <div>
    <h1 class="cover-title">{esc(model["name"])}</h1>
    <div class="cover-sub">{esc(model["family"])} · {trained}</div>
    <p class="cover-lede">Resistance forecasts for {n_series} organism–antibiotic pairs across {n_orgs} organisms,
    based on hospital isolates recorded {HISTORY_YEARS[0]}–{HISTORY_YEARS[-1]}.</p>
  </div>
</div>
<div class="cover-tiles">{tiles_html}</div>
<div class="cover-foot"><span>Prepared {today.day} {today:%B %Y}</span><span>Source: {esc(SOURCE)}</span></div>"""


def overview_page(model, rank, models, baseline, n_series) -> str:
    name = model["name"]
    better = baseline - model["mae"]
    years = count_word(len(HISTORY_YEARS))
    fmt = {"first": FORECAST_YEARS[0], "last": FORECAST_YEARS[-1], "n": n_series,
           "years": years, "Years": years.capitalize()}
    statistical = model["family"] == "Statistical"
    approach = (PER_SERIES if statistical else ACROSS_SERIES).format(**fmt)
    facts = [
        ("Learns from", f"Each series on its own ({len(HISTORY_YEARS)} yearly values)" if statistical
         else f"All {n_series} series at once"),
        ("Predicts", "The next values of each series" if statistical
         else "Next year's change, added to the latest value"),
        ("Tested on", f"{HISTORY_YEARS[-1]}, after training on {HISTORY_YEARS[0]}–{HISTORY_YEARS[-2]}"),
        ("Forecasts", f"{FORECAST_YEARS[0]} and {FORECAST_YEARS[-1]} for {n_series} organism–antibiotic pairs"),
    ]
    tiles = [
        ("Average miss", f'{model["mae"]:.1f}<small> pts</small>', f"Mean absolute error, {HISTORY_YEARS[-1]} backtest"),
        ("Accuracy rank", f"{rank}<small> of {len(models)}</small>", "Ranked by average miss"),
        ("RMSE", f'{model["rmse"]:.1f}<small> pts</small>', "Weighs large misses more heavily"),
        ("Versus repeating last year", f'{abs(better):.1f}<small> pts {"better" if better > 0 else "worse"}</small>',
         f"Repeating the {HISTORY_YEARS[-2]} value missed by {baseline:.1f} pts"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="label">{label}</div><div class="value">{value}</div>'
                         f'<div class="foot">{foot}</div></div>' for label, value, foot in tiles)
    lede = (f"Trained on {HISTORY_YEARS[0]}–{HISTORY_YEARS[-2]} and asked to predict {HISTORY_YEARS[-1]}, "
            f"{esc(name)} missed the recorded value by {model['mae']:.1f} percentage points on average, "
            f"ranking {rank} of {len(models)} models.")
    body = f"""
<div class="overview">
  <div class="card">
    <h2>How accurate was it?</h2>
    <div class="sub">Average miss when predicting {HISTORY_YEARS[-1]} · percentage points · shorter is better</div>
    {accuracy_chart(models, name, baseline)}
    <div class="keys">
      <span><i class="swatch" style="background:{BLUE}"></i>{esc(name)}</span>
      <span><i class="swatch" style="background:{PALE}"></i>Other models</span>
      <span>{key("baseline")}Repeating the {HISTORY_YEARS[-2]} value</span>
    </div>
  </div>
  <div class="stack">
    <div class="tiles">{tiles_html}</div>
    <div class="card about">
      <h2>How it works</h2>
      <p>{esc(DETAILS[name].format(**fmt))}</p>
      <p>{esc(approach)}</p>
      <ul class="facts">{"".join(f"<li><b>{k}</b><span>{esc(v)}</span></li>" for k, v in facts)}</ul>
    </div>
  </div>
</div>"""
    return page("Model overview", name, lede, body, tail=f"· {model['family']}")


def highlights_page(model, series, organisms) -> str:
    name = model["name"]
    rows = []
    for s in series:
        f26 = s["forecast"][name][0]
        others = statistics.median(p[0] for m, p in s["forecast"].items() if m != name)
        rows.append({"s": s, "f26": f26, "jump": f26 - s["history"][-1], "others": others, "gap": f26 - others})

    org_rows = []
    for org in organisms:
        sub = [r for r in rows if r["s"]["organism"] == org]
        rising = sum(r["jump"] >= 0.05 for r in sub)
        mean_jump = sum(r["jump"] for r in sub) / len(sub)
        top = max(sub, key=lambda r: r["f26"])
        org_rows.append(
            f'<tr><td class="abx">{esc(org)}</td><td class="num">{len(sub)}</td>'
            f'<td class="num">{rising} of {len(sub)}</td>'
            f'<td class="num">{arrow(mean_jump)}{signed(mean_jump)} pts</td>'
            f'<td>{esc(top["s"]["antibiotic"])}</td><td class="num strong">{pct(top["f26"])}</td></tr>')

    def where(r):
        return esc(r["s"]["organism"])

    highest = [(esc(r["s"]["antibiotic"]), where(r), pct(r["f26"]))
               for r in sorted(rows, key=lambda r: r["f26"], reverse=True)[:5]]
    rises = [(esc(r["s"]["antibiotic"]), where(r), f'{arrow(r["jump"])}{signed(r["jump"])} pts')
             for r in sorted(rows, key=lambda r: r["jump"], reverse=True)[:5]]
    apart = [(esc(r["s"]["antibiotic"]), where(r),
              f'{signed(r["gap"])} pts<small>{r["f26"]:.1f}% vs {r["others"]:.1f}%</small>')
             for r in sorted(rows, key=lambda r: abs(r["gap"]), reverse=True)[:5]]

    y0, y1 = HISTORY_YEARS[-1], FORECAST_YEARS[0]
    body = f"""
<div class="card">
  <h2>By organism</h2>
  <div class="sub">{esc(name)} forecast for {y1} compared with the value recorded in {y0}</div>
  <table class="orgs">
    <colgroup><col style="width:72mm"><col style="width:26mm"><col style="width:38mm"><col style="width:40mm">
    <col><col style="width:26mm"></colgroup>
    <thead><tr><th>Organism</th><th class="num">Antibiotics</th><th class="num">Forecast to rise in {y1}</th>
    <th class="num">Average change {y0}→{str(y1)[2:]}</th><th>Highest {y1} forecast</th><th class="num">% resistant</th></tr></thead>
    <tbody>{"".join(org_rows)}</tbody>
  </table>
</div>
<div class="lists">
  {ranked_list(f"Highest forecasts for {y1}", "% resistant, all organisms", highest)}
  {ranked_list(f"Biggest rises, {y0} → {y1}", "Change from the recorded value, percentage points", rises)}
  {ranked_list("Furthest from the other models",
               f"{y1} forecast minus the median of the other {count_word(len(series[0]['forecast']) - 1)} models", apart)}
</div>"""
    lede = (f"Where {esc(name)} sees resistance heading in {y1}, organism by organism, and the forecasts that stand out.")
    return page("Key forecasts", name, lede, body, tail=f"· {y1} at a glance")


def organism_page(model, organism, items, idx, total, n_models) -> str:
    name = model["name"]
    items = sorted(items, key=lambda s: s["forecast"][name][0], reverse=True)
    n = len(items)
    row_h = min(13.0, 124.0 / n)
    rows = []
    for s in items:
        f26, f27 = s["forecast"][name]
        last = s["history"][-1]
        all26 = [p[0] for p in s["forecast"].values()]
        rows.append(
            f'<tr style="height:{row_h:.2f}mm"><td class="abx">{esc(s["antibiotic"])}</td>'
            f'<td class="spark">{sparkline(s["history"], (f26, f27), 66, row_h - 1.0)}</td>'
            f'<td class="num">{pct(last)}</td><td class="num strong">{pct(f26)}</td><td class="num">{pct(f27)}</td>'
            f'<td class="num">{arrow(f26 - last)}{signed(f26 - last)}</td>'
            f'<td class="num">{min(all26):.0f}–{max(all26):.0f}%</td></tr>'
        )
    first, last_year = HISTORY_YEARS[0], HISTORY_YEARS[-1]
    legend = (f'<ul class="legend"><li>{key("recorded")}Recorded {first}–{last_year}</li>'
              f'<li>{key("forecast")}{esc(name)} forecast</li><li>{key("zone")}Forecast years</li></ul>')
    body = f"""
<div class="card">
  <div class="card-head">
    <div><h2>Resistance by antibiotic</h2>
    <div class="sub">% of isolates resistant · trend drawn on a fixed 0–100% scale</div></div>
    {legend}
  </div>
  <table>
    <colgroup><col style="width:50mm"><col style="width:70mm"><col style="width:26mm"><col style="width:26mm">
    <col style="width:26mm"><col style="width:30mm"><col></colgroup>
    <thead><tr><th>Antibiotic</th><th>Trend {first} → {FORECAST_YEARS[-1]}</th><th class="num">Recorded {last_year}</th>
    <th class="num">Forecast {FORECAST_YEARS[0]}</th><th class="num">Forecast {FORECAST_YEARS[-1]}</th>
    <th class="num">Change {last_year}→{str(FORECAST_YEARS[0])[2:]} (pts)</th>
    <th class="num">All {n_models} models, {FORECAST_YEARS[0]}</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>"""
    lede = f"{n} antibiotics · {esc(name)} forecast · highest {FORECAST_YEARS[0]} forecast first"
    return page(f"Organism {idx + 1} of {total}", organism, lede, body)


def notes_page(model, models, baseline, series) -> str:
    name = model["name"]
    misses = [s["history"][-1] - s["history"][-2] for s in series]
    base_rmse = math.sqrt(sum(d * d for d in misses) / len(misses))
    ranked = sorted(models, key=lambda m: m["mae"])
    rows = "".join(
        f'<tr class="{"mine" if m["name"] == name else ""}"><td class="num">{i}</td><td>{esc(m["name"])}</td>'
        f'<td>{esc(m["family"])}</td><td class="num">{m["mae"]:.1f}</td><td class="num">{m["rmse"]:.1f}</td></tr>'
        for i, m in enumerate(ranked, start=1))
    rows += (f'<tr class="base"><td class="num">–</td><td>Repeat the {HISTORY_YEARS[-2]} value</td><td>Baseline</td>'
             f'<td class="num">{baseline:.1f}</td><td class="num">{base_rmse:.1f}</td></tr>')
    notes = [
        f"<b>Backtest.</b> Every model saw only {HISTORY_YEARS[0]}–{HISTORY_YEARS[-2]} and predicted {HISTORY_YEARS[-1]}. "
        f"Average miss is the mean gap to the recorded value in percentage points; RMSE weighs large misses more.",
        f"<b>Two kinds of model.</b> Statistical models were fitted to each series alone. Machine-learning and "
        f"deep-learning models learned the year-to-year change from all {len(series)} series at once.",
        f"<b>Very short history.</b> {count_word(len(HISTORY_YEARS)).capitalize()} yearly values per series is little "
        f"to learn from, so even the most accurate model misses by about {ranked[0]['mae']:.0f} points on average. "
        f"Read forecasts as a direction, not a precise number.",
        f"<b>{FORECAST_YEARS[-1]} is less certain.</b> It is two steps beyond the data, so errors compound.",
        "<b>Capped range.</b> Forecasts are limited to 0–100%.",
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
<div class="notes-grid">
  <div class="card">
    <h2>All {len(models)} models compared</h2>
    <div class="sub">{HISTORY_YEARS[-1]} backtest in percentage points, most accurate first · {esc(name)} highlighted</div>
    <table class="models">
      <colgroup><col style="width:12mm"><col><col style="width:34mm"><col style="width:22mm"><col style="width:18mm"></colgroup>
      <thead><tr><th class="num">Rank</th><th>Model</th><th>Type</th><th class="num">Avg miss</th><th class="num">RMSE</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>Good to know</h2>
    <ul class="notes">{"".join(f"<li>{n}</li>" for n in notes)}</ul>
  </div>
</div>"""
    return page("Method & notes", "How to read these forecasts", f"Source: {esc(SOURCE)} · how the models were tested "
                "and what the data can and cannot say", body)


# --- Assemble & print ------------------------------------------------------------------

def report_html(model, models, best, baseline, series, organisms) -> tuple:
    rank = [m["name"] for m in sorted(models, key=lambda m: m["mae"])].index(model["name"]) + 1
    pages = [
        cover_page(model, rank, models, best, len(series), len(organisms)),
        overview_page(model, rank, models, baseline, len(series)),
        highlights_page(model, series, organisms),
    ]
    for i, org in enumerate(organisms):
        items = [s for s in series if s["organism"] == org]
        pages.append(organism_page(model, org, items, i, len(organisms), len(models)))
    pages.append(notes_page(model, models, baseline, series))

    total = len(pages)
    sections = []
    for number, content in enumerate(pages, start=1):
        footer = ("" if number == 1 else
                  f'<footer class="footer"><span>{esc(model["name"])} · {esc(TITLE)}</span>'
                  f'<span>Page {number} of {total}</span></footer>')
        sections.append(f'<section class="page">{content}{footer}</section>')

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{esc(model["name"])} · Antibiotic Resistance Forecast {FORECAST_YEARS[0]}–{FORECAST_YEARS[-1]}</title>
<style>{CSS}</style></head>
<body>{"".join(sections)}</body></html>"""
    return document, total


def build(data, html_dir: Path, pdf_dir: Path, report=lambda done, total: None) -> list[tuple[Path, int]]:
    """Lay out and print one report per model for data = (models, best, baseline, series).

    Returns each PDF with its page count; report(done, total) is called after each one.
    """
    models, best, baseline, series = data
    organisms = list(dict.fromkeys(s["organism"] for s in series))
    browser = find_browser()
    html_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for i, model in enumerate(models):
        document, total = report_html(model, models, best, baseline, series, organisms)
        html_path = html_dir / f"{slug(model['name'])}.html"
        pdf_path = pdf_dir / f"forecast_{slug(model['name'])}.pdf"
        html_path.write_text(document, encoding="utf-8")
        print_pdf(browser, html_path, pdf_path)
        written.append((pdf_path, total))
        report(i + 1, len(models))
    return written


def main() -> None:
    for pdf_path, total in build(load(), HTML_DIR, PDF_DIR):
        print(f"Wrote {pdf_path.relative_to(ROOT)} ({total} pages, {pdf_path.stat().st_size / 1024:.0f} KB)")
    print(f"Done using {find_browser().name}")


if __name__ == "__main__":
    main()

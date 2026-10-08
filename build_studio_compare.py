"""Build the Studio's comparison report: the models the user picked, side by side, up to the year picked.

The Studio's other report (build_studio_report.py) covers the one model on screen; this one sets two or more of
them against each other and against repeating the year before. It has the same light-blue pages: a cover, how
accurate each model was and where each sees resistance heading, the forecasts at a glance for the last year
picked, a page per organism with a column per model, and the models and notes. One colour serves every model, so
the most accurate is set apart by shading, and the others by name. It is laid out as HTML and printed to PDF
with headless Edge or Chrome.

The desktop app calls build() through exports.write_compare_pdf.
"""

import datetime
import math
from pathlib import Path

import numpy as np

from build_forecast_pdf import (all_rising, count_word, esc, excluded_note, find_browser, has_early_zeros, hbar,
                                print_pdf, signed)
from build_model_pdfs import BAND, BLUE, GRID, INK, MUTED, PALE, RULE, arrow, key, page, ranked_list
from build_studio_report import CSS as REPORT_CSS, ROWS_PER_PAGE, span, tiles_html, years_word

SHORT_NAMES = {"Exponential Smoothing": "Exp. smoothing"}
NARROW_NAMES = {"Exponential Smoothing": "Exp. smooth."}  # for model columns under 17 mm
TABLE_SHARE = 178.0  # mm of an organism table for the trend and model columns
CHART_W = 120.0  # mm of each chart on the accuracy page, two side by side
SIDE_BY_SIDE = 4  # up to this many models, the accuracy page also has a table of them side by side
STAND_IN = "too little variation"  # in a learned model's "about" when it predicted the average change instead

CSS = REPORT_CSS + """
.cover-sub span { white-space: nowrap; }
.cover-sub.list { font-size: 11.5pt; line-height: 1.45; max-width: 240mm; }
.tile .value.name { font-size: 13pt; line-height: 1.3; margin-top: 2.2mm; }
.compare { display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; align-items: start; }
table.cmp th.group { text-align: center; border-bottom: 0.25mm solid #d5e4f4; padding-bottom: 1.2mm; }
table.cmp th.m { text-align: right; text-transform: none; letter-spacing: 0; font-size: 7pt; white-space: normal;
                 overflow-wrap: anywhere; padding: 1.4mm 1.4mm; }
table.cmp td.m { padding: 0 1.4mm; font-size: 8.4pt; }
table.cmp .best { background: #e8f1fb; font-weight: 700; }
table.cmp th.best { color: #0f2747; }
.val small { white-space: nowrap; }
.notes-grid.wide { grid-template-columns: 1.35fr 1fr; }
table.models td { padding-top: 0.6mm; padding-bottom: 0.6mm; }
table.models td.desc { white-space: normal; color: #3c5778; font-size: 7.4pt; line-height: 1.3; font-weight: 400; }
table.models td.mname b { display: block; font-weight: inherit; }
table.models td.mname small { display: block; font-size: 6.8pt; font-weight: 400; color: #5d7593; }
table.side td { height: 7.4mm; }
table.side tr.mine td { background: #e8f1fb; font-weight: 700; }
"""


def short(name: str) -> str:
    return SHORT_NAMES.get(name, name)


def names(models) -> str:
    """'ARIMA', 'ARIMA and GRU', 'ARIMA, Theta and GRU'."""
    items = [m["name"] if isinstance(m, dict) else m for m in models]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


class Comparison:
    """What every page needs: the models picked, the recorded values and each model's forecasts up to the year."""

    def __init__(self, engine, year: int):
        if len(engine.model_names) < 2:
            raise ValueError("The comparison covers two or more models")
        meta = engine.meta
        self.models = meta["models"]  # in the Studio's order
        self.ranked = sorted(self.models, key=lambda m: m["mae"])
        self.best = self.ranked[0]
        self.index = {m["name"]: j for j, m in enumerate(self.models)}
        self.hist = list(engine.history_years)
        self.k = year - self.hist[-1]
        self.years = list(range(self.hist[-1] + 1, year + 1))
        self.final = self.years[-1]
        self.recorded = np.asarray(engine.recorded, dtype=float)
        self.paths = np.asarray(engine.paths[:, :, :self.k], dtype=float)  # [series, model, year]
        final = self.paths[:, :, -1]
        self.series = [{"i": i, "organism": s["organism"], "antibiotic": s["antibiotic"], "history": self.recorded[i].tolist(),
                        "final": {m["name"]: float(final[i, j]) for j, m in enumerate(self.models)},
                        "lo": float(final[i].min()), "hi": float(final[i].max())}
                       for i, s in enumerate(engine.series)]
        for s in self.series:
            s["best"] = s["final"][self.best["name"]]
            s["jump"] = s["best"] - s["history"][-1]
            s["spread"] = s["hi"] - s["lo"]
            s["rising"] = sum(v - s["history"][-1] >= 0.05 for v in s["final"].values())
        self.organisms = list(dict.fromkeys(s["organism"] for s in self.series))
        misses = self.recorded[:, -1] - self.recorded[:, -2]
        self.base_mae = float(np.abs(misses).mean())
        self.base_rmse = float(np.sqrt((misses ** 2).mean()))
        self.beat = [m for m in self.ranked if m["mae"] < self.base_mae]
        self.stand_ins = [m for m in self.models if STAND_IN in m["about"]]
        self.source = meta["workbook"]
        self.excluded = excluded_note(meta.get("excluded") or {})
        self.title = f"Antibiotic resistance forecast {span(self.years)}"

    def sparkline(self, s, w: float, h: float) -> str:
        """A pair's trend: recorded, the most accurate model's forecast, and the range of every model."""
        paths = self.paths[s["i"]]
        return spread_sparkline(s["history"], paths[self.index[self.best["name"]]].tolist(),
                                paths.min(axis=0).tolist(), paths.max(axis=0).tolist(), w, h)

    def average(self, name: str) -> float:
        """A model's forecast for the last year picked, averaged over every pair."""
        return float(self.paths[:, self.index[name], -1].mean())


# --- Charts ----------------------------------------------------------------------------

def scale(lo_value: float, hi_value: float, ticks: int = 5) -> tuple[float, float, float]:
    """A round axis (low, high, step) covering the values with about `ticks` gridlines."""
    raw = max(hi_value - lo_value, 1.0) / ticks
    step = next(s for s in (1, 2, 5, 10, 20, 25) if s >= raw)
    lo = step * math.floor(lo_value / step)
    hi = step * math.ceil(hi_value / step)
    return lo, max(hi, lo + step), step


def ranked_rows(c: "Comparison", w: float, row_h: float, top: float, draw_row) -> tuple[list, float, list]:
    """The model names down the left, most accurate first on a shaded row; draw_row adds each row's marks.

    Returns what goes under the gridlines, the chart's height, and the rows to draw over them.
    """
    h = top + row_h * len(c.ranked) + 7
    parts = [f'<rect x="0" y="{top:.2f}" width="{w}" height="{row_h:.2f}" rx="1.2" fill="#eef5fc"/>']
    rows = []
    for i, m in enumerate(c.ranked):
        y0 = top + i * row_h
        weight = 700 if i == 0 else 600
        rows.append(f'<text x="2" y="{y0 + row_h / 2 - 0.2:.2f}" font-size="3.1" font-weight="{weight}" '
                    f'fill="{INK}">{esc(m["name"])}</text>')
        rows.append(f'<text x="2" y="{y0 + row_h / 2 + 2.9:.2f}" font-size="2.4" fill="{MUTED}">{esc(m["family"])}</text>')
        rows += draw_row(i, m, y0)
    return parts, h, rows


def accuracy_chart(c: "Comparison", w: float, row_h: float) -> str:
    """Each model's average miss in the backtest, shortest first, against repeating the year before."""
    top, label_w = 9.0, 40.0
    _, scale_max, step = scale(0, max(c.base_mae, c.ranked[-1]["mae"]), ticks=4)
    xv = lambda v: label_w + v / scale_max * (w - label_w - 12)
    bar_h = min(4.4, row_h * 0.48)

    def bar(i, m, y0):
        weight = 700 if i == 0 else 600
        label, lx = f'{m["mae"]:.1f}', xv(m["mae"]) + 1.6
        yb = y0 + (row_h - bar_h) / 2
        return [hbar(xv(0), yb, max(xv(m["mae"]) - xv(0), 0.6), bar_h, 1.0, BLUE if m in c.beat else PALE),
                f'<rect x="{lx - 0.7:.2f}" y="{yb + 0.1:.2f}" width="{1.75 * len(label) + 1.2:.2f}" height="{bar_h - 0.2:.2f}" '
                f'fill="{"#eef5fc" if i == 0 else "#fff"}"/>',
                f'<text x="{lx:.2f}" y="{yb + bar_h / 2 + 1.05:.2f}" font-size="3" font-weight="{weight}" fill="{INK}">{label}</text>']

    parts, h, rows = ranked_rows(c, w, row_h, top, bar)
    for t in np.arange(0, scale_max + step / 2, step):
        parts.append(f'<line x1="{xv(t):.2f}" x2="{xv(t):.2f}" y1="{top - 1}" y2="{h - 6}" stroke="{GRID}" stroke-width="0.25"/>')
        parts.append(f'<text x="{xv(t):.2f}" y="{h - 2}" font-size="2.5" fill="{MUTED}" text-anchor="middle">{t:g} pts</text>')
    # The baseline goes under the bars; the value labels sit on a backing box so it never cuts through them.
    bx = xv(c.base_mae)
    tx = min(max(bx, label_w + 19), w - 19)
    parts.append(f'<line x1="{bx:.2f}" x2="{bx:.2f}" y1="{top - 2.2}" y2="{h - 6}" stroke="{INK}" '
                 f'stroke-width="0.3" stroke-dasharray="1 0.8"/>')
    parts.append(f'<text x="{tx:.2f}" y="{top - 3.6}" font-size="2.6" font-weight="600" fill="{INK}" '
                 f'text-anchor="middle">Repeat {c.hist[-2]} value · {c.base_mae:.1f}</text>')
    return f'<svg width="{w}mm" height="{h:.1f}mm" viewBox="0 0 {w} {h:.1f}" role="img">{"".join(parts + rows)}</svg>'


def direction_chart(c: "Comparison", w: float, row_h: float) -> str:
    """Each model's forecast for the last year picked, averaged over every pair, drawn from the recorded average."""
    top, label_w, value_w = 9.0, 40.0, 13.0
    recorded = float(c.recorded[:, -1].mean())
    averages = {m["name"]: c.average(m["name"]) for m in c.models}
    lo, hi, step = scale(max(0.0, min(recorded, *averages.values()) - 1), min(100.0, max(recorded, *averages.values()) + 1))
    xv = lambda v: label_w + 2 + (v - lo) / (hi - lo) * (w - label_w - value_w - 4)
    rx = xv(recorded)

    def lollipop(i, m, y0):
        v = averages[m["name"]]
        mid = y0 + row_h / 2
        weight = 700 if i == 0 else 600
        return [f'<line x1="{rx:.2f}" x2="{xv(v):.2f}" y1="{mid:.2f}" y2="{mid:.2f}" stroke="{BLUE}" stroke-width="0.5" '
                f'stroke-linecap="round"/>',
                f'<circle cx="{xv(v):.2f}" cy="{mid:.2f}" r="1.2" fill="{BLUE}" stroke="#fff" stroke-width="0.4"/>',
                f'<text x="{w - 1:.2f}" y="{mid + 1.05:.2f}" font-size="3" font-weight="{weight}" fill="{INK}" '
                f'text-anchor="end">{v:.1f}%</text>']

    parts, h, rows = ranked_rows(c, w, row_h, top, lollipop)
    for t in np.arange(lo, hi + step / 2, step):
        parts.append(f'<line x1="{xv(t):.2f}" x2="{xv(t):.2f}" y1="{top - 1}" y2="{h - 6}" stroke="{GRID}" stroke-width="0.25"/>')
        parts.append(f'<text x="{xv(t):.2f}" y="{h - 2}" font-size="2.5" fill="{MUTED}" text-anchor="middle">{t:g}%</text>')
    tx = min(max(rx, label_w + 19), w - value_w - 17)
    parts.append(f'<line x1="{rx:.2f}" x2="{rx:.2f}" y1="{top - 2.2}" y2="{h - 6}" stroke="{INK}" '
                 f'stroke-width="0.3" stroke-dasharray="1 0.8"/>')
    parts.append(f'<text x="{tx:.2f}" y="{top - 3.6}" font-size="2.6" font-weight="600" fill="{INK}" '
                 f'text-anchor="middle">Recorded {c.hist[-1]} · {recorded:.1f}%</text>')
    return f'<svg width="{w}mm" height="{h:.1f}mm" viewBox="0 0 {w} {h:.1f}" role="img">{"".join(parts + rows)}</svg>'


def spread_sparkline(history, best, lo, hi, w, h) -> str:
    """Recorded years solid, the most accurate model dashed, and the range of every model shaded (0-100%)."""
    last = len(history) - 1
    n = last + 1 + len(best)
    px, py = 1.6, 0.6
    x = lambda i: px + i * (w - 2 * px) / (n - 1)
    y = lambda v: h - py - v / 100 * (h - 2 * py)
    pts = lambda pairs: " ".join(f"{a:.2f},{b:.2f}" for a, b in pairs)
    zone = (x(last) + x(last + 1)) / 2
    start = (x(last), y(history[-1]))
    ahead = [start] + [(x(last + 1 + j), y(v)) for j, v in enumerate(best)]
    band = ([start] + [(x(last + 1 + j), y(v)) for j, v in enumerate(hi)]
            + [(x(last + 1 + j), y(v)) for j, v in reversed(list(enumerate(lo)))])
    return (
        f'<svg width="{w}mm" height="{h:.2f}mm" viewBox="0 0 {w} {h:.2f}" aria-hidden="true">'
        f'<rect x="{zone:.2f}" y="0" width="{w - zone:.2f}" height="{h:.2f}" rx="0.6" fill="{BAND}"/>'
        f'<line x1="{px}" x2="{w - px}" y1="{y(0):.2f}" y2="{y(0):.2f}" stroke="{RULE}" stroke-width="0.25"/>'
        f'<polygon points="{pts(band)}" fill="{BLUE}" fill-opacity="0.18"/>'
        f'<polyline points="{pts([(x(i), y(v)) for i, v in enumerate(history)])}" fill="none" stroke="{BLUE}" '
        f'stroke-width="0.5" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<polyline points="{pts(ahead)}" fill="none" stroke="{BLUE}" stroke-width="0.5" stroke-dasharray="1.2 0.8" '
        f'stroke-linecap="round"/>'
        f'<circle cx="{start[0]:.2f}" cy="{start[1]:.2f}" r="0.8" fill="{BLUE}" stroke="#fff" stroke-width="0.35"/>'
        f'<circle cx="{ahead[-1][0]:.2f}" cy="{ahead[-1][1]:.2f}" r="0.75" fill="#fff" stroke="{BLUE}" stroke-width="0.4"/>'
        "</svg>"
    )


def band_key() -> str:
    return (f'<svg class="icon" width="9mm" height="3mm" viewBox="0 0 9 3" aria-hidden="true">'
            f'<rect x="0.5" y="0" width="8" height="3" rx="0.6" fill="{BLUE}" fill-opacity="0.18"/></svg>')


def dot_key() -> str:
    return (f'<svg class="icon" width="9mm" height="3mm" viewBox="0 0 9 3" aria-hidden="true">'
            f'<line x1="0.5" x2="6" y1="1.5" y2="1.5" stroke="{BLUE}" stroke-width="0.5"/>'
            f'<circle cx="6.5" cy="1.5" r="1.2" fill="{BLUE}"/></svg>')


# --- Pages -----------------------------------------------------------------------------

def cover_page(c: Comparison) -> str:
    today = datetime.date.today()
    n = len(c.models)
    tiles = (f'<div class="tile"><div class="label">Most accurate</div>'
             f'<div class="value name">{esc(c.best["name"])}</div>'
             f'<div class="foot">Average miss {c.best["mae"]:.1f} pts, tested on {c.hist[-1]}</div></div>') + tiles_html([
        ("Beat no change", f"{len(c.beat)}<small> of {n}</small>",
         f"Repeating the {c.hist[-2]} value missed by {c.base_mae:.1f} pts"),
        ("Forecasts", f"{len(c.series)}<small> pairs</small>", f"For {span(c.years)}, from each model"),
        ("Organisms", f"{len(c.organisms)}", "One page each in this report"),
    ])
    return f"""
<div class="cover-panel">
  <div class="cover-top"><span class="kicker">{esc(c.title)}</span><span class="series">Model comparison</span></div>
  <div>
    <h1 class="cover-title">{count_word(n).capitalize()} models compared</h1>
    <div class="cover-sub{" list" if n > 4 else ""}">{" · ".join(f"<span>{esc(m['name'])}</span>" for m in c.models)}</div>
    <p class="cover-lede">Resistance forecasts for {len(c.series)} organism–antibiotic pairs across
    {len(c.organisms)} organisms, {span(c.years)}, from each of the {count_word(n)} models, based on the values
    recorded {span(c.hist)}.</p>
  </div>
</div>
<div class="cover-tiles">{tiles}</div>
<div class="cover-foot"><span>Prepared {today.day} {today:%B %Y}</span><span>Source: {esc(c.source)}</span></div>"""


def accuracy_page(c: Comparison) -> str:
    n = len(c.models)
    row_h = min(9.4, 112.0 / n)
    if not c.beat:
        verdict = f"none of the {count_word(n)} beat repeating the {c.hist[-2]} value ({c.base_mae:.1f} pts)"
    elif len(c.beat) == n:
        verdict = f"all {count_word(n)} beat repeating the {c.hist[-2]} value ({c.base_mae:.1f} pts)"
    else:
        verdict = (f"{count_word(len(c.beat))} of {count_word(n)} beat repeating the {c.hist[-2]} value "
                   f"({c.base_mae:.1f} pts): {esc(names(c.beat))}")
    lede = (f"Each model was trained on {span(c.hist[:-1])} and asked to predict {c.hist[-1]}. "
            f"{esc(c.best['name'])} missed by least, {c.best['mae']:.1f} percentage points on average; {verdict}.")
    swatch = lambda color, text: f'<span><i class="swatch" style="background:{color}"></i>{text}</span>'
    beat_text = f"Beat repeating the {c.hist[-2]} value"
    keys = ([swatch(BLUE, beat_text)] if c.beat else []) + (
        [swatch(PALE, "Did not" if c.beat else f"Did not beat repeating the {c.hist[-2]} value")] if len(c.beat) < n else [])
    body = f"""
<div class="compare">
  <div class="card">
    <h2>How accurate was each?</h2>
    <div class="sub">Average miss when predicting {c.hist[-1]} · percentage points · shorter is better</div>
    {accuracy_chart(c, CHART_W, row_h)}
    <div class="keys">{"".join(keys)}<span>{key("baseline")}Repeating the {c.hist[-2]} value</span></div>
  </div>
  <div class="card">
    <h2>Where each sees resistance heading</h2>
    <div class="sub">Forecast for {c.final}, averaged over all {len(c.series)} pairs · % resistant · most accurate first</div>
    {direction_chart(c, CHART_W, row_h)}
    <div class="keys"><span>{dot_key()}Average {c.final} forecast</span>
    <span>{key("baseline")}Average recorded in {c.hist[-1]}</span></div>
  </div>
</div>{side_by_side(c) if n <= SIDE_BY_SIDE else ""}"""
    return page("Model accuracy", "Which model to trust", lede, body, tail=f"· {count_word(n)} models")


def side_by_side(c: Comparison) -> str:
    """The few models picked in one table: accuracy, against no change, and where their forecasts go."""
    n = len(c.series)
    rows = []
    for i, m in enumerate(c.ranked):
        better = c.base_mae - m["mae"]
        rising = sum(s["final"][m["name"]] - s["history"][-1] >= 0.05 for s in c.series)
        rows.append(
            f'<tr class="{"mine" if i == 0 else ""}"><td class="abx">{esc(m["name"])}</td><td>{esc(m["family"])}</td>'
            f'<td class="num">{m["mae"]:.1f}</td><td class="num">{m["rmse"]:.1f}</td>'
            f'<td class="num">{abs(better):.1f} pts {"better" if better > 0 else "worse"}</td>'
            f'<td class="num">{c.average(m["name"]):.1f}%</td><td class="num">{rising} of {n}</td></tr>')
    return f"""
<div class="card">
  <h2>Side by side</h2>
  <div class="sub">Backtest in percentage points, most accurate first · forecasts for {c.final} over all {n} pairs</div>
  <table class="side">
    <colgroup><col><col style="width:36mm"><col style="width:26mm"><col style="width:22mm"><col style="width:40mm">
    <col style="width:38mm"><col style="width:40mm"></colgroup>
    <thead><tr><th>Model</th><th>Type</th><th class="num">Avg miss</th><th class="num">RMSE</th>
    <th class="num">Versus no change</th><th class="num">Average {c.final}</th>
    <th class="num">Forecast to rise by {c.final}</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>"""


def highlights_page(c: Comparison) -> str:
    y0, y1 = c.hist[-1], c.final
    n = len(c.models)
    best = c.best["name"]
    org_rows = []
    for org in c.organisms:
        sub = [s for s in c.series if s["organism"] == org]
        recorded = sum(s["history"][-1] for s in sub) / len(sub)
        averages = [sum(s["final"][m["name"]] for s in sub) / len(sub) for m in c.models]
        widest = max(sub, key=lambda s: s["spread"])
        org_rows.append(
            f'<tr><td class="abx">{esc(org)}</td><td class="num">{len(sub)}</td>'
            f'<td class="num">{recorded:.1f}%</td>'
            f'<td class="num strong">{sum(s["best"] for s in sub) / len(sub):.1f}%</td>'
            f'<td class="num">{min(averages):.1f}–{max(averages):.1f}%</td>'
            f'<td>{esc(widest["antibiotic"])}</td><td class="num">{widest["spread"]:.1f}</td></tr>')

    item = lambda s, val: (esc(s["antibiotic"]), esc(s["organism"]), val)
    span_of = lambda s: f'{s["lo"]:.1f}–{s["hi"]:.1f}%'
    highest = [item(s, f'{s["best"]:.1f}%<small>all {n}: {span_of(s)}</small>')
               for s in sorted(c.series, key=lambda s: s["best"], reverse=True)[:5]]
    rises = [item(s, f'{arrow(s["jump"])}{signed(s["jump"])} pts<small>{s["rising"]} of {n} models rise</small>')
             for s in sorted(c.series, key=lambda s: s["jump"], reverse=True)[:5]]
    apart = [item(s, f'{span_of(s)}<small>{s["spread"]:.1f} pts apart</small>')
             for s in sorted(c.series, key=lambda s: s["spread"], reverse=True)[:5]]
    body = f"""
<div class="card">
  <h2>By organism</h2>
  <div class="sub">Averages over each organism's antibiotics · % resistant · {esc(best)} is the most accurate of the
  {count_word(n)} models</div>
  <table class="orgs">
    <colgroup><col style="width:62mm"><col style="width:24mm"><col style="width:28mm"><col style="width:32mm">
    <col style="width:38mm"><col><col style="width:24mm"></colgroup>
    <thead><tr><th>Organism</th><th class="num">Antibiotics</th><th class="num">Recorded {y0}</th>
    <th class="num">{esc(short(best))} {y1}</th><th class="num">All {n} models, {y1}</th>
    <th>Most disputed antibiotic</th><th class="num">Spread (pts)</th></tr></thead>
    <tbody>{"".join(org_rows)}</tbody>
  </table>
</div>
<div class="lists">
  {ranked_list(f"Highest forecasts for {y1}", f"{esc(best)} · % resistant, with the range of all {n}", highest)}
  {ranked_list(f"Biggest rises, {y0} → {y1}", f"{esc(best)} · change from the recorded value", rises)}
  {ranked_list("Where the models disagree most", f"Lowest to highest {y1} forecast of the {count_word(n)} models", apart)}
</div>"""
    lede = (f"Where the {count_word(n)} models see resistance heading by {y1}, organism by organism, and where they "
            f"agree and disagree.")
    return page("Key forecasts", f"{y1} at a glance", lede, body, tail=f"· {count_word(n)} models")


def organism_page(c: Comparison, organism, items, idx, part, parts) -> str:
    n = len(c.models)
    best = c.best["name"]
    row_h = min(12.0, 112.0 / max(len(items), 1))
    # The model columns share what the antibiotic (at least 46 mm), recorded and spread columns leave with the
    # trend, which gives way when there are too many models for both.
    with_trend = n <= 8
    model_w = min(20.0, (TABLE_SHARE - (36.0 if with_trend else 0.0)) / n)
    trend_w = min(70.0, TABLE_SHARE - model_w * n) if with_trend else 0.0
    rows = []
    for s in items:
        last = s["history"][-1]
        cells = "".join(f'<td class="num m{" best" if m["name"] == best else ""}">{s["final"][m["name"]]:.1f}</td>'
                        for m in c.models)
        trend = f'<td class="spark">{c.sparkline(s, trend_w - 3, row_h - 1.0)}</td>' if with_trend else ""
        rows.append(f'<tr style="height:{row_h:.2f}mm"><td class="abx">{esc(s["antibiotic"])}</td>{trend}'
                    f'<td class="num">{last:.1f}</td>{cells}<td class="num">{s["spread"]:.1f}</td></tr>')
    legend = (f'<ul class="legend"><li>{key("recorded")}Recorded {span(c.hist)}</li>'
              f'<li>{key("forecast")}{esc(best)} forecast</li><li>{band_key()}Range of the {count_word(n)} models</li></ul>'
              if with_trend else "")
    model_cols = "".join(f'<col style="width:{model_w:.2f}mm">' for _ in c.models)
    head_name = lambda name: NARROW_NAMES.get(name, name) if model_w < 17 else short(name)
    model_heads = "".join(f'<th class="m{" best" if m["name"] == best else ""}">{esc(head_name(m["name"]))}</th>'
                          for m in c.models)
    trend_col = f'<col style="width:{trend_w:.2f}mm">' if with_trend else ""
    trend_head = f'<th rowspan="2">Trend {c.hist[0]} → {c.final}</th>' if with_trend else ""
    body = f"""
<div class="card">
  <div class="card-head">
    <div><h2>Forecast for {c.final} by model</h2>
    <div class="sub">% of isolates resistant · {esc(best)}, the most accurate, shaded · highest {esc(best)} forecast
    first{" · trend on a fixed 0–100% scale" if with_trend else ""}</div></div>
    {legend}
  </div>
  <table class="cmp">
    <colgroup><col>{trend_col}<col style="width:16mm">{model_cols}<col style="width:18mm"></colgroup>
    <thead>
      <tr><th rowspan="2">Antibiotic</th>{trend_head}<th class="group">Recorded</th>
      <th class="group" colspan="{n}">Forecast for {c.final} · % resistant</th>
      <th rowspan="2" class="num">Spread (pts)</th></tr>
      <tr><th class="num">{c.hist[-1]}</th>{model_heads}</tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>"""
    count = len([s for s in c.series if s["organism"] == organism])
    more = f" · part {part + 1} of {parts}" if parts > 1 else ""
    ahead = f"{years_word(c.k)} ahead" if c.k > 1 else "one year ahead"
    lede = f"{count} antibiotics · {count_word(n)} models' forecasts for {c.final}, {ahead}{more}"
    return page(f"Organism {idx + 1} of {len(c.organisms)}", organism, lede, body)


def notes_page(c: Comparison) -> str:
    n = len(c.series)
    rows = "".join(
        f'<tr class="{"mine" if i == 1 else ""}"><td class="num">{i}</td>'
        f'<td class="mname"><b>{esc(m["name"])}</b><small>{esc(m["family"])}</small></td>'
        f'<td class="desc">{esc(m["about"].split(". This data")[0])}</td>'
        f'<td class="num">{m["mae"]:.1f}</td><td class="num">{m["rmse"]:.1f}</td></tr>'
        for i, m in enumerate(c.ranked, start=1))
    rows += (f'<tr class="base"><td class="num">–</td><td class="mname"><b>Repeat the {c.hist[-2]} value</b>'
             f'<small>Baseline</small></td><td class="desc">Predicts no change</td>'
             f'<td class="num">{c.base_mae:.1f}</td><td class="num">{c.base_rmse:.1f}</td></tr>')
    families = {f: [m for m in c.models if m["family"] == f] for f in dict.fromkeys(m["family"] for m in c.models)}
    statistical = families.get("Statistical", [])
    learned = families.get("Machine learning", []) + families.get("Deep learning", [])
    hybrid = families.get("Hybrid", [])
    how = []
    if statistical:
        how.append(f"{esc(names(statistical))} {'was' if len(statistical) == 1 else 'were'} fitted to each series alone")
    if learned:
        how.append(f"{esc(names(learned))} learned the year-to-year change from all {n} series at once")
    if hybrid:
        how.append("SARIMA-LSTM fitted SARIMA to each series and an LSTM to its misses across all series")
    notes = [
        f"<b>Backtest.</b> Every model saw only {span(c.hist[:-1])} and predicted {c.hist[-1]}. Average miss is the "
        f"mean gap to the recorded value in percentage points; RMSE weighs large misses more.",
        f"<b>How they learn.</b> {'; '.join(how)}.",
        f"<b>Short history.</b> {count_word(len(c.hist)).capitalize()} yearly values per series is little to learn "
        f"from, and even the most accurate of these models misses by about {c.best['mae']:.0f} points on average. "
        f"Read the forecasts as a direction, not a precise number.",
        "<b>Agreement is not accuracy.</b> Where the models disagree, the forecast depends on which one you trust; "
        "where they agree, they may still share a blind spot. The backtest is the better guide.",
        (f"<b>Further years are less certain.</b> {c.final} is {count_word(c.k)} steps beyond the data, and each step "
         f"builds on the one before, so errors compound." if c.k > 1 else
         f"<b>One year ahead.</b> {c.final} is one step beyond the data; years further ahead would be less certain."),
        "<b>Capped range.</b> Forecasts are limited to 0–100%.",
    ]
    if c.stand_ins:
        notes.append(f"<b>Too little variation.</b> This data has too little variation for {esc(names(c.stand_ins))} to "
                     f"learn from, so {'it predicts' if len(c.stand_ins) == 1 else 'they predict'} the average yearly "
                     f"change instead.")
    if all_rising(c.series):
        notes.append(f"<b>Upward lean.</b> Every antibiotic in this data rose from {c.hist[0]} to {c.hist[-1]}, which "
                     f"models that follow trends carry forward.")
    if has_early_zeros(c.series):
        notes.append("<b>Zeros.</b> Some antibiotics show exactly 0% in early years, which may mean not tested rather "
                     "than fully susceptible.")
    if c.excluded:
        notes.append(f"<b>Not included.</b> {esc(c.excluded)}")
    body = f"""
<div class="notes-grid wide">
  <div class="card">
    <h2>The {count_word(len(c.models))} models</h2>
    <div class="sub">Trained on {span(c.hist[:-1])}, predicting {c.hist[-1]} · percentage points · most accurate first</div>
    <table class="models">
      <colgroup><col style="width:11mm"><col style="width:38mm"><col>
      <col style="width:16mm"><col style="width:14mm"></colgroup>
      <thead><tr><th class="num">Rank</th><th>Model</th><th>How it works</th>
      <th class="num">Avg miss</th><th class="num">RMSE</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>Good to know</h2>
    <ul class="notes">{"".join(f"<li>{note}</li>" for note in notes)}</ul>
  </div>
</div>"""
    return page("Method & notes", "How to read these forecasts",
                f"Source: {esc(c.source)} · how the models were tested and what the data can and cannot say", body)


# --- Assemble & print ------------------------------------------------------------------

def report_html(c: Comparison) -> tuple[str, int]:
    pages = [cover_page(c), accuracy_page(c), highlights_page(c)]
    for i, org in enumerate(c.organisms):
        items = sorted((s for s in c.series if s["organism"] == org), key=lambda s: s["best"], reverse=True)
        chunks = [items[k:k + ROWS_PER_PAGE] for k in range(0, len(items), ROWS_PER_PAGE)]
        pages += [organism_page(c, org, chunk, i, part, len(chunks)) for part, chunk in enumerate(chunks)]
    pages.append(notes_page(c))

    total = len(pages)
    sections = []
    for number, content in enumerate(pages, start=1):
        footer = ("" if number == 1 else
                  f'<footer class="footer"><span>Model comparison · {esc(c.title)}</span>'
                  f'<span>Page {number} of {total}</span></footer>')
        sections.append(f'<section class="page">{content}{footer}</section>')
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Model comparison · {esc(c.title)}</title>
<style>{CSS}</style></head>
<body>{"".join(sections)}</body></html>"""
    return document, total


def build(engine, year: int, html_path: Path, pdf_path: Path) -> int:
    """Lay out and print the comparison of an engine's two or more models, up to `year`; returns its page count."""
    document, total = report_html(Comparison(engine, year))
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(document, encoding="utf-8")
    print_pdf(find_browser(), html_path, Path(pdf_path))
    return total

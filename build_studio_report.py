"""Build the Studio's PDF report: one model's forecasts for every year up to the one picked.

The Studio shows one model at a time, so unlike the dashboard app's per-model reports (build_model_pdfs.py) this one
never ranks models: the model's accuracy is set against repeating the year before, as on the Studio's forecast page.
It has the same light-blue pages: a cover, how accurate the model was and how it works, the forecasts at a glance for
the last year picked, a page per organism with a column per forecast year, and notes. It is laid out as HTML and
printed to PDF with headless Edge or Chrome.

The desktop app calls build() through exports.write_report_pdf.
"""

import datetime
import math
from pathlib import Path

import numpy as np

from build_forecast_pdf import (all_rising, count_word, esc, excluded_note, find_browser, has_early_zeros, hbar,
                                print_pdf, signed)
from build_model_pdfs import BAND, BLUE, CSS as MODEL_CSS, DETAILS, GRID, INK, MUTED, PALE, arrow, key, page, \
    ranked_list, sparkline
from studio_models import ARMA_MIN_VALUES

ROWS_PER_PAGE = 20  # antibiotics on one organism page; an organism with more continues on the next
TABLE_SHARE = 184.0  # mm of an organism table for the trend and year columns

STUDIO_DETAILS = {
    "SARIMA": "Seasonal ARIMA, fitted to each series with the order that gives the lowest AIC. Yearly data has no "
              "seasonal cycle, so the seasonal terms are switched off. A series with fewer than "
              f"{count_word(ARMA_MIN_VALUES)} yearly values can only be a random walk, with or without drift, since "
              "AR and MA terms would fit its noise; from "
              f"{count_word(ARMA_MIN_VALUES)} values on, AR(1) and MA(1) terms are tried as well.",
    "SARIMA-LSTM": "A hybrid of two models, after Zhang (2003). SARIMA, with its order chosen for each series by AIC, "
                   "forecasts the trend of each series. What it misses within the recorded years is learned by a "
                   "small LSTM network (16 hidden units, averaged over five random starts) from the two previous years "
                   "and the organism. Each forecast year is SARIMA's forecast from the years before it plus the miss "
                   "the LSTM expects.",
}
APPROACH = {
    "statistical": "Fitted to each organism–antibiotic series on its own, using only that series' {years} yearly "
                   "values, and extended {k} ahead.",
    "learned": "{Years} values per series are too few to train on, so the model is trained once across all {n} "
               "series: it learns next year's change from the previous two years and the organism, and adds that "
               "change to the latest value. Each forecast year is fed back in to forecast the next.",
    "hybrid": "SARIMA is fitted to each series on its own. {Years} values per series are too few to train a network "
              "on, so the LSTM is trained once across all {n} series. Each forecast year is fed back in to forecast "
              "the next.",
}

CSS = MODEL_CSS + """
table.years th.group { text-align: center; border-bottom: 0.25mm solid #d5e4f4; padding-bottom: 1.2mm; }
table.years th.yr { text-align: right; padding: 1.4mm 1.6mm; }
table.years td.yr { padding: 0 1.6mm; font-size: 8.4pt; }
table.years td.final { background: #f2f7fd; font-weight: 700; }
table.years th.final { color: #0f2747; }
.stack-left { display: flex; flex-direction: column; gap: 5mm; }
.tested td { height: 8mm; font-size: 8.6pt; }
.tested tr.base td { color: #5d7593; font-style: italic; }
"""


def span(years) -> str:
    years = list(years)
    return str(years[0]) if len(years) == 1 else f"{years[0]}–{years[-1]}"


def years_word(k: int) -> str:
    return f"{count_word(k)} year{'s' if k != 1 else ''}"


def kind(model) -> str:
    return {"Statistical": "statistical", "Hybrid": "hybrid"}.get(model["family"], "learned")


class Report:
    """What every page needs: the one model, the recorded values and its forecasts up to the year picked."""

    def __init__(self, engine, year: int):
        if len(engine.model_names) != 1:
            raise ValueError("The report covers one model")
        meta = engine.meta
        self.model = meta["models"][0]
        self.name = self.model["name"]
        self.hist = list(engine.history_years)
        self.k = year - self.hist[-1]
        self.years = list(range(self.hist[-1] + 1, year + 1))
        self.final = self.years[-1]
        self.recorded = np.asarray(engine.recorded, dtype=float)
        self.forecast = np.asarray(engine.paths[:, 0, :self.k], dtype=float)
        self.series = [{"organism": s["organism"], "antibiotic": s["antibiotic"], "history": self.recorded[i].tolist(),
                        "forecast": self.forecast[i].tolist()} for i, s in enumerate(engine.series)]
        self.organisms = list(dict.fromkeys(s["organism"] for s in self.series))
        misses = self.recorded[:, -1] - self.recorded[:, -2]
        self.base_mae = float(np.abs(misses).mean())
        self.base_rmse = float(np.sqrt((misses ** 2).mean()))
        self.source = meta["workbook"]
        self.excluded = excluded_note(meta.get("excluded") or {})
        self.title = f"Antibiotic resistance forecast {span(self.years)}"
        years = count_word(len(self.hist))
        self.fmt = {"first": self.years[0], "last": self.final, "n": len(self.series), "years": years,
                    "Years": years.capitalize(), "k": years_word(self.k)}

    @property
    def details(self) -> str:
        text = STUDIO_DETAILS.get(self.name) or DETAILS[self.name]
        return text.format(**self.fmt)

    @property
    def stand_in(self) -> str | None:
        """Why a learned model fell back to predicting the average change, if it did."""
        parts = self.model["about"].split(". ", 1)
        return parts[1] if len(parts) == 2 and "too little variation" in parts[1] else None


# --- Charts ----------------------------------------------------------------------------

def accuracy_chart(r: Report, w=141.0) -> str:
    """The model's average miss and RMSE in the backtest, each beside repeating the year before."""
    groups = [("Average miss", "Mean absolute error", r.model["mae"], r.base_mae),
              ("RMSE", "Weighs large misses more", r.model["rmse"], r.base_rmse)]
    top_value = max(v for *_, a, b in groups for v in (a, b))
    step = 5 if top_value <= 25 else 10
    scale_max = step * max(1, math.ceil(top_value / step))
    label_w, bar_h, group_h = 40.0, 4.6, 15.0
    xv = lambda v: label_w + v / scale_max * (w - label_w - 12)
    h = group_h * len(groups) + 7
    parts = []
    for t in range(0, scale_max + 1, step):
        parts.append(f'<line x1="{xv(t):.2f}" x2="{xv(t):.2f}" y1="0" y2="{h - 6}" stroke="{GRID}" stroke-width="0.25"/>')
        parts.append(f'<text x="{xv(t):.2f}" y="{h - 2}" font-size="2.5" fill="{MUTED}" text-anchor="middle">{t} pts</text>')
    for g, (label, sub, mine, base) in enumerate(groups):
        y0 = g * group_h + 1
        parts.append(f'<text x="0" y="{y0 + 4.4:.2f}" font-size="3.1" font-weight="600" fill="{INK}">{label}</text>')
        parts.append(f'<text x="0" y="{y0 + 7.8:.2f}" font-size="2.4" fill="{MUTED}">{sub}</text>')
        for j, (value, fill, weight) in enumerate([(mine, BLUE, 700), (base, PALE, 400)]):
            y = y0 + 1 + j * (bar_h + 1.4)
            parts.append(hbar(xv(0), y, max(xv(value) - xv(0), 0.6), bar_h, 1.0, fill))
            parts.append(f'<text x="{xv(value) + 1.6:.2f}" y="{y + 3.4:.2f}" font-size="3" font-weight="{weight}" '
                         f'fill="{INK}">{value:.1f}</text>')
    return f'<svg width="{w}mm" height="{h:.1f}mm" viewBox="0 0 {w} {h:.1f}" role="img">{"".join(parts)}</svg>'


def average_chart(r: Report, w=141.0, h=52.0) -> str:
    """Average resistance across every series: the recorded years solid, the forecast dashed."""
    hist = r.recorded.mean(axis=0).tolist()
    ahead = r.forecast.mean(axis=0).tolist()
    values, years = hist + ahead, r.hist + r.years
    lo = 10 * math.floor(min(values) / 10)
    hi = max(10 * math.ceil(max(values) / 10), lo + 20)
    left, right, top, bottom = 9.0, 4.0, 6.0, 7.0
    x = lambda i: left + i * (w - left - right) / (len(values) - 1)
    y = lambda v: top + (hi - v) / (hi - lo) * (h - top - bottom)
    pts = lambda idx: " ".join(f"{x(i):.2f},{y(values[i]):.2f}" for i in idx)
    n = len(hist)
    zone = (x(n - 1) + x(n)) / 2
    parts = [f'<rect x="{zone:.2f}" y="{top - 4:.2f}" width="{w - right - zone + 2:.2f}" height="{h - bottom - top + 4:.2f}" '
             f'rx="0.8" fill="{BAND}"/>']
    tick = 10 if hi - lo <= 50 else 20
    for v in range(lo, hi + 1, tick):
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y(v):.2f}" y2="{y(v):.2f}" stroke="{GRID}" stroke-width="0.25"/>')
        parts.append(f'<text x="{left - 1.5}" y="{y(v) + 0.9:.2f}" font-size="2.4" fill="{MUTED}" text-anchor="end">{v}%</text>')
    parts.append(f'<polyline points="{pts(range(n))}" fill="none" stroke="{BLUE}" stroke-width="0.6" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append(f'<polyline points="{pts(range(n - 1, len(values)))}" fill="none" stroke="{BLUE}" stroke-width="0.6" '
                 f'stroke-dasharray="1.4 0.9" stroke-linecap="round"/>')
    for i, (year, v) in enumerate(zip(years, values)):
        filled = i < n
        parts.append(f'<circle cx="{x(i):.2f}" cy="{y(v):.2f}" r="0.85" fill="{BLUE if filled else "#fff"}" '
                     f'stroke="{"#fff" if filled else BLUE}" stroke-width="0.4"/>')
        parts.append(f'<text x="{x(i):.2f}" y="{y(v) - 2:.2f}" font-size="2.4" fill="{INK}" text-anchor="middle" '
                     f'font-weight="{700 if year == r.final else 400}">{v:.1f}</text>')
        parts.append(f'<text x="{x(i):.2f}" y="{h - 2}" font-size="2.4" fill="{MUTED}" text-anchor="middle">{year}</text>')
    return f'<svg width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}" role="img">{"".join(parts)}</svg>'


def tiles_html(tiles) -> str:
    return "".join(f'<div class="tile"><div class="label">{label}</div><div class="value">{value}</div>'
                   f'<div class="foot">{foot}</div></div>' for label, value, foot in tiles)


# --- Pages -----------------------------------------------------------------------------

def versus_baseline(r: Report) -> tuple[str, str]:
    better = r.base_mae - r.model["mae"]
    return (f'{abs(better):.1f}<small> pts {"better" if better > 0 else "worse"}</small>',
            f"Repeating the {r.hist[-2]} value missed by {r.base_mae:.1f} pts")


def cover_page(r: Report) -> str:
    today = datetime.date.today()
    trained = {"statistical": "fitted to each series on its own",
               "learned": f"trained across all {len(r.series)} series",
               "hybrid": f"SARIMA for each series, an LSTM across all {len(r.series)}"}[kind(r.model)]
    value, foot = versus_baseline(r)
    tiles = [
        ("Average miss", f'{r.model["mae"]:.1f}<small> pts</small>',
         f"Trained on {span(r.hist[:-1])}, tested on {r.hist[-1]}"),
        ("Versus no change", value, foot),
        ("Forecasts", f"{len(r.series)}<small> pairs</small>", f"For {span(r.years)}"),
        ("Organisms", f"{len(r.organisms)}", "One page each in this report"),
    ]
    return f"""
<div class="cover-panel">
  <div class="cover-top"><span class="kicker">{esc(r.title)}</span><span class="series">Model report</span></div>
  <div>
    <h1 class="cover-title">{esc(r.name)}</h1>
    <div class="cover-sub">{esc(r.model["family"])} · {esc(trained)}</div>
    <p class="cover-lede">Resistance forecasts for {len(r.series)} organism–antibiotic pairs across
    {len(r.organisms)} organisms, {span(r.years)}, based on the values recorded {span(r.hist)}.</p>
  </div>
</div>
<div class="cover-tiles">{tiles_html(tiles)}</div>
<div class="cover-foot"><span>Prepared {today.day} {today:%B %Y}</span><span>Source: {esc(r.source)}</span></div>"""


def overview_page(r: Report) -> str:
    n = len(r.series)
    learns, predicts = {
        "statistical": (f"Each series on its own ({len(r.hist)} yearly values)", "The next values of each series"),
        "learned": (f"All {n} series at once", "Next year's change, added to the latest value"),
        "hybrid": (f"Each series for SARIMA; all {n} series for the LSTM", "SARIMA's forecast plus the LSTM's correction"),
    }[kind(r.model)]
    facts = [("Learns from", learns), ("Predicts", predicts),
             ("Tested on", f"{r.hist[-1]}, after training on {span(r.hist[:-1])}"),
             ("Forecasts", f"{span(r.years)} for {n} organism–antibiotic pairs")]
    value, foot = versus_baseline(r)
    tiles = [
        ("Average miss", f'{r.model["mae"]:.1f}<small> pts</small>', f"Mean absolute error, {r.hist[-1]} backtest"),
        ("RMSE", f'{r.model["rmse"]:.1f}<small> pts</small>', "Weighs large misses more heavily"),
        ("Versus repeating last year", value, foot),
        ("Forecast horizon", f"{r.k}<small> year{'s' if r.k != 1 else ''}</small>", f"{span(r.years)}"),
    ]
    paragraphs = [r.details, APPROACH[kind(r.model)].format(**r.fmt)]
    if r.stand_in:
        paragraphs.append(f"On this data: {r.stand_in}.")
    lede = (f"Trained on {span(r.hist[:-1])} and asked to predict {r.hist[-1]}, {esc(r.name)} missed the recorded value "
            f"by {r.model['mae']:.1f} percentage points on average; repeating the {r.hist[-2]} value missed by "
            f"{r.base_mae:.1f}.")
    body = f"""
<div class="overview">
  <div class="stack-left">
    <div class="card">
      <h2>How accurate was it?</h2>
      <div class="sub">Miss when predicting {r.hist[-1]} · percentage points · shorter is better</div>
      {accuracy_chart(r)}
      <div class="keys">
        <span><i class="swatch" style="background:{BLUE}"></i>{esc(r.name)}</span>
        <span><i class="swatch" style="background:{PALE}"></i>Repeating the {r.hist[-2]} value</span>
      </div>
    </div>
    <div class="card">
      <h2>Average resistance, all antibiotics</h2>
      <div class="sub">% of isolates resistant, averaged over the {n} pairs · recorded, then {esc(r.name)}'s forecast</div>
      {average_chart(r)}
    </div>
  </div>
  <div class="stack">
    <div class="tiles">{tiles_html(tiles)}</div>
    <div class="card about">
      <h2>How it works</h2>
      {"".join(f"<p>{esc(p)}</p>" for p in paragraphs)}
      <ul class="facts">{"".join(f"<li><b>{k}</b><span>{esc(v)}</span></li>" for k, v in facts)}</ul>
    </div>
  </div>
</div>"""
    return page("Model overview", r.name, lede, body, tail=f"· {r.model['family']}")


def highlights_page(r: Report) -> str:
    y0, y1 = r.hist[-1], r.final
    rows = [{"s": s, "fc": s["forecast"][-1], "jump": s["forecast"][-1] - s["history"][-1]} for s in r.series]

    org_rows = []
    for org in r.organisms:
        sub = [x for x in rows if x["s"]["organism"] == org]
        rising = sum(x["jump"] >= 0.05 for x in sub)
        mean_jump = sum(x["jump"] for x in sub) / len(sub)
        top = max(sub, key=lambda x: x["fc"])
        org_rows.append(
            f'<tr><td class="abx">{esc(org)}</td><td class="num">{len(sub)}</td>'
            f'<td class="num">{rising} of {len(sub)}</td>'
            f'<td class="num">{arrow(mean_jump)}{signed(mean_jump)} pts</td>'
            f'<td>{esc(top["s"]["antibiotic"])}</td><td class="num strong">{top["fc"]:.1f}%</td></tr>')

    item = lambda x, val: (esc(x["s"]["antibiotic"]), esc(x["s"]["organism"]), val)
    change = lambda x: f'{arrow(x["jump"])}{signed(x["jump"])} pts'
    highest = [item(x, f'{x["fc"]:.1f}%') for x in sorted(rows, key=lambda x: x["fc"], reverse=True)[:5]]
    rises = [item(x, change(x)) for x in sorted(rows, key=lambda x: x["jump"], reverse=True)[:5]]
    falls = [x for x in sorted(rows, key=lambda x: x["jump"]) if x["jump"] <= -0.05][:5]
    if falls:
        third = ranked_list(f"Biggest falls, {y0} → {y1}", "Change from the recorded value, percentage points",
                            [item(x, change(x)) for x in falls])
    else:
        lowest = [item(x, f'{x["fc"]:.1f}%') for x in sorted(rows, key=lambda x: x["fc"])[:5]]
        third = ranked_list(f"Lowest forecasts for {y1}", f"% resistant · none is forecast to fall by {y1}", lowest)
    body = f"""
<div class="card">
  <h2>By organism</h2>
  <div class="sub">{esc(r.name)} forecast for {y1} compared with the value recorded in {y0}</div>
  <table class="orgs">
    <colgroup><col style="width:72mm"><col style="width:26mm"><col style="width:38mm"><col style="width:40mm">
    <col><col style="width:26mm"></colgroup>
    <thead><tr><th>Organism</th><th class="num">Antibiotics</th><th class="num">Forecast to rise by {y1}</th>
    <th class="num">Average change {y0}→{str(y1)[2:]}</th><th>Highest {y1} forecast</th><th class="num">% resistant</th></tr></thead>
    <tbody>{"".join(org_rows)}</tbody>
  </table>
</div>
<div class="lists">
  {ranked_list(f"Highest forecasts for {y1}", "% resistant, all organisms", highest)}
  {ranked_list(f"Biggest rises, {y0} → {y1}", "Change from the recorded value, percentage points", rises)}
  {third}
</div>"""
    lede = f"Where {esc(r.name)} sees resistance heading by {y1}, organism by organism, and the forecasts that stand out."
    return page("Key forecasts", r.name, lede, body, tail=f"· {y1} at a glance")


def organism_page(r: Report, organism, items, idx, part, parts) -> str:
    n = len(items)
    row_h = min(12.0, 112.0 / max(n, 1))
    # The trend and the year columns share what the antibiotic (at least 48 mm) and change columns leave.
    year_w = min(22.0, (TABLE_SHARE - 52.0) / (r.k + 1))
    trend_w = min(80.0, TABLE_SHARE - year_w * (r.k + 1))
    rows = []
    for s in items:
        last = s["history"][-1]
        change = s["forecast"][-1] - last
        cells = "".join(f'<td class="num yr{" final" if j == r.k - 1 else ""}">{v:.1f}</td>'
                        for j, v in enumerate(s["forecast"]))
        rows.append(
            f'<tr style="height:{row_h:.2f}mm"><td class="abx">{esc(s["antibiotic"])}</td>'
            f'<td class="spark">{sparkline(s["history"], s["forecast"], trend_w - 3, row_h - 1.0)}</td>'
            f'<td class="num yr">{last:.1f}</td>{cells}'
            f'<td class="num">{arrow(change)}{signed(change)}</td></tr>')
    first, last_year = r.hist[0], r.hist[-1]
    legend = (f'<ul class="legend"><li>{key("recorded")}Recorded {span(r.hist)}</li>'
              f'<li>{key("forecast")}{esc(r.name)} forecast</li><li>{key("zone")}Forecast years</li></ul>')
    year_cols = "".join(f'<col style="width:{year_w:.2f}mm">' for _ in range(r.k + 1))
    year_heads = "".join(f'<th class="num yr{" final" if y == r.final else ""}">{y}</th>' for y in r.years)
    body = f"""
<div class="card">
  <div class="card-head">
    <div><h2>Resistance by antibiotic</h2>
    <div class="sub">% of isolates resistant · trend drawn on a fixed 0–100% scale · highest {r.final} forecast first</div></div>
    {legend}
  </div>
  <table class="years">
    <colgroup><col><col style="width:{trend_w:.2f}mm">{year_cols}<col style="width:26mm"></colgroup>
    <thead>
      <tr><th rowspan="2">Antibiotic</th><th rowspan="2">Trend {first} → {r.final}</th>
      <th class="group">Recorded</th><th class="group" colspan="{r.k}">Forecast · % resistant</th>
      <th rowspan="2" class="num">Change {last_year}→{str(r.final)[2:]} (pts)</th></tr>
      <tr><th class="num yr">{last_year}</th>{year_heads}</tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>"""
    count = len([s for s in r.series if s["organism"] == organism])
    more = f" · part {part + 1} of {parts}" if parts > 1 else ""
    lede = f"{count} antibiotics · {esc(r.name)} forecast for {span(r.years)}{more}"
    return page(f"Organism {idx + 1} of {len(r.organisms)}", organism, lede, body)


def notes_page(r: Report) -> str:
    n = len(r.series)
    how = {"statistical": f"{esc(r.name)} was fitted to each series alone.",
           "learned": f"{esc(r.name)} learned the year-to-year change from all {n} series at once.",
           "hybrid": f"SARIMA was fitted to each series alone, and the LSTM learned SARIMA's misses from all {n} "
                     f"series at once."}[kind(r.model)]
    notes = [
        f"<b>Backtest.</b> {esc(r.name)} saw only {span(r.hist[:-1])} and predicted {r.hist[-1]}. Average miss is the "
        f"mean gap to the recorded value in percentage points; RMSE weighs large misses more.",
        f"<b>How it learns.</b> {how}",
        f"<b>Short history.</b> {count_word(len(r.hist)).capitalize()} yearly values per series is little to learn "
        f"from, and {esc(r.name)} misses by about {r.model['mae']:.0f} points on average. Read the forecasts as a "
        f"direction, not a precise number.",
        (f"<b>Further years are less certain.</b> {r.final} is {count_word(r.k)} steps beyond the data, and each "
         f"step builds on the one before, so errors compound." if r.k > 1 else
         f"<b>One year ahead.</b> {r.final} is one step beyond the data; years further ahead would be less certain."),
        "<b>Capped range.</b> Forecasts are limited to 0–100%.",
    ]
    if r.stand_in:
        notes.append(f"<b>Too little variation.</b> {esc(r.stand_in)}.")
    if all_rising(r.series):
        notes.append(f"<b>Upward lean.</b> Every antibiotic in this data rose from {r.hist[0]} to {r.hist[-1]}, which "
                     f"models that follow trends carry forward.")
    if has_early_zeros(r.series):
        notes.append("<b>Zeros.</b> Some antibiotics show exactly 0% in early years, which may mean not tested rather "
                     "than fully susceptible.")
    if r.excluded:
        notes.append(f"<b>Not included.</b> {esc(r.excluded)}")
    body = f"""
<div class="notes-grid">
  <div class="card">
    <h2>How it was tested</h2>
    <div class="sub">Trained on {span(r.hist[:-1])}, predicting {r.hist[-1]} · percentage points</div>
    <table class="models tested">
      <colgroup><col><col style="width:34mm"><col style="width:22mm"><col style="width:18mm"></colgroup>
      <thead><tr><th>Forecast</th><th>Type</th><th class="num">Avg miss</th><th class="num">RMSE</th></tr></thead>
      <tbody>
        <tr class="mine"><td>{esc(r.name)}</td><td>{esc(r.model["family"])}</td>
        <td class="num">{r.model["mae"]:.1f}</td><td class="num">{r.model["rmse"]:.1f}</td></tr>
        <tr class="base"><td>Repeat the {r.hist[-2]} value</td><td>Baseline</td>
        <td class="num">{r.base_mae:.1f}</td><td class="num">{r.base_rmse:.1f}</td></tr>
      </tbody>
    </table>
    <p class="sub" style="margin-top:3mm">The baseline predicts no change. A model earns its keep by missing by less.</p>
  </div>
  <div class="card">
    <h2>Good to know</h2>
    <ul class="notes">{"".join(f"<li>{note}</li>" for note in notes)}</ul>
  </div>
</div>"""
    return page("Method & notes", "How to read these forecasts",
                f"Source: {esc(r.source)} · how the model was tested and what the data can and cannot say", body)


# --- Assemble & print ------------------------------------------------------------------

def report_html(r: Report) -> tuple[str, int]:
    pages = [cover_page(r), overview_page(r), highlights_page(r)]
    for i, org in enumerate(r.organisms):
        items = sorted((s for s in r.series if s["organism"] == org), key=lambda s: s["forecast"][-1], reverse=True)
        chunks = [items[c:c + ROWS_PER_PAGE] for c in range(0, len(items), ROWS_PER_PAGE)]
        pages += [organism_page(r, org, chunk, i, part, len(chunks)) for part, chunk in enumerate(chunks)]
    pages.append(notes_page(r))

    total = len(pages)
    sections = []
    for number, content in enumerate(pages, start=1):
        footer = ("" if number == 1 else
                  f'<footer class="footer"><span>{esc(r.name)} · {esc(r.title)}</span>'
                  f'<span>Page {number} of {total}</span></footer>')
        sections.append(f'<section class="page">{content}{footer}</section>')
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{esc(r.name)} · {esc(r.title)}</title>
<style>{CSS}</style></head>
<body>{"".join(sections)}</body></html>"""
    return document, total


def build(engine, year: int, html_path: Path, pdf_path: Path) -> int:
    """Lay out and print the report for a one-model engine, up to `year`; returns its page count."""
    document, total = report_html(Report(engine, year))
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(document, encoding="utf-8")
    print_pdf(find_browser(), html_path, Path(pdf_path))
    return total

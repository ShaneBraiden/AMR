"""What the desktop apps save: the forecasts as CSV or Excel, and the PDF reports - the dashboard app's summary and
per-model reports, which compare its 10 models, and the Studio's two: a report on the one model it shows, and one
comparing the models the user picked.

Everything is built from a trained ForecastEngine, so an export always matches the dashboard and the
workbook it was trained on.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import build_forecast_pdf
import build_model_pdfs
import build_studio_compare
import build_studio_report
from forecast_engine import ForecastEngine

KINDS = {
    "csv": "Forecasts (CSV)",
    "excel": "Forecasts (Excel)",
    "summary_pdf": "Summary report (PDF)",
    "model_pdfs": "Per-model reports (PDF)",
    "report_pdf": "Report (PDF)",  # the Studio's: the one model it shows
    "compare_pdf": "Comparison report (PDF)",  # the Studio's: the models the user picked
}


def forecast_table(engine: ForecastEngine, year: int, column: str = "forecast_{}") -> pd.DataFrame:
    """One row per organism, antibiotic and model, with a column per forecast year up to `year`."""
    years = range(engine.first_year, year + 1)
    return pd.DataFrame([
        {"organism": s["organism"], "antibiotic": s["antibiotic"], "model": model,
         **{column.format(y): float(engine.paths[i, k, j]) for j, y in enumerate(years)}}
        for i, s in enumerate(engine.series)
        for k, model in enumerate(engine.model_names)
    ])


def recorded_table(engine: ForecastEngine) -> pd.DataFrame:
    return pd.DataFrame([
        {"organism": s["organism"], "antibiotic": s["antibiotic"],
         **{f"{y} (%)": float(engine.recorded[i, j]) for j, y in enumerate(engine.history_years)}}
        for i, s in enumerate(engine.series)
    ])


def accuracy_table(engine: ForecastEngine) -> pd.DataFrame:
    meta = engine.meta
    years = engine.history_years
    rows = [
        {"model": m["name"], "type": m["family"], "avg miss (pts)": m["mae"], "rmse (pts)": m["rmse"],
         "most accurate": "yes" if m["name"] == meta["bestModel"] else "", "how it works": m["about"]}
        for m in sorted(meta["models"], key=lambda m: m["mae"])
    ]
    rows.append({"model": f"Repeat the {years[-2]} value", "type": "Baseline", "avg miss (pts)": meta["baselineMae"],
                 "how it works": "For comparison: predicts no change"})
    table = pd.DataFrame(rows)
    table.attrs["note"] = (f"Backtest: each model was trained on {years[0]}–{years[-2]} and predicted {years[-1]}. "
                           f"Avg miss is the mean absolute gap to the recorded value, in percentage points.")
    return table


def write_csv(engine: ForecastEngine, year: int, path: Path) -> Path:
    forecast_table(engine, year).to_csv(path, index=False, encoding="utf-8-sig")  # the BOM makes Excel read UTF-8
    return path


def write_excel(engine: ForecastEngine, year: int, path: Path) -> Path:
    sheets = {
        "Forecasts": forecast_table(engine, year, "{} (%)"),
        "Recorded": recorded_table(engine),
        "Model accuracy": accuracy_table(engine),
    }
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for name, table in sheets.items():
            table.to_excel(xl, sheet_name=name, index=False)
            ws = xl.sheets[name]
            ws.freeze_panes = "A2"
            for col, header in enumerate(table.columns, start=1):
                letter = ws.cell(row=1, column=col).column_letter
                values = table[header].astype(str).tolist()
                ws.column_dimensions[letter].width = min(60, max(len(str(header)), *map(len, values)) + 2)
                if table[header].dtype.kind == "f":
                    for (cell,) in ws.iter_rows(min_row=2, min_col=col, max_col=col):
                        cell.number_format = "0.0"
        ws = xl.sheets["Model accuracy"]
        ws.cell(row=len(sheets["Model accuracy"]) + 3, column=1, value=sheets["Model accuracy"].attrs["note"])
    return path


def report_data(engine: ForecastEngine) -> tuple:
    """(models, best, baseline, series) for the PDF builders: the two years after the last recorded one."""
    recorded = engine.recorded
    baseline = float(np.abs(recorded[:, -1] - recorded[:, -2]).mean())
    series = [
        {"organism": s["organism"], "antibiotic": s["antibiotic"], "history": recorded[i].tolist(),
         "forecast": {name: tuple(engine.paths[i, k, :2].tolist()) for k, name in enumerate(engine.model_names)}}
        for i, s in enumerate(engine.series)
    ]
    return [dict(m) for m in engine.meta["models"]], engine.meta["bestModel"], baseline, series


def configure_reports(engine: ForecastEngine) -> None:
    build_model_pdfs.configure(engine.history_years, engine.meta["workbook"], engine.meta["excluded"])


def write_summary_pdf(engine: ForecastEngine, path: Path, work_dir: Path) -> Path:
    configure_reports(engine)
    with tempfile.TemporaryDirectory(dir=work_dir, ignore_cleanup_errors=True) as tmp:
        build_forecast_pdf.build(report_data(engine), Path(tmp) / "forecast_report.html", Path(path))
    return path


def write_model_pdfs(engine: ForecastEngine, folder: Path, work_dir: Path) -> list[Path]:
    """One PDF per model in `folder`; built aside first, so a failure halfway leaves no partial set."""
    configure_reports(engine)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_dir, ignore_cleanup_errors=True) as tmp:
        built = build_model_pdfs.build(report_data(engine), Path(tmp) / "html", Path(tmp) / "pdf")
        written = []
        for pdf, _pages in built:
            target = folder / pdf.name
            shutil.move(pdf, target)
            written.append(target)
    return written


def write_report_pdf(engine: ForecastEngine, year: int, path: Path, work_dir: Path) -> Path:
    """The Studio's report on a one-model engine, with its forecasts up to `year`."""
    with tempfile.TemporaryDirectory(dir=work_dir, ignore_cleanup_errors=True) as tmp:
        build_studio_report.build(engine, year, Path(tmp) / "report.html", Path(path))
    return path


def write_compare_pdf(engine: ForecastEngine, year: int, path: Path, work_dir: Path) -> Path:
    """The Studio's report comparing the models of an engine with two or more, with their forecasts up to `year`."""
    with tempfile.TemporaryDirectory(dir=work_dir, ignore_cleanup_errors=True) as tmp:
        build_studio_compare.build(engine, year, Path(tmp) / "comparison.html", Path(path))
    return path

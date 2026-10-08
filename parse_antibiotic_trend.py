"""Parse 'antibiotic trend (2022-2025).xlsx' into tidy, analysis-ready files.

Each sheet in the workbook is one organism, laid out as:
    row 1  title (merged)
    row 2  subtitle (merged)
    row 4  header: Antibiotic | 2022 (%) | 2023 (%) | ... | Net Change (%)
    row 5+ one row per antibiotic, values stored as fractions (0.25 = 25%)

The year columns are read from the header, so a workbook with more years (say 2022-2026) works too, as
long as every sheet has the same consecutive years. read_workbook() raises WorkbookError, with a message
saying what to fix, for anything else.

Outputs (written to ./output):
    resistance_long.csv  one row per organism x antibiotic x year (tidy format)
    resistance_wide.csv  one row per organism x antibiotic, a column per year
    resistance.json      nested by organism
"""

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException

SOURCE = Path(__file__).parent / "antibiotic trend (2022-2025).xlsx"
OUT_DIR = Path(__file__).parent / "output"
HEADER_SEARCH_ROWS = 10
YEAR_HEADER = re.compile(r"^\s*(\d{4})\s*\(%\)\s*$")
NET_CHANGE_HEADER = "net change (%)"
MIN_YEARS = 4  # the learned models need two lag years, plus one held back for the backtest

# Spelling variants in the source mapped to the standard drug name.
# Names not listed here are kept as written.
ANTIBIOTIC_NAMES = {
    "Amoxyclav": "Amoxicillin-clavulanate",
    "Cefaprazone with sulnactum": "Cefoperazone-sulbactam",
    "Cefaperazone with sulbactum": "Cefoperazone-sulbactam",
    "Cephotaxime": "Cefotaxime",
    "Gentamycin": "Gentamicin",
    "Meropenam": "Meropenem",
    "Nitrofurantion": "Nitrofurantoin",
    "Piperacillin with Tazobactum": "Piperacillin-tazobactam",
}


ORGANISM_NAMES = {
    "E.COLI": "Escherichia coli",
    "KLEBSIELLA PNEUMONIAE": "Klebsiella pneumoniae",
    "COAG.NEG.STAPHYLOCOCCUS": "Coagulase-negative Staphylococcus",
    "ACINETOBACTER SPECIES": "Acinetobacter spp.",
    "STAPHYLOCOCCUS AUREUS": "Staphylococcus aureus",
    "STREPTOCOCCUS SPECIES": "Streptococcus spp.",
    "ENTEROCOCCUS": "Enterococcus spp.",
    "PSEUDOMONAS AERUGINOSA": "Pseudomonas aeruginosa",
}


class WorkbookError(ValueError):
    """The workbook does not have the layout this reads; the message says what to fix."""


@dataclass
class Workbook:
    wide: pd.DataFrame  # one row per organism x antibiotic: organism, antibiotic, antibiotic_raw, pct_<year>..., net_change_pct
    notes: dict  # organism -> note, for sheets with a text row in place of data (e.g. Streptococcus)
    years: list  # the recorded years, consecutive
    organisms: list  # every sheet's organism, in workbook order


def organism_name(sheet_name: str) -> str:
    """The organism a sheet is about: a known short name, an ALL-CAPS name in title case, else as written."""
    name = re.sub(r"\s+", " ", sheet_name).strip()
    return ORGANISM_NAMES.get(name, name.title() if name.isupper() else name)


def find_header(ws) -> tuple[int, list]:
    for number, row in enumerate(ws.iter_rows(max_row=HEADER_SEARCH_ROWS, values_only=True), start=1):
        if row and isinstance(row[0], str) and row[0].strip().lower() == "antibiotic":
            return number, list(row)
    raise WorkbookError(f"Sheet '{ws.title}': no header row starting with 'Antibiotic' in the first "
                        f"{HEADER_SEARCH_ROWS} rows.")


def year_columns(ws, header) -> tuple[list[int], int | None]:
    """The years in the header, which must be consecutive, and the Net Change column if there is one."""
    years = []
    for cell in header[1:]:
        match = YEAR_HEADER.match(str(cell)) if cell is not None else None
        if not match:
            break
        years.append(int(match.group(1)))
    if not years:
        raise WorkbookError(f"Sheet '{ws.title}': the columns after 'Antibiotic' should be years written "
                            f"like '2022 (%)'.")
    if years != list(range(years[0], years[0] + len(years))):
        raise WorkbookError(f"Sheet '{ws.title}': the year columns ({', '.join(map(str, years))}) must be "
                            f"consecutive years in order.")
    after = 1 + len(years)
    net = after if after < len(header) and str(header[after]).strip().lower() == NET_CHANGE_HEADER else None
    return years, net


def parse_sheet(ws) -> tuple[list[dict], str | None, list[int]]:
    header_row, header = find_header(ws)
    years, net_col = year_columns(ws, header)

    rows, note = [], None
    for number, cells in enumerate(ws.iter_rows(min_row=header_row + 1, values_only=True), start=header_row + 1):
        cells = list(cells) + [None] * (len(header) - len(cells))
        name, values = cells[0], cells[1: 1 + len(years)]
        if name is None or str(name).strip() == "":
            continue
        # A row with text but no numbers is a note (e.g. Streptococcus has no data).
        if all(v is None for v in values):
            note = str(name).strip()
            continue
        raw = str(name).strip()
        for year, v in zip(years, values):
            where = f"Sheet '{ws.title}', row {number} ({raw}), {year}"
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise WorkbookError(f"{where}: expected a percentage but found {'an empty cell' if v is None else repr(v)}.")
            if not 0 <= v <= 1:
                raise WorkbookError(f"{where}: {v} is outside 0–100%. Values should be Excel percentages "
                                    f"(a cell showing 25% stores 0.25).")
        net = cells[net_col] if net_col is not None else None
        rows.append(
            {
                "antibiotic": ANTIBIOTIC_NAMES.get(raw, raw),
                "antibiotic_raw": raw,
                **dict(zip(years, values)),
                "net_change": net if isinstance(net, (int, float)) else values[-1] - values[0],
            }
        )
    return rows, note, years


def load(path: Path):
    """Open a workbook for reading its values, or raise WorkbookError."""
    try:
        return openpyxl.load_workbook(path, data_only=True)
    except (InvalidFileException, zipfile.BadZipFile, KeyError, OSError) as e:
        raise WorkbookError(f"Could not open '{Path(path).name}' as an Excel workbook (.xlsx): {e}") from e


def read_workbook(path: Path) -> Workbook:
    """Read and check the workbook; writes nothing."""
    wb = load(path)
    wide_records, notes, organisms, years = [], {}, [], None
    for ws in wb.worksheets:
        organism = organism_name(ws.title)
        rows, note, sheet_years = parse_sheet(ws)
        if years is None:
            years = sheet_years
        elif sheet_years != years:
            raise WorkbookError(f"Sheet '{ws.title}' has years {sheet_years[0]}–{sheet_years[-1]} but the first sheet "
                                f"has {years[0]}–{years[-1]}; every sheet needs the same years.")
        organisms.append(organism)
        if note:
            notes[organism] = note
        seen = set()
        for r in rows:
            if r["antibiotic"] in seen:
                raise WorkbookError(f"Sheet '{ws.title}': {r['antibiotic']} is listed twice.")
            seen.add(r["antibiotic"])
            wide_records.append({"organism": organism, **r})

    if not wide_records:
        raise WorkbookError("The workbook has no antibiotic rows with data.")
    if len(years) < MIN_YEARS:
        raise WorkbookError(f"The workbook has {len(years)} years ({years[0]}–{years[-1]}); the models need at "
                            f"least {MIN_YEARS} consecutive years.")

    wide = pd.DataFrame(wide_records)
    pct_cols = years + ["net_change"]
    wide[pct_cols] = (wide[pct_cols].astype(float) * 100).round(2)
    wide = wide.rename(columns={y: f"pct_{y}" for y in years} | {"net_change": "net_change_pct"})
    return Workbook(wide=wide, notes=notes, years=years, organisms=organisms)


def main() -> None:
    data = read_workbook(SOURCE)
    wide, notes, years = data.wide, data.notes, data.years
    pct = [f"pct_{y}" for y in years]

    # Sanity check: net change in the source should equal last year - first year.
    mismatch = (wide["net_change_pct"] - (wide[pct[-1]] - wide[pct[0]])).abs() > 0.05
    if mismatch.any():
        print(f"WARNING: net change != {years[-1]} - {years[0]} for:")
        print(wide.loc[mismatch, ["organism", "antibiotic", "net_change_pct"]])

    long = wide.melt(
        id_vars=["organism", "antibiotic", "antibiotic_raw"],
        value_vars=pct,
        var_name="year",
        value_name="resistance_pct",
    )
    long["year"] = long["year"].str.removeprefix("pct_").astype(int)
    long = long.sort_values(["organism", "antibiotic", "year"]).reset_index(drop=True)

    OUT_DIR.mkdir(exist_ok=True)
    wide.to_csv(OUT_DIR / "resistance_wide.csv", index=False)
    long.to_csv(OUT_DIR / "resistance_long.csv", index=False)

    nested = {}
    for organism in data.organisms:
        sub = wide[wide["organism"] == organism]
        nested[organism] = {
            "note": notes.get(organism),
            "antibiotics": [
                {
                    "antibiotic": r["antibiotic"],
                    "antibiotic_raw": r["antibiotic_raw"],
                    "resistance_pct": {str(y): r[f"pct_{y}"] for y in years},
                    "net_change_pct": r["net_change_pct"],
                }
                for r in sub.to_dict("records")
            ],
        }
    (OUT_DIR / "resistance.json").write_text(json.dumps(nested, indent=2), encoding="utf-8")

    print(f"Parsed {len(data.organisms)} sheets, {len(wide)} organism-antibiotic rows, {years[0]}-{years[-1]}")
    print(wide.groupby("organism", sort=False).size().to_string())
    for org, note in notes.items():
        print(f"Note - {org}: {note}")


if __name__ == "__main__":
    main()

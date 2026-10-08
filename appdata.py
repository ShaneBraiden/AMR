"""Where the app keeps its files: the active workbook and the cache of trained models.

Two apps share this code, each a profile with its own name, page and data folder:

  forecast  Antibiotic Resistance Forecast (desktop.py): opens on the bundled workbook; page static/index.html
  entry     Antibiotic Resistance Forecast Studio (entry_desktop.py): starts empty, and the user uploads or types
            in the values and confirms them; page static/studio.html

The forecast profile is the default; entry_desktop.py (or app.py --profile entry) switches with use_profile()
before anything else runs.

User data lives in %LOCALAPPDATA%\\<app name> (or ARF_DATA_DIR, if set):

  workbook/<key>.xlsx  a copy of each workbook loaded, trained from and kept for Retrain
  workbook/state.json  the active workbook: its file name, where it was loaded from, and its cache key
  cache/<key>/         trained models for one workbook (ForecastEngine.save), the newest KEEP_CACHES kept
  session.json         entry profile: the step the user was on and the data they were editing
  logs/app.log         the desktop app's output
  tmp/                 scratch space for building PDF reports and writing confirmed data sets

Read-only resources (the pages, the bundled workbook, the seed cache trained at build time) sit next to this
file, or inside the bundle when the app is frozen with PyInstaller.
"""

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from forecast_engine import ForecastEngine, workbook_key

RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
DEFAULT_WORKBOOK = RESOURCE_DIR / "antibiotic trend (2022-2025).xlsx"
SEED_CACHE = RESOURCE_DIR / "seed_cache"
KEEP_CACHES = 5

PROFILES = {
    "forecast": {"name": "Antibiotic Resistance Forecast", "page": "index.html", "bundled": True},
    "entry": {"name": "Antibiotic Resistance Forecast Studio", "page": "studio.html", "bundled": False},
}


def use_profile(profile: str) -> None:
    """Switch to an app's name, page and data folder; call it before anything else reads them."""
    global PROFILE, APP_NAME, BUNDLED, PAGE, DATA_DIR, WORKBOOK_DIR, CACHE_DIR, LOG_FILE, TMP_DIR, STATE_FILE
    global SESSION_FILE
    settings = PROFILES[profile]
    PROFILE, APP_NAME, BUNDLED = profile, settings["name"], settings["bundled"]
    PAGE = RESOURCE_DIR / "static" / settings["page"]
    DATA_DIR = Path(os.environ.get("ARF_DATA_DIR") or Path(os.environ.get("LOCALAPPDATA") or Path.home()) / APP_NAME)
    WORKBOOK_DIR = DATA_DIR / "workbook"
    CACHE_DIR = DATA_DIR / "cache"
    LOG_FILE = DATA_DIR / "logs" / "app.log"
    TMP_DIR = DATA_DIR / "tmp"
    STATE_FILE = WORKBOOK_DIR / "state.json"
    SESSION_FILE = DATA_DIR / "session.json"


use_profile("forecast")


@dataclass(frozen=True)
class Workbook:
    name: str  # file name shown in the app
    source: str  # where it was loaded from; Retrain re-reads it from there, to pick up edits
    key: str  # workbook_key of the contents

    @property
    def copy(self) -> Path:
        return WORKBOOK_DIR / f"{self.key}.xlsx"

    def current_source(self) -> Path:
        """The original file if it is still there, else the copy taken when it was loaded."""
        source = Path(self.source)
        return source if source.is_file() else self.copy


def ensure_dirs() -> None:
    for d in (WORKBOOK_DIR, CACHE_DIR, LOG_FILE.parent, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)


def read_session() -> dict:
    """What the entry profile's page saved last: the step the user was on and the data they were editing."""
    try:
        session = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        return session if isinstance(session, dict) else {}
    except (OSError, ValueError):
        return {}


def write_session(session: dict) -> None:
    """Save the session whole, through a temporary file, so a crash never leaves half of it."""
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    partial = SESSION_FILE.with_name(SESSION_FILE.name + ".partial")
    partial.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")
    partial.replace(SESSION_FILE)


def remember(path: Path, name: str | None = None, key: str | None = None) -> Workbook:
    """Copy a workbook into the data folder, under its cache key: the hash of the file, unless given one."""
    path = Path(path)
    wb = Workbook(name=name or path.name, source=str(path.resolve()), key=key or workbook_key(path))
    if not wb.copy.exists():
        WORKBOOK_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, wb.copy)
    return wb


def active_workbook() -> Workbook | None:
    """The workbook the app last opened; on first launch the bundled one, or None in an app without one."""
    try:
        wb = Workbook(**json.loads(STATE_FILE.read_text(encoding="utf-8")))
        if wb.copy.exists():
            return wb
    except (OSError, ValueError, TypeError):
        pass
    return remember(DEFAULT_WORKBOOK) if BUNDLED else None


def set_active(wb: Workbook) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(asdict(wb), indent=2), encoding="utf-8")


def cached_engine(key: str) -> ForecastEngine | None:
    """The trained models for a workbook from this user's cache or the seed cache, if either has them."""
    for folder in (CACHE_DIR / key, SEED_CACHE / key):
        if (folder / "meta.json").is_file():
            engine = ForecastEngine.from_cache(folder)
            if folder.parent == CACHE_DIR:
                os.utime(folder)  # recently used caches survive pruning
            return engine
    return None


def store(engine: ForecastEngine, key: str, keep: set[str] = frozenset()) -> None:
    """Save trained models, then drop all but the newest KEEP_CACHES caches and their workbook copies."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    engine.save(CACHE_DIR / key)
    folders = sorted((f for f in CACHE_DIR.iterdir() if f.is_dir()), key=lambda f: f.stat().st_mtime, reverse=True)
    kept = {f.name for f in folders[:KEEP_CACHES]} | set(keep) | {key}
    for f in folders:
        if f.name not in kept:
            shutil.rmtree(f, ignore_errors=True)
    for copy in WORKBOOK_DIR.glob("*.xlsx"):
        if copy.stem not in kept:
            copy.unlink(missing_ok=True)

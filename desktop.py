"""Antibiotic Resistance Forecast - the desktop app.

Runs the dashboard server (app.py) on a free local port and shows it in the app's own window (Microsoft
Edge WebView2), with native dialogs for opening a workbook and saving exports (exports.py). The Studio app
(entry_desktop.py) is this same program in the entry profile (appdata.py).

Run:  python desktop.py                     the app
      python desktop.py --self-test         train on the bundled workbook, check against the seed cache,
                                            write every export, exit 0 if all is well (used by the build);
                                            --workbook and --seed-cache test another workbook and cache
      python desktop.py --build-cache DIR   train on the bundled workbook and save the models in DIR/<key>,
                                            the seed cache the packaged app opens on first launch
"""

import argparse
import ctypes
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path

import appdata

FROZEN = getattr(sys, "frozen", False)
WEBVIEW2_DOWNLOAD = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"
WEBVIEW2_CLIENT = r"Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
MAX_LOG_BYTES = 5_000_000
BACKGROUNDS = {"forecast": ("#f9f9f7", "#0d0d0d"), "entry": ("#f4f6f8", "#0f1318")}  # each page's light, dark
FILE_TYPES = {".csv": "CSV file (*.csv)", ".xlsx": "Excel workbook (*.xlsx)", ".pdf": "PDF document (*.pdf)"}


def log_to_file() -> None:
    """Send output to the log file: the app has no console, and uvicorn fails on a missing stdout."""
    appdata.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if appdata.LOG_FILE.exists() and appdata.LOG_FILE.stat().st_size > MAX_LOG_BYTES:
        appdata.LOG_FILE.replace(appdata.LOG_FILE.with_suffix(".log.1"))
    sys.stdout = sys.stderr = open(appdata.LOG_FILE, "a", encoding="utf-8", buffering=1)
    print(f"\n--- {appdata.APP_NAME} started {datetime.now():%Y-%m-%d %H:%M:%S}")


def hide_console_windows() -> None:
    """Start child processes without a console window of their own.

    Prophet runs its Stan program once per series (190 times a training) and Edge prints the PDFs; from an
    app without a console, each would flash up a console window.
    """
    if sys.platform != "win32":
        return
    original = subprocess.Popen.__init__

    def __init__(self, *args, creationflags=0, **kwargs):
        original(self, *args, creationflags=creationflags | subprocess.CREATE_NO_WINDOW, **kwargs)

    subprocess.Popen.__init__ = __init__


def webview2_installed() -> bool:
    import winreg

    for root, path in ((winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\{WEBVIEW2_CLIENT}"),
                       (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\{WEBVIEW2_CLIENT}"),
                       (winreg.HKEY_CURRENT_USER, rf"SOFTWARE\{WEBVIEW2_CLIENT}")):
        try:
            with winreg.OpenKey(root, path) as key:
                version, _ = winreg.QueryValueEx(key, "pv")
                if version and version != "0.0.0.0":
                    return True
        except OSError:
            pass
    return False


def ask_for_webview2() -> None:
    MB_YESNO, MB_ICONWARNING, IDYES = 0x4, 0x30, 6
    answer = ctypes.windll.user32.MessageBoxW(
        None, f"{appdata.APP_NAME} needs the Microsoft Edge WebView2 Runtime, which is not installed on this PC.\n\n"
              "Open the Microsoft download page now? Install it, then start the app again.",
        appdata.APP_NAME, MB_YESNO | MB_ICONWARNING)
    if answer == IDYES:
        webbrowser.open(WEBVIEW2_DOWNLOAD)


def windows_dark_mode() -> bool:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def window_background() -> str:
    """The page's background, so the window doesn't flash the other theme while the page loads."""
    light, dark = BACKGROUNDS[appdata.PROFILE]
    theme = appdata.read_session().get("theme") if appdata.PROFILE == "entry" else None
    return dark if theme == "dark" or (theme != "light" and windows_dark_mode()) else light


def fits_on_screen(width: int, height: int) -> bool:
    """Whether a window this big (in logical pixels, title bar included) fits the primary screen's work area.

    The work area and the DPI are both scaled or both not, depending on the process's DPI awareness, so their
    ratio is in logical pixels either way.
    """
    from ctypes import wintypes

    SPI_GETWORKAREA = 0x0030
    user32 = ctypes.windll.user32
    area = wintypes.RECT()
    if not user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(area), 0):
        return True
    scale = user32.GetDpiForSystem() / 96
    return width * scale <= area.right - area.left and height * scale <= area.bottom - area.top


def start_server() -> str:
    """Serve app.py on a free local port in the background; returns its URL once it is listening."""
    import uvicorn

    from app import app

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, name="server", daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("The dashboard server did not start; see the log above.")
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}"


def default_name(engine, kind: str, year: int, models: list[str] = ()) -> str:
    y1, y2 = engine.first_year, engine.first_year + 1
    span = str(year) if year == y1 else f"{y1}-{year}"
    years = span
    if models:
        years += f" {models[0]}" if len(models) == 1 else f" {len(models)} models"
    return {
        "csv": f"Resistance forecasts {years}.csv",
        "excel": f"Resistance forecasts {years}.xlsx",
        "summary_pdf": f"Resistance forecast report {y1}-{y2}.pdf",
        "model_pdfs": f"Resistance forecast model reports {y1}-{y2}",
        "report_pdf": f"Resistance forecast report {years}.pdf",
        "compare_pdf": f"Resistance forecast model comparison {span}.pdf",
    }[kind]


def model_list(models) -> list[str]:
    """The Studio's models for an export, sent by the page as one name or a list."""
    return [models] if isinstance(models, str) else [str(m) for m in models or []]


class DesktopApi:
    """What the page can ask of the app, as window.pywebview.api.<method>(...); each call runs on its own thread.

    Names starting with _ stay hidden from the page.
    """

    def __init__(self):
        self._window = None
        self._exporting = threading.Lock()
        self._chosen = set()  # exports only go where the user picked in a dialog
        self._saved = set()  # and only files this app wrote can be opened from the page

    def choose_workbook(self) -> dict:
        import webview

        from app import trainer

        picked = self._window.create_file_dialog(webview.FileDialog.OPEN, file_types=("Excel workbook (*.xlsx)",))
        if not picked:
            return {"ok": False, "cancelled": True}
        try:
            return trainer.open_workbook(Path(picked[0]))
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": f"Could not open the workbook: {e}"}

    def read_workbook_for_review(self) -> dict:
        """The Studio: ask for a workbook and read its values for review; nothing is trained until confirmed."""
        import webview

        from app import review

        picked = self._window.create_file_dialog(webview.FileDialog.OPEN, file_types=("Excel workbook (*.xlsx)",))
        if not picked:
            return {"ok": False, "cancelled": True}
        try:
            return review(Path(picked[0]))
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": f"Could not read the workbook: {e}"}

    def save_data_excel(self, dataset: dict) -> dict:
        """The Studio: save a data set as a workbook in the layout it reads, where the user picks."""
        import datasets

        if datasets.problems(dataset):
            return {"ok": False, "error": "Fix the problems marked in the review first."}
        name = re.sub(r'[<>:"/\\|?*]', "-", (dataset.get("name") or "").strip()) or "Resistance data"
        target = self._ask_save_path(f"{name}.xlsx")
        if target is None:
            return {"ok": False, "cancelled": True}
        try:
            datasets.write_workbook(dataset, target)
        except PermissionError:
            traceback.print_exc()
            return {"ok": False, "error": f"Could not write {target.name}. If it is open in another program, close it "
                                          f"and try again, or save it somewhere else."}
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": f"Could not save the data: {e}"}
        self._saved.add(str(target))
        print(f"Saved the data set to {target}")
        return {"ok": True, "path": str(target)}

    def _ask_save_path(self, name: str) -> Path | None:
        """A Save dialog for a file named like `name`; the chosen path keeps name's extension."""
        import webview

        suffix = Path(name).suffix
        picked = self._window.create_file_dialog(webview.FileDialog.SAVE, save_filename=name,
                                                 file_types=(FILE_TYPES[suffix],))
        target = Path(picked[0] if isinstance(picked, (tuple, list)) else picked) if picked else None
        if target and target.suffix.lower() != suffix:
            target = target.with_name(target.name + suffix)
        return target

    def _engine(self, year: int, models: list[str]):
        """The trained engine, or with models (the Studio) just their forecasts, and the year in range."""
        from app import trainer

        engine = trainer.engine
        if engine and models:
            engine = engine.only(models) if set(models) <= set(engine.model_names) else None
        return engine, (min(max(int(year), engine.first_year), engine.max_year) if engine else None)

    def choose_export_path(self, kind: str, year: int, models: str | list[str] | None = None) -> dict:
        """Ask where to save an export: a file, or for the per-model reports a folder to create one in."""
        import webview

        import exports

        models = model_list(models)
        engine, year = self._engine(year, models)
        if engine is None or kind not in exports.KINDS:
            return {"ok": False, "error": "There is nothing to export until the models are trained."}
        if kind == "report_pdf" and len(models) != 1:
            return {"ok": False, "error": "Pick a model to report on first."}
        if kind == "compare_pdf" and len(models) < 2:
            return {"ok": False, "error": "Pick at least two models to compare."}
        name = default_name(engine, kind, year, models)
        if kind == "model_pdfs":
            picked = self._window.create_file_dialog(webview.FileDialog.FOLDER)
            target = Path(picked[0]) / name if picked else None
        else:
            target = self._ask_save_path(name)
        if target is None:
            return {"ok": False, "cancelled": True}
        self._chosen.add(str(target))
        return {"ok": True, "path": str(target)}

    def export(self, kind: str, year: int, path: str, models: str | list[str] | None = None) -> dict:
        """Write an export to the path choose_export_path returned."""
        import exports

        engine, year = self._engine(year, model_list(models))
        if engine is None or kind not in exports.KINDS or path not in self._chosen:
            return {"ok": False, "error": "Choose where to save the export first."}
        target = Path(path)
        if not self._exporting.acquire(blocking=False):
            return {"ok": False, "error": "Another export is still being written. Try again when it has finished."}
        try:
            if kind == "csv":
                exports.write_csv(engine, year, target)
            elif kind == "excel":
                exports.write_excel(engine, year, target)
            elif kind == "summary_pdf":
                exports.write_summary_pdf(engine, target, appdata.TMP_DIR)
            elif kind == "report_pdf":
                exports.write_report_pdf(engine, year, target, appdata.TMP_DIR)
            elif kind == "compare_pdf":
                exports.write_compare_pdf(engine, year, target, appdata.TMP_DIR)
            else:
                exports.write_model_pdfs(engine, target, appdata.TMP_DIR)
        except PermissionError:
            traceback.print_exc()
            return {"ok": False, "error": f"Could not write {target.name}. If it is open in another program, close it "
                                          f"and try again, or save it somewhere else."}
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": f"The export failed: {e}"}
        finally:
            self._exporting.release()
        self._saved.add(str(target))
        print(f"Exported {exports.KINDS[kind]} to {target}")
        return {"ok": True, "path": str(target), "label": exports.KINDS[kind]}

    def open_path(self, path: str) -> bool:
        """Open an exported file, or show an exported folder, in Windows."""
        if path not in self._saved or not os.path.exists(path):
            return False
        os.startfile(path)
        return True

    def open_log(self) -> bool:
        if not appdata.LOG_FILE.exists():
            return False
        os.startfile(appdata.LOG_FILE)
        return True


def run_window() -> None:
    import webview

    import app

    if FROZEN or sys.stdout is None:
        log_to_file()
        app.trainer.log_file = appdata.LOG_FILE
    hide_console_windows()
    if not webview2_installed():
        ask_for_webview2()
        return

    appdata.ensure_dirs()
    url = start_server()
    api = DesktopApi()
    width, height = 1280, 860
    api._window = webview.create_window(appdata.APP_NAME, url, js_api=api, width=width, height=height,
                                        min_size=(900, 640), background_color=window_background(),
                                        maximized=not fits_on_screen(width, height + 40))  # e.g. 1080p at 125%
    webview.start(gui="edgechromium", storage_path=str(appdata.DATA_DIR / "webview"))


def build_cache(folder: Path) -> None:
    from forecast_engine import ForecastEngine, workbook_key

    engine = ForecastEngine.train(appdata.DEFAULT_WORKBOOK)
    engine.save(Path(folder) / workbook_key(appdata.DEFAULT_WORKBOOK))
    print(f"Saved the seed cache for {appdata.DEFAULT_WORKBOOK.name} in {folder}")


def self_test(workbook: Path, seed_cache: Path) -> int:
    """Train every model on a workbook, compare with the seed cache, and write every export.

    The Studio first takes the workbook the way a user would: reads it for review, and trains on the data set
    it confirms, which must give the same forecasts as the workbook itself. It has two models more than the
    dashboard app, so it compares the 10 they share. It trains one model at a time, so models trained alone and
    merged must match the same models trained with all the others.
    """
    import numpy as np

    import datasets
    import exports
    from forecast_engine import MODELS, STUDIO_MODELS, ForecastEngine, workbook_key

    if FROZEN or sys.stdout is None:
        log_to_file()
    hide_console_windows()
    print(f"Self-test of {appdata.APP_NAME}: training on {workbook.name}")
    try:
        with tempfile.TemporaryDirectory(prefix="arf-selftest-", ignore_cleanup_errors=True) as tmp:
            out = Path(tmp)
            source = workbook
            if appdata.PROFILE == "entry":
                dataset = datasets.from_workbook(workbook)
                found = datasets.problems(dataset)
                assert not found, f"the workbook reads with problems: {found[:3]}"
                source = datasets.write_workbook(dataset, out / "confirmed.xlsx")
                print(f"Reviewed {len(dataset['organisms'])} organisms and "
                      f"{sum(len(o['rows']) for o in dataset['organisms'])} medicines; confirmed as {source.name}")

            studio = appdata.PROFILE == "entry"
            offered = STUDIO_MODELS if studio else MODELS
            started = time.monotonic()
            engine = ForecastEngine.train(source, models=[m["name"] for m in offered])
            print(f"Trained {len(engine.model_names)} models in {time.monotonic() - started:.0f} s:",
                  ", ".join(engine.model_names))
            assert len(engine.model_names) == len(offered), f"expected {len(offered)} models"
            listed = [(m["name"], m["family"], m["about"]) for m in offered]
            trained = [(m["name"], m["family"], m["about"].split(". This data")[0]) for m in engine.meta["models"]]
            assert listed == trained, f"forecast_engine's model list differs from the trained models: {trained}"

            if studio:
                alone = ["SARIMA", "Exponential Smoothing", "XGBoost"]
                merged = ForecastEngine.train(source, models=alone[:1]).merged(ForecastEngine.train(source, models=alone[1:]))
                assert merged.model_names == alone, f"merged models: {merged.model_names}"
                gap = float(np.abs(merged.paths - engine.only(alone).paths).max())
                print(f"{' and '.join(alone)} trained alone: largest difference {gap:.2f} pts")
                assert gap == 0, "a model trained alone forecasts differently"

            seed = seed_cache / workbook_key(workbook)
            if seed.is_dir():
                cached = ForecastEngine.from_cache(seed)
                gap = float(np.abs(cached.paths - engine.only(cached.model_names).paths).max())
                print(f"Seed cache: largest difference {gap:.2f} pts")
                assert gap == 0, "the forecasts differ from the seed cache"
            else:
                print("Seed cache: none found, not compared")

            year = engine.max_year
            written = [exports.write_csv(engine, year, out / "forecasts.csv"),
                       exports.write_excel(engine, year, out / "forecasts.xlsx")]
            if studio:  # the models picked, a report on the one shown, and the picked models compared
                written += [exports.write_excel(engine.only(["ARIMA"]), year, out / "one-model.xlsx"),
                            exports.write_report_pdf(engine.only(["SARIMA-LSTM"]), year, out / "report.pdf", out),
                            exports.write_report_pdf(engine.only(["LightGBM"]), engine.first_year, out / "one-year.pdf", out),
                            exports.write_compare_pdf(engine, year, out / "all-compared.pdf", out),
                            exports.write_compare_pdf(engine.only(["ARIMA", "GRU"]), engine.first_year,
                                                      out / "two-compared.pdf", out)]
            else:
                written += [exports.write_summary_pdf(engine, out / "summary.pdf", out),
                            *exports.write_model_pdfs(engine, out / "models", out)]
            sizes = {p.name: p.stat().st_size for p in written}
            assert all(sizes.values()), f"empty exports: {sizes}"
            print(f"Exports: {len(sizes)} files written")
    except Exception:
        traceback.print_exc()
        print("Self-test FAILED")
        return 1
    print("Self-test passed")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true", help="check the build and exit")
    parser.add_argument("--build-cache", metavar="DIR", help="train on the bundled workbook and save the models")
    parser.add_argument("--workbook", type=Path, help="self-test: the workbook to train on (default: the bundled one)")
    parser.add_argument("--seed-cache", type=Path, help="self-test: the cache to compare with (default: the bundled one)")
    args = parser.parse_args()

    if args.self_test:
        code = self_test(args.workbook or appdata.DEFAULT_WORKBOOK, args.seed_cache or appdata.SEED_CACHE)
    elif args.build_cache:
        build_cache(Path(args.build_cache))
        code = 0
    else:
        run_window()
        code = 0
    if sys.stdout:
        sys.stdout.flush()
    os._exit(code)  # don't wait on the server and trainer threads, or a PDF still printing


if __name__ == "__main__":
    main()

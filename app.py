"""Antibiotic resistance forecast web app: one server for both the dashboard page and its API.

The desktop apps (desktop.py, entry_desktop.py) run this server inside their own window; run on its own, it
opens the page in your browser. On startup it opens the trained models for the active workbook from the cache,
or trains them (about a minute) if this workbook has not been trained before. Every year is then instant:

  /                    the page: the dashboard, or in the entry profile the Studio (appdata.py)
  /api/train           POST: retrain the 10 models, re-reading the workbook, in the background
  /api/status          training progress, whether the models are ready, and which workbook they are for
  /api/meta            recorded values, the models and their backtest error, the year range
  /api/forecast?year=  every model's forecasts up to that year; filter with &organism= and &antibiotic=
  /docs                interactive API docs

The Studio adds (datasets.py describes a data set). It trains only the models the user picks, so its
/api/meta and /api/forecast cover the models generated so far for the confirmed data set:

  /api/session         GET the step and data set the user left off with; PUT to save them, which returns
                       the data set's problems
  /api/confirm         POST a data set without problems: make it the one to forecast, with any models
                       already generated for it from the cache; nothing is trained yet
  /api/models          the 12 models the user can pick from: the dashboard's 10, SARIMA and SARIMA-LSTM
  /api/generate        POST {"models": [names]}: forecast the confirmed data set with those models, training
                       in the background, one at a time, those not generated before; poll /api/status
  /api/upload?name=    POST an .xlsx file's bytes: its values, for review (the page uses it in a browser)
  /api/workbook        POST a data set: it as an .xlsx file in the source layout (likewise)

Run:  python app.py                    start the server and open the dashboard
      python app.py --no-browser       start the server only
      python app.py --port 8001        serve on another port
      python app.py --profile entry    the Studio, with its own data folder
"""

import argparse
import contextlib
import queue
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

import appdata
import datasets
from forecast_engine import STUDIO_MODEL_NAMES, STUDIO_MODELS, ForecastEngine
from parse_antibiotic_trend import WorkbookError, read_workbook

HOST, PORT = "127.0.0.1", 8000


class Trainer:
    """Keeps the trained models for the active workbook, and trains new ones in the background.

    While training runs, the previous engine keeps answering; it is swapped out only once the new one has
    finished, and kept if training fails.

    In the Studio the workbook can be active with no engine yet: the user confirms the data, then picks
    models, and generate() trains them one at a time, adding each to the ones trained before on the same data
    as soon as it is done, so a model that fails keeps those trained before it.

    Every training runs on the same thread. PyTorch sets up its CPU threading per thread, and a retrain
    on a fresh thread gave LSTM forecasts up to 1 point off the first training; on one thread they repeat
    exactly, matching build_forecast_dashboard.py.
    """

    def __init__(self):
        self.engine = None
        self.workbook = None  # appdata.Workbook the engine was trained on
        self.log_file = None  # set by desktop.py when output goes to a log file
        self._lock = threading.Lock()
        self._state = {"state": "idle"}
        self._job = None  # the workbook being trained, or the last one that failed
        self._job_models = None  # the models it is training (None: all of them)
        self._started = 0.0
        self._requests = queue.Queue()
        threading.Thread(target=self._work, name="trainer", daemon=True).start()

    def boot(self) -> None:
        """Open the active workbook's models from the cache, or start training them."""
        appdata.ensure_dirs()
        wb = appdata.active_workbook()
        if wb is None:
            return  # the Studio starts empty and waits for data
        try:
            engine = appdata.cached_engine(wb.key)
        except Exception:
            traceback.print_exc()  # a damaged cache: train again
            engine = None
        if engine:
            self._use(engine, wb)
            print(f"Opened the trained models for {wb.name} from the cache")
        elif appdata.PROFILE == "entry":
            self._use(None, wb)  # the Studio trains when the user picks a model
        else:
            self._start(wb)

    def open_workbook(self, path: Path, name: str | None = None, retrain: bool = False, key: str | None = None,
                      activate: bool = False, train: bool = True) -> dict:
        """Check a workbook and make it the active one: instantly if it was trained before, else by training.

        key names its cache in place of the file's hash. With activate, it is the active workbook from now on,
        even while it trains, so the app comes back to it if closed before training ends. Without train, it
        is active at once with whatever models its cache has, or none, for the Studio to generate().
        Nothing changes if the workbook can't be read; the error says what is wrong with it.
        """
        path = Path(path)
        if self.training:
            return {"ok": False, "error": "The models are still training. Try again when they have finished."}
        try:
            read_workbook(path)
        except WorkbookError as e:
            return {"ok": False, "error": str(e)}
        wb = appdata.remember(path, name, key)
        if activate:
            appdata.set_active(wb)
        engine = None if retrain else appdata.cached_engine(wb.key)
        if engine or not train:
            self._use(engine, wb)
            return {"ok": True, "training": False}
        if not self._start(wb):
            return {"ok": False, "error": "The models are still training. Try again when they have finished."}
        return {"ok": True, "training": True}

    def generate(self, models: list[str]) -> dict:
        """The Studio: forecast the active workbook with the models picked, training those not trained before."""
        if not models:
            return {"ok": False, "error": "Pick at least one model."}
        if unknown := [m for m in models if m not in STUDIO_MODEL_NAMES]:
            return {"ok": False, "error": f"There is no model called {unknown[0]}."}
        if self.workbook is None:
            return {"ok": False, "error": "Confirm your data first."}
        trained = self.engine.model_names if self.engine else []
        todo = [m for m in STUDIO_MODEL_NAMES if m in models and m not in trained]
        if not todo:
            return {"ok": True, "training": False}
        if not self._start(self.workbook, todo):
            return {"ok": False, "error": "A model is still training. Try again when it has finished."}
        return {"ok": True, "training": True}

    def retrain(self) -> dict:
        """Train again on the workbook that failed last, else the active one, re-reading it to pick up edits."""
        wb = self._job or self.workbook or appdata.active_workbook()
        if wb is None:
            return {"ok": False, "error": "There is no data to train on yet."}
        return self.open_workbook(wb.current_source(), name=wb.name, retrain=True)

    @property
    def training(self) -> bool:
        return self._state["state"] == "training"

    def _start(self, wb: appdata.Workbook, models: list[str] | None = None) -> bool:
        with self._lock:
            if self.training:
                return False
            self._job, self._job_models = wb, models
            self._started = time.monotonic()
            self._state = {"state": "training", "model": None, "step": 0, "steps": None, "progress": 0.0}
        self._requests.put((wb, models))
        return True

    def _work(self):
        while True:
            self._train(*self._requests.get())

    def _report(self, model, index, total, fraction):
        self._state = {"state": "training", "model": model, "step": index + 1, "steps": total,
                       "progress": round((index + fraction) / total, 3)}

    def _train(self, wb: appdata.Workbook, models: list[str] | None):
        print(f"Training {', '.join(models) if models else 'every model'} on {wb.name}")
        model = None  # the Studio's model in training
        try:
            if not models:
                engine = ForecastEngine.train(wb.copy, name=wb.name, report=self._report)
                self._store(engine, wb)
            else:  # one at a time, each added to the models trained before on this data once it is done
                same = self.workbook is not None and self.workbook.key == wb.key
                engine = self.engine if same else appdata.cached_engine(wb.key)
                for k, model in enumerate(models):
                    report = lambda name, _index, _total, fraction, k=k: self._report(name, k, len(models), fraction)
                    trained = ForecastEngine.train(wb.copy, name=wb.name, report=report, models=[model])
                    engine = engine.merged(trained) if engine else trained
                    self._store(engine, wb)
                    if same:
                        self.engine = engine  # Ready while the next one trains, and kept if that one fails
        except Exception as e:
            traceback.print_exc()
            message = str(e) if isinstance(e, WorkbookError) else f"{type(e).__name__}: {e}"
            self._state = {"state": "failed", "error": message, "model": model}
            return
        self._use(engine, wb)
        print(f"Models trained in {time.monotonic() - self._started:.0f} s")

    def _store(self, engine: ForecastEngine, wb: appdata.Workbook):
        try:
            appdata.store(engine, wb.key, keep={self.workbook.key} if self.workbook else set())
        except OSError:
            traceback.print_exc()  # the models still work this session; they'll retrain next launch

    def _use(self, engine: ForecastEngine | None, wb: appdata.Workbook):
        if engine:
            engine.meta["workbook"] = wb.name  # a cache shared by two names was saved under the first
        self.engine, self.workbook, self._job, self._job_models = engine, wb, None, None
        appdata.set_active(wb)
        self._state = {"state": "ready"}

    def status(self) -> dict:
        status = {**self._state, "trained": self.engine is not None, "profile": appdata.PROFILE,
                  "logFile": str(self.log_file) if self.log_file else None,
                  "dataset": self.workbook.name if self.workbook else None,
                  "models": self.engine.model_names if self.engine else []}
        if self.engine:
            status.update({k: self.engine.meta[k] for k in ("workbook", "trainedAt", "seconds")})
        if self._job:
            status["trainingWorkbook"] = self._job.name
            status["trainingModels"] = self._job_models
        if self.training:
            status["elapsed"] = round(time.monotonic() - self._started, 1)
        return status


trainer = Trainer()


@contextlib.asynccontextmanager
async def lifespan(_app):
    trainer.boot()
    yield


app = FastAPI(title="Antibiotic resistance forecast", lifespan=lifespan)


def trained_engine() -> ForecastEngine:
    if trainer.engine is None:
        raise HTTPException(409, "The models are not trained yet. Start training with POST /api/train.")
    return trainer.engine


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(appdata.PAGE, headers={"Cache-Control": "no-store"})


@app.post("/api/train", status_code=202)
def train():
    """Retrain in the background, re-reading the workbook; poll /api/status. Does nothing if already training."""
    trainer.retrain()
    return trainer.status()


@app.get("/api/status")
def status():
    return trainer.status()


@app.get("/api/meta")
def meta():
    return trained_engine().meta


@app.get("/api/forecast")
def forecast(year: int, organism: str | None = None, antibiotic: str | None = None):
    """Every model's forecasts up to `year`, from the first year after the recorded ones."""
    engine = trained_engine()
    if not engine.first_year <= year <= engine.max_year:
        raise HTTPException(422, f"Pick a year from {engine.first_year} to {engine.max_year}")
    result = engine.forecast(year, organism, antibiotic)
    if not result["series"]:
        raise HTTPException(404, "No series matches that organism and antibiotic")
    return result


# --- The Studio (entry profile): data sets the user uploads or types in, reviews and confirms ---

SESSION_KEYS = ("step", "draft", "theme", "uploaded", "edited", "view")


def studio_only():
    if appdata.PROFILE != "entry":
        raise HTTPException(404, "Only the Studio app has this")


@app.get("/api/session", dependencies=[Depends(studio_only)])
def get_session():
    """Where the user left off: the step, the data set being edited and its problems."""
    session = appdata.read_session()
    if not session.get("draft") and (wb := appdata.active_workbook()):
        with contextlib.suppress(WorkbookError):  # the confirmed data, if the session was lost
            session["draft"] = {**datasets.from_workbook(wb.copy), "name": wb.name, "file": None}
    draft = session.get("draft")
    return {**session, "problems": datasets.problems(draft) if draft else []}


@app.put("/api/session", dependencies=[Depends(studio_only)])
def put_session(session: dict = Body(...)):
    """Save the step and the data set being edited; returns the data set's problems."""
    saved = {k: session[k] for k in SESSION_KEYS if k in session}
    appdata.write_session(saved)
    draft = saved.get("draft")
    return {"problems": datasets.problems(draft) if draft else []}


@app.post("/api/confirm", dependencies=[Depends(studio_only)])
def confirm(dataset: dict = Body(...)):
    """Make a data set the one to forecast, with the models already generated for it; /api/generate trains."""
    found = datasets.problems(dataset)
    if found:
        return {"ok": False, "problems": found,
                "error": f"Fix the {len(found)} problem{'s' if len(found) > 1 else ''} marked in the review first."}
    name = (dataset.get("name") or "").strip() or "Entered data"
    appdata.ensure_dirs()
    with tempfile.TemporaryDirectory(dir=appdata.TMP_DIR, ignore_cleanup_errors=True) as tmp:
        path = datasets.write_workbook(dataset, Path(tmp) / f"{uuid.uuid4().hex}.xlsx")
        result = trainer.open_workbook(path, name=name, key=datasets.dataset_key(dataset), activate=True,
                                       train=False)
    if result["ok"]:
        appdata.write_session({**appdata.read_session(), "step": 3, "draft": dataset, "uploaded": False,
                               "edited": False})
    return result


@app.get("/api/models", dependencies=[Depends(studio_only)])
def models():
    """The models the user can pick from: name, family, how it works, and seconds to train on 95 series."""
    return STUDIO_MODELS


@app.post("/api/generate", dependencies=[Depends(studio_only)])
def generate(body: dict = Body(...)):
    """Forecast the confirmed data set with the models picked, training those not trained before; poll /api/status."""
    models = body.get("models") or body.get("model") or []
    return trainer.generate([models] if isinstance(models, str) else [str(m) for m in models])


def review(path: Path) -> dict:
    try:
        return {"ok": True, "dataset": datasets.from_workbook(path)}
    except WorkbookError as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/upload", dependencies=[Depends(studio_only)])
async def upload(request: Request, name: str):
    """Read an uploaded workbook's values for review; the body is the .xlsx file."""
    body = await request.body()
    appdata.ensure_dirs()
    with tempfile.TemporaryDirectory(dir=appdata.TMP_DIR, ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / (Path(name).name or "workbook.xlsx")
        path.write_bytes(body)
        return await run_in_threadpool(review, path)


@app.post("/api/workbook", dependencies=[Depends(studio_only)])
def workbook_file(dataset: dict = Body(...)):
    """A data set without problems as an .xlsx file, in the same layout as the workbooks the app reads."""
    if datasets.problems(dataset):
        raise HTTPException(422, "Fix the problems marked in the review first.")
    appdata.ensure_dirs()
    with tempfile.TemporaryDirectory(dir=appdata.TMP_DIR, ignore_cleanup_errors=True) as tmp:
        data = datasets.write_workbook(dataset, Path(tmp) / "data.xlsx").read_bytes()
    return Response(data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


class BrowserServer(uvicorn.Server):
    """uvicorn server that opens the dashboard once it is listening."""

    async def startup(self, sockets=None):
        await super().startup(sockets)
        if self.started:
            url = f"http://{self.config.host}:{self.config.port}"
            print(f"Opening {url} in your browser")
            # In a thread: some browser launchers wait for the browser to exit, which would stall the server.
            threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-browser", action="store_true", help="start the server without opening the dashboard")
    parser.add_argument("--port", type=int, default=PORT, help=f"port to serve on (default {PORT})")
    parser.add_argument("--profile", choices=appdata.PROFILES, default="forecast",
                        help="forecast: the dashboard on the bundled workbook; entry: the Studio, which starts empty")
    args = parser.parse_args()
    appdata.use_profile(args.profile)
    config = uvicorn.Config(app, host=HOST, port=args.port)
    (uvicorn.Server if args.no_browser else BrowserServer)(config).run()

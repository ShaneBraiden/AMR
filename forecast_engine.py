"""Forecasting engine behind the app (app.py and desktop.py).

Trains the same 10 models as build_forecast_dashboard.py on a workbook and rolls each one forward year by
year to YEARS_AHEAD years past the last recorded year:

  Statistical - forecast each series that many steps ahead.
  Machine / deep learning - predict next year's change from the two years before it and add it on,
      feeding each forecast year back in as the lag for the next.

A model's forecast for a year does not depend on how far past it the path is rolled, so the server
answers any year in range by slicing these paths, and for the 2022-2025 workbook 2026 and 2027 match
forecast_dashboard.html.

Training takes about a minute, so a trained engine is saved (save) and reopened (from_cache) instead of
retrained; workbook_key names the cache after the workbook's contents.

The Studio trains only the model the user picks: train(models=[name]) trains that one, and merged() adds it
to the models trained before on the same data. Each model's forecasts come out the same whether it is
trained alone or with the others. It offers two models the dashboard app does not, SARIMA and a SARIMA-LSTM
hybrid (studio_models.py).

build_forecast_dashboard pulls in torch, Prophet and the boosting libraries, so it is imported when
training starts rather than here: importing this module stays quick and the server starts at once.
"""

import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from parse_antibiotic_trend import read_workbook

YEARS_AHEAD = 10
CACHE_VERSION = 1  # bump when the models or their settings change, so older caches are retrained

# The dashboard app's 10 models in training order. build_forecast_dashboard defines them, but importing it
# loads torch, so they are listed here too; the self-test checks the two agree.
# seconds: how long each took to train on the 95 series of the 2022-2025 workbook.
MODELS = [
    {"name": "ARIMA", "family": "Statistical", "about": "ARIMA(0,1,0) with drift, fitted to each series", "seconds": 9},
    {"name": "Exponential Smoothing", "family": "Statistical", "about": "Holt's damped trend, fitted to each series",
     "seconds": 1},
    {"name": "Theta", "family": "Statistical", "about": "Theta method, fitted to each series", "seconds": 7},
    {"name": "Prophet", "family": "Statistical", "about": "Meta's trend model with changepoints, fitted to each series",
     "seconds": 55},
    {"name": "Random Forest", "family": "Machine learning", "about": "500 decision trees, trained across all series",
     "seconds": 4},
    {"name": "XGBoost", "family": "Machine learning", "about": "Gradient-boosted trees, trained across all series",
     "seconds": 2},
    {"name": "LightGBM", "family": "Machine learning", "about": "Leaf-wise boosted trees, trained across all series",
     "seconds": 1},
    {"name": "CatBoost", "family": "Machine learning", "about": "Ordered boosted trees, trained across all series",
     "seconds": 1},
    {"name": "LSTM", "family": "Deep learning", "about": "Long short-term memory network, trained across all series",
     "seconds": 38},
    {"name": "GRU", "family": "Deep learning", "about": "Gated recurrent unit network, trained across all series",
     "seconds": 25},
]
MODEL_NAMES = [m["name"] for m in MODELS]

# The Studio's models: the 10, with SARIMA beside ARIMA and the SARIMA-LSTM hybrid last (studio_models.py,
# whose descriptions the self-test checks these against). Their seconds are scaled from ARIMA's and LSTM's, which
# they took 1.25 and 1.65 times as long as on a faster run.
SARIMA = {"name": "SARIMA", "family": "Statistical", "about": "Seasonal ARIMA with its order chosen for each series by AIC",
          "seconds": 11}
HYBRID = {"name": "SARIMA-LSTM", "family": "Hybrid",
          "about": "SARIMA for each series, plus an LSTM trained across all series on what SARIMA misses", "seconds": 60}
STUDIO_MODELS = [MODELS[0], SARIMA, *MODELS[1:], HYBRID]
STUDIO_MODEL_NAMES = [m["name"] for m in STUDIO_MODELS]


def workbook_key(path: Path) -> str:
    """Cache name for a workbook: changes whenever its contents or the models change."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"v{CACHE_VERSION}-{digest[:24]}"


def lag_rows(history, orgs, levels, upto):
    """Training rows for the learned models: lags at year t-1 and t-2, target = change into year t."""
    from build_forecast_dashboard import lag_features

    X = pd.concat([lag_features(history[:, t - 1], history[:, t - 2], orgs, levels) for t in range(2, upto)],
                  ignore_index=True)
    y = np.concatenate([history[:, t] - history[:, t - 1] for t in range(2, upto)])
    return X, y


def fit_learned(name, model, X, y):
    """Fit a learned model, or when its library refuses the data, an average-change model in its place.

    Typed-in data can be too small or too flat for some libraries: LightGBM needs more than one training row,
    and CatBoost refuses a target that never changes. Predicting the average change is then the most there is
    to learn. Returns the fitted model and whether it is the stand-in.
    """
    try:
        return model.fit(X, y), False
    except Exception as e:
        print(f"{name} could not learn from this data ({type(e).__name__}: {e}); it predicts the average change instead")
        return DummyRegressor().fit(X, y), True


def roll_forward(model, prev, last, orgs, levels, steps):
    """Add the predicted change to the latest value, one year at a time, keeping each year within 0-100%."""
    from build_forecast_dashboard import lag_features

    path = []
    for _ in range(steps):
        nxt = np.clip(last + model.predict(lag_features(last, prev, orgs, levels)), 0, 100)
        path.append(nxt)
        prev, last = last, nxt
    return np.column_stack(path)


class ForecastEngine:
    """Every model's forecast path for every series, and the metadata the dashboard shows.

    Make one with train() (about a minute) or from_cache() (instant).
    """

    def __init__(self, meta: dict, paths: np.ndarray, recorded: np.ndarray):
        self.meta = meta
        self.paths = paths  # paths[series, model, year]: forecasts for first_year..max_year
        self.recorded = recorded  # recorded[series, year]: the workbook's values; meta's history is rounded
        self.model_names = [m["name"] for m in meta["models"]]
        self.series = meta["series"]
        self.history_years = meta["historyYears"]
        self.first_year = meta["minYear"]
        self.max_year = meta["maxYear"]

    @classmethod
    def train(cls, workbook: Path, name: str | None = None, report=lambda *args: None,
              years_ahead: int = YEARS_AHEAD, models: list[str] | None = None) -> "ForecastEngine":
        """Read the workbook and train the dashboard app's 10 models on it, or only the named models, which
        may include the Studio's two.

        report(model_name, index, total, fraction) is called as training moves through the models, with
        fraction the share of the current model done.
        """
        from build_forecast_dashboard import STATISTICAL, lag_features, learned_models
        from studio_models import HYBRID_ABOUT, SARIMA_ABOUT, sarima, sarima_lstm

        wanted = MODEL_NAMES if models is None else models
        if unknown := set(wanted) - set(STUDIO_MODEL_NAMES):
            raise ValueError(f"No such model: {', '.join(sorted(unknown))}")
        started = time.monotonic()
        data = read_workbook(workbook)
        wide, history_years = data.wide, data.years
        history = wide[[f"pct_{y}" for y in history_years]].to_numpy()
        orgs = wide["organism"].to_numpy()
        levels = sorted(set(orgs))
        last = history[:, -1]
        steps = years_ahead
        per_series = {**STATISTICAL, "SARIMA": (sarima, SARIMA_ABOUT)}
        statistical = {k: per_series[k] for k in STUDIO_MODEL_NAMES if k in per_series and k in wanted}
        learned = {k: v for k, v in learned_models().items() if k in wanted}
        hybrid = HYBRID["name"] in wanted
        total = len(statistical) + len(learned) + hybrid

        models, backtest, paths = [], {}, {}

        for k, (model_name, (fn, about)) in enumerate(statistical.items()):
            print(f"Fitting {model_name} on each series...")
            models.append({"name": model_name, "family": "Statistical", "about": about})
            back, path = [], []
            for i, h in enumerate(history):
                back.append(fn(h[:-1], 1)[0])
                path.append(fn(h, steps))
                report(model_name, k, total, (i + 1) / len(history))
            backtest[model_name], paths[model_name] = np.array(back), np.array(path)

        # Backtest on every year but the last, then retrain on all of them for the forecasts.
        n = len(history_years)
        X_back, y_back = lag_rows(history, orgs, levels, n - 1)
        X_all, y_all = lag_rows(history, orgs, levels, n)
        X_last = lag_features(history[:, -2], history[:, -3], orgs, levels)

        for k, (model_name, (model, family, about)) in enumerate(learned.items(), start=len(statistical)):
            print(f"Training {model_name} across all series...")
            report(model_name, k, total, 0.0)
            models.append({"name": model_name, "family": family, "about": about})
            fitted, stand_in = fit_learned(model_name, model, X_back, y_back)
            backtest[model_name] = history[:, -2] + fitted.predict(X_last)
            report(model_name, k, total, 0.5)
            fitted, stand_in_all = fit_learned(model_name, model, X_all, y_all)
            paths[model_name] = roll_forward(fitted, history[:, -2], last, orgs, levels, steps)
            if stand_in or stand_in_all:
                models[-1]["about"] += (". This data has too little variation for it to learn from, so it predicts the "
                                        "average yearly change instead")

        if hybrid:
            model_name, k = HYBRID["name"], total - 1
            print("Fitting SARIMA on each series and an LSTM on what it misses...")
            models.append({"name": model_name, "family": HYBRID["family"], "about": HYBRID_ABOUT})
            backtest[model_name], paths[model_name] = sarima_lstm(
                history, orgs, levels, steps, lambda fraction: report(model_name, k, total, fraction))

        for m in models:
            pred = np.clip(backtest[m["name"]], 0, 100)
            m["mae"] = round(float(mean_absolute_error(last, pred)), 2)
            m["rmse"] = round(float(np.sqrt(mean_squared_error(last, pred))), 2)

        model_names = [m["name"] for m in models]
        meta = {
            "workbook": name or Path(workbook).name,
            "trainedAt": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.monotonic() - started),
            "historyYears": history_years,
            "minYear": history_years[-1] + 1,
            "maxYear": history_years[-1] + years_ahead,
            "models": models,
            "bestModel": min(models, key=lambda m: m["mae"])["name"],
            "baselineMae": round(float(mean_absolute_error(last, history[:, -2])), 2),
            "excluded": data.notes,
            "series": [
                {"organism": wide.at[i, "organism"], "antibiotic": wide.at[i, "antibiotic"],
                 "history": history[i].round(1).tolist()}
                for i in range(len(wide))
            ],
        }
        stacked = np.stack([np.clip(paths[m], 0, 100).round(1) for m in model_names], axis=1)
        return cls(meta, stacked, history)

    def merged(self, other: "ForecastEngine") -> "ForecastEngine":
        """This engine with other's models added, replacing any of the same name; both trained on the same data."""
        if other.series != self.series or other.history_years != self.history_years:
            raise ValueError("Only engines trained on the same data can be merged")
        info = {m["name"]: m for m in self.meta["models"]} | {m["name"]: m for m in other.meta["models"]}
        paths = ({n: self.paths[:, k] for k, n in enumerate(self.model_names)}
                 | {n: other.paths[:, k] for k, n in enumerate(other.model_names)})
        names = sorted(info, key=STUDIO_MODEL_NAMES.index)
        models = [info[n] for n in names]
        meta = {**other.meta, "models": models, "bestModel": min(models, key=lambda m: m["mae"])["name"]}
        return ForecastEngine(meta, np.stack([paths[n] for n in names], axis=1), other.recorded)

    def only(self, names: list[str]) -> "ForecastEngine":
        """The same forecasts, for just the named models."""
        keep = [self.model_names.index(n) for n in names]
        models = [self.meta["models"][k] for k in keep]
        meta = {**self.meta, "models": models, "bestModel": min(models, key=lambda m: m["mae"])["name"]}
        return ForecastEngine(meta, self.paths[:, keep], self.recorded)

    def save(self, directory: Path) -> None:
        """Write the engine to a folder, replacing it whole so a crash never leaves half a cache."""
        directory = Path(directory)
        tmp = directory.with_name(directory.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        np.save(tmp / "paths.npy", self.paths)
        np.save(tmp / "recorded.npy", self.recorded)
        (tmp / "meta.json").write_text(json.dumps(self.meta), encoding="utf-8")
        shutil.rmtree(directory, ignore_errors=True)
        tmp.rename(directory)

    @classmethod
    def from_cache(cls, directory: Path) -> "ForecastEngine":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(meta, np.load(directory / "paths.npy"), np.load(directory / "recorded.npy"))

    def forecast(self, year: int, organism: str | None = None, antibiotic: str | None = None) -> dict:
        """Every model's forecasts from first_year to `year`, optionally for one organism or antibiotic."""
        if not self.first_year <= year <= self.max_year:
            raise ValueError(f"year must be between {self.first_year} and {self.max_year}")
        n = year - self.history_years[-1]
        return {
            "year": year,
            "years": list(range(self.first_year, year + 1)),
            "models": self.model_names,
            "series": [
                {"organism": s["organism"], "antibiotic": s["antibiotic"], "forecast": self.paths[i, :, :n].tolist()}
                for i, s in enumerate(self.series)
                if organism in (None, s["organism"]) and antibiotic in (None, s["antibiotic"])
            ],
        }

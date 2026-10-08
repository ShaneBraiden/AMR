"""Train 10 forecasting models on the resistance data and build forecast_dashboard.html.

Every organism-antibiotic series has four yearly values (2022-2025). Two kinds of model:

  Statistical - ARIMA, Exponential Smoothing, Theta, Prophet
      fitted to each series on its own and extrapolated.
  Machine / deep learning - Random Forest, XGBoost, LightGBM, CatBoost, LSTM, GRU
      four points are too few to train these per series, so each is trained once across
      all 95 series: it learns the year-on-year change from the two previous years and
      the organism.

Backtest: every model sees only 2022-2024 and predicts 2025; its error is the average
absolute miss in percentage points. The forecasts use all four years, and 2027 is
forecast from the 2026 forecast.

Run:  python build_forecast_dashboard.py
"""

import functools
import json
import logging
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor

import parse_antibiotic_trend

ROOT = Path(__file__).parent
WIDE_CSV = ROOT / "output" / "resistance_wide.csv"
TEMPLATE = ROOT / "dashboard_template.html"
OUT_HTML = ROOT / "forecast_dashboard.html"
HISTORY_YEARS = [2022, 2023, 2024, 2025]
FORECAST_YEARS = [2026, 2027]
SEED = 42

warnings.filterwarnings("ignore")  # statsmodels warns on every fit of a 3-4 point series
# Prophet and its Stan backend log every fit; cmdstanpy resets its level on import, so disable it.
for name in ("cmdstanpy", "prophet", "prophet.plot"):
    logging.getLogger(name).disabled = True


# --- Statistical models: fitted per series --------------------------------------------

def arima(history, steps):
    # (0,1,0) with drift is the largest ARIMA order 3-4 points can support.
    return ARIMA(history, order=(0, 1, 0), trend="t").fit().forecast(steps)


def exponential_smoothing(history, steps):
    # Holt's damped trend. Fixed smoothing weights: too few points to estimate them.
    model = ExponentialSmoothing(
        history, trend="add", damped_trend=True, initialization_method="known",
        initial_level=history[0], initial_trend=history[1] - history[0],
    )
    fit = model.fit(smoothing_level=0.8, smoothing_trend=0.2, damping_trend=0.9, optimized=False)
    return fit.forecast(steps)


def theta(history, steps):
    return np.asarray(ThetaModel(history, period=1, deseasonalize=False).fit().forecast(steps))


@functools.cache
def use_bundled_tbb():
    """Make Prophet's Stan model load the TBB library it was built with (Windows only).

    Windows looks in System32 before PATH, so a tbb.dll that other software put in System32 is loaded
    instead of the one Prophet ships, and the model exits with 0xC0000139 ("Error during optimization").
    A copy beside prophet_model.bin is found before System32.
    """
    if sys.platform != "win32":
        return
    import prophet as prophet_pkg

    stan_dir = Path(prophet_pkg.__file__).parent / "stan_model"
    target = stan_dir / "tbb.dll"
    bundled = next(stan_dir.glob("cmdstan-*/stan/lib/stan_math/lib/tbb/tbb.dll"), None)
    if bundled and not target.exists():
        try:
            shutil.copy2(bundled, target)
        except OSError as e:
            raise RuntimeError(f"Prophet needs {bundled} copied to {target}; copy it by hand ({e})") from e


def prophet(history, steps):
    from prophet import Prophet  # deferred so the logging setup above silences its import noise

    use_bundled_tbb()
    ds =pd.date_range(f"{HISTORY_YEARS[0]}-01-01", periods=len(history) + steps, freq="YS")
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.fit(pd.DataFrame({"ds": ds[: len(history)], "y": history}))
    return model.predict(pd.DataFrame({"ds": ds[len(history):]}))["yhat"].to_numpy()


STATISTICAL = {
    "ARIMA": (arima, "ARIMA(0,1,0) with drift, fitted to each series"),
    "Exponential Smoothing": (exponential_smoothing, "Holt's damped trend, fitted to each series"),
    "Theta": (theta, "Theta method, fitted to each series"),
    "Prophet": (prophet, "Meta's trend model with changepoints, fitted to each series"),
}


# --- Machine / deep learning models: trained across all series ------------------------

def lag_features(lag1, lag2, organisms, organism_levels) -> pd.DataFrame:
    lag1, lag2 = np.asarray(lag1, dtype=float), np.asarray(lag2, dtype=float)
    X = pd.DataFrame({"lag1": lag1, "lag2": lag2, "change": lag1 - lag2})
    for org in organism_levels:
        X[f"org_{org}"] = (np.asarray(organisms) == org).astype(float)
    return X


class _RecurrentNet(torch.nn.Module):
    def __init__(self, cell, n_inputs, hidden):
        super().__init__()
        self.rnn = {"lstm": torch.nn.LSTM, "gru": torch.nn.GRU}[cell](n_inputs, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, seq):
        out, _ = self.rnn(seq)
        return self.head(out[:, -1])


class RecurrentRegressor:
    """LSTM or GRU reading the two previous years, with the organism one-hot at each step.

    Averages five seeds: a network trained on ~200 rows depends heavily on its starting weights.
    """

    def __init__(self, cell, hidden=16, epochs=400, lr=0.01, seeds=range(5)):
        self.cell, self.hidden, self.epochs, self.lr, self.seeds = cell, hidden, epochs, lr, seeds

    @staticmethod
    def _sequences(X):
        orgs = X.filter(like="org_").to_numpy()
        steps = [np.hstack([X[[col]].to_numpy() / 100, orgs]) for col in ("lag2", "lag1")]
        return torch.tensor(np.stack(steps, axis=1), dtype=torch.float32)

    def fit(self, X, y):
        seq = self._sequences(X)
        target = torch.tensor(np.asarray(y) / 100, dtype=torch.float32).unsqueeze(1)
        self.nets = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            net = _RecurrentNet(self.cell, seq.shape[2], self.hidden)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
            for _ in range(self.epochs):
                opt.zero_grad()
                torch.nn.functional.mse_loss(net(seq), target).backward()
                opt.step()
            self.nets.append(net.eval())
        return self

    def predict(self, X):
        seq = self._sequences(X)
        with torch.no_grad():
            return np.mean([net(seq).squeeze(1).numpy() for net in self.nets], axis=0) * 100


def learned_models() -> dict:
    return {
        "Random Forest": (
            RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=SEED),
            "Machine learning", "500 decision trees, trained across all series",
        ),
        "XGBoost": (
            XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8,
                         colsample_bytree=0.8, random_state=SEED),
            "Machine learning", "Gradient-boosted trees, trained across all series",
        ),
        "LightGBM": (
            LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=4, min_child_samples=5,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=SEED, verbose=-1),
            "Machine learning", "Leaf-wise boosted trees, trained across all series",
        ),
        "CatBoost": (
            CatBoostRegressor(iterations=500, depth=3, learning_rate=0.03, random_seed=SEED,
                              verbose=False, allow_writing_files=False),
            "Machine learning", "Ordered boosted trees, trained across all series",
        ),
        "LSTM": (
            RecurrentRegressor("lstm"),
            "Deep learning", "Long short-term memory network, trained across all series",
        ),
        "GRU": (
            RecurrentRegressor("gru"),
            "Deep learning", "Gated recurrent unit network, trained across all series",
        ),
    }


def main() -> None:
    parse_antibiotic_trend.main()
    wide = pd.read_csv(WIDE_CSV)
    history = wide[[f"pct_{y}" for y in HISTORY_YEARS]].to_numpy()
    orgs = wide["organism"].to_numpy()
    levels = sorted(set(orgs))
    y22, y23, y24, y25 = history.T

    models, backtest, forecasts = [], {}, {}

    for name, (fn, about) in STATISTICAL.items():
        print(f"Fitting {name} on each series...")
        models.append({"name": name, "family": "Statistical", "about": about})
        backtest[name] = np.array([fn(h[:-1], 1)[0] for h in history])
        forecasts[name] = np.array([fn(h, len(FORECAST_YEARS)) for h in history])

    # Learned models predict the change from last year, then add it back.
    X_step_2024 = lag_features(y23, y22, orgs, levels)
    X_step_2025 = lag_features(y24, y23, orgs, levels)
    X_all = pd.concat([X_step_2024, X_step_2025], ignore_index=True)
    y_all = np.concatenate([y24 - y23, y25 - y24])

    for name, (model, family, about) in learned_models().items():
        print(f"Training {name} across all series...")
        models.append({"name": name, "family": family, "about": about})
        model.fit(X_step_2024, y24 - y23)
        backtest[name] = y24 + model.predict(X_step_2025)

        model.fit(X_all, y_all)
        f26 = np.clip(y25 + model.predict(lag_features(y25, y24, orgs, levels)), 0, 100)
        f27 = f26 + model.predict(lag_features(f26, y25, orgs, levels))
        forecasts[name] = np.column_stack([f26, f27])

    for m in models:
        pred = np.clip(backtest[m["name"]], 0, 100)
        m["mae"] = round(mean_absolute_error(y25, pred), 2)
        m["rmse"] = round(float(np.sqrt(mean_squared_error(y25, pred))), 2)
        forecasts[m["name"]] = np.clip(forecasts[m["name"]], 0, 100).round(1)

    names = [m["name"] for m in models]
    best = min(models, key=lambda m: m["mae"])["name"]
    baseline_mae = round(mean_absolute_error(y25, y24), 2)

    pd.DataFrame(models).to_csv(ROOT / "output" / "model_backtest.csv", index=False)
    pd.concat(
        [
            wide[["organism", "antibiotic"]].assign(
                model=name, forecast_2026=forecasts[name][:, 0], forecast_2027=forecasts[name][:, 1]
            )
            for name in names
        ],
        ignore_index=True,
    ).to_csv(ROOT / "output" / "model_forecasts.csv", index=False)

    payload = {
        "historyYears": HISTORY_YEARS,
        "forecastYears": FORECAST_YEARS,
        "models": models,
        "bestModel": best,
        "baselineMae": baseline_mae,
        "excluded": {"Streptococcus spp.": "No antibiotics showed an increase from 2022 to 2025."},
        "series": [
            {
                "organism": wide.at[i, "organism"],
                "antibiotic": wide.at[i, "antibiotic"],
                "history": history[i].round(1).tolist(),
                "forecast": [forecasts[name][i].tolist() for name in names],
            }
            for i in range(len(wide))
        ],
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    OUT_HTML.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", data), encoding="utf-8")

    print("\nBacktest - trained on 2022-2024, predicting 2025 (error in percentage points):")
    print(pd.DataFrame(models)[["name", "family", "mae", "rmse"]].sort_values("mae").to_string(index=False))
    print(f"Naive baseline (2025 = 2024) MAE: {baseline_mae}")
    print(f"Most accurate: {best}")
    print(f"Wrote {OUT_HTML.name}")


if __name__ == "__main__":
    main()

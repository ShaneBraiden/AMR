"""The two models only the Studio offers: SARIMA, and a SARIMA-LSTM hybrid.

  SARIMA - fitted to each series, with its order chosen for that series by AIC.
  SARIMA-LSTM - Zhang's hybrid: SARIMA forecasts each series' trend, and an LSTM trained across all
      series learns what SARIMA misses; the forecast is the two added together.

Fitting loads statsmodels and torch, so they are imported then: build_studio_report reads ARMA_MIN_VALUES from here
without loading them.
"""

import warnings

import numpy as np
import pandas as pd

# Fewer values than this and AR or MA terms fit the noise: on the 2022-2025 workbook they forecast 105% in
# 2026 for a series recorded at 77% in 2025.
ARMA_MIN_VALUES = 8

SARIMA_ABOUT = "Seasonal ARIMA with its order chosen for each series by AIC"
HYBRID_ABOUT = "SARIMA for each series, plus an LSTM trained across all series on what SARIMA misses"


def sarima_fit(history):
    """The SARIMA fit with the lowest AIC among the orders the series is long enough for.

    Yearly data has no seasonal cycle, so the seasonal terms are off, as auto-SARIMA tools set them for
    yearly data. Every series may take a random walk with or without drift; AR(1) and MA(1) terms are tried
    once a series has ARMA_MIN_VALUES values.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    orders = [(0, 1, 0)]
    if len(history) >= ARMA_MIN_VALUES:
        orders += [(1, 1, 0), (0, 1, 1), (1, 1, 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a few points rarely let the optimiser converge cleanly
        fits = [SARIMAX(np.asarray(history, dtype=float), order=order, trend=trend).fit(disp=False)
                for order in orders for trend in ("n", "c")]  # with d=1, "c" is the drift
    return min(fits, key=lambda fit: fit.aic)


def sarima(history, steps):
    return np.asarray(sarima_fit(history).forecast(steps))


def sarima_lstm(history, orgs, levels, steps, report=lambda fraction: None):
    """The hybrid's forecast of the last recorded year from the years before it (the backtest), and its
    forecasts for `steps` years after the data: (backtest[series], paths[series, step]).

    report(fraction) is called as the work moves on.
    """
    back = _hybrid(history[:, :-1], orgs, levels, 1, lambda f: report(f / 2))[:, 0]
    paths = _hybrid(history, orgs, levels, steps, lambda f: report(0.5 + f / 2))
    return back, paths


def _hybrid(history, orgs, levels, steps, report):
    """Fit SARIMA to each series and an LSTM to its residuals, then roll forward `steps` years.

    The residuals are the recorded values minus SARIMA's one-step predictions within the data. The LSTM reads
    the two years before each residual, with the organism, as the LSTM model does to learn yearly changes.

    Each forecast year is made the way the residuals were: SARIMA's one-step forecast from the years before it,
    plus the residual the LSTM expects, and then joins the series for both models' next step. Adding the
    corrections to SARIMA's multi-step forecast instead set each one against the last recorded year rather than
    the year before, and two in three series zigzagged.
    """
    from build_forecast_dashboard import RecurrentRegressor, lag_features

    fits = []
    for i, h in enumerate(history):
        fits.append(sarima_fit(h))
        report(0.4 * (i + 1) / len(history))
    predicted = np.array([np.asarray(fit.predict()) for fit in fits])

    years = range(2, history.shape[1])
    X = pd.concat([lag_features(history[:, t - 1], history[:, t - 2], orgs, levels) for t in years],
                  ignore_index=True)
    residuals = np.concatenate([history[:, t] - predicted[:, t] for t in years])
    net = RecurrentRegressor("lstm").fit(X, residuals)
    report(1.0)

    prev, last = history[:, -2], history[:, -1]
    path = []
    for j in range(steps):
        linear = np.array([fit.forecast(1)[0] for fit in fits])
        nxt = np.clip(linear + net.predict(lag_features(last, prev, orgs, levels)), 0, 100)
        path.append(nxt)
        prev, last = last, nxt
        if j < steps - 1:
            fits = [fit.extend(np.array([v])) for fit, v in zip(fits, nxt)]
    return np.column_stack(path)

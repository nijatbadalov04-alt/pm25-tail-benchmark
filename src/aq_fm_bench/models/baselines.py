"""
Naive forecasting floors. Each takes a 1-D history array (ending
at the forecast origin) and returns an `horizon`-step-ahead point forecast.

    persistence       : carry the last observed value forward (flat).
    seasonal_naive(p) : repeat the last p observed values (p=24 -> "same hour yesterday").
"""
from __future__ import annotations

import numpy as np


def persistence(history: np.ndarray, horizon: int) -> np.ndarray:
    history = np.asarray(history, dtype="float64")
    return np.repeat(history[-1], horizon)


def seasonal_naive(history: np.ndarray, horizon: int, period: int = 24) -> np.ndarray:
    history = np.asarray(history, dtype="float64")
    season = history[-period:]
    reps = int(np.ceil(horizon / period))
    return np.tile(season, reps)[:horizon]


def make_seasonal_naive(period: int = 24):
    """Return a forecaster closure with a fixed seasonal period."""
    def _f(history: np.ndarray, horizon: int) -> np.ndarray:
        return seasonal_naive(history, horizon, period=period)
    _f.__name__ = f"seasonal_naive_{period}h"
    return _f

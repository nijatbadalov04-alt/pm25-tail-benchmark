"""Point-forecast metrics. All mask NaNs pairwise so partial-missing actuals are fine."""
from __future__ import annotations

import numpy as np


def _mask(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")
    m = np.isfinite(yt) & np.isfinite(yp)
    return yt[m], yp[m]


def mae(y_true, y_pred) -> float:
    yt, yp = _mask(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if yt.size else float("nan")


def rmse(y_true, y_pred) -> float:
    yt, yp = _mask(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if yt.size else float("nan")


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in %, guarded against zero denominators."""
    yt, yp = _mask(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    denom = np.abs(yt) + np.abs(yp)
    denom = np.where(denom == 0, np.nan, denom)
    return float(100.0 * np.nanmean(2.0 * np.abs(yt - yp) / denom))


def bias(y_true, y_pred) -> float:
    """Mean signed error (pred - true). Negative => under-prediction."""
    yt, yp = _mask(y_true, y_pred)
    return float(np.mean(yp - yt)) if yt.size else float("nan")


def all_point_metrics(y_true, y_pred) -> dict[str, float]:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred),
            "smape": smape(y_true, y_pred), "bias": bias(y_true, y_pred),
            "n": int(_mask(y_true, y_pred)[0].size)}

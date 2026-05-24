"""
Rolling-origin evaluation: every `stride_hours` over the test
window, take a context ending at the origin, forecast `horizon` steps, and compare to
the realised values. Contexts are forward-filled (models need contiguous history);
actuals are kept RAW so we only score against genuinely observed values.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

Forecaster = Callable[[np.ndarray, int], np.ndarray]


def rolling_origin_eval(
    raw: pd.Series,
    forecaster: Forecaster,
    horizon: int,
    *,
    test_start: str,
    test_end: str,
    context_len: int = 512,
    stride_hours: int = 168,
    return_origins: bool = False,
):
    """Return (preds, actuals, info[, origin_records]) over all valid origins."""
    vals = raw.to_numpy(dtype="float64")
    ffill = raw.ffill().to_numpy(dtype="float64")
    idx = raw.index
    ts0 = pd.Timestamp(test_start, tz="UTC")
    ts1 = pd.Timestamp(test_end, tz="UTC")

    in_test = (idx >= ts0) & (idx < ts1)
    origins = np.flatnonzero(in_test)[::stride_hours]

    preds, acts, records = [], [], []
    n_used = n_skip = 0
    for p in origins:
        if p - context_len + 1 < 0 or p + 1 + horizon > len(idx):
            n_skip += 1
            continue
        ctx = ffill[p - context_len + 1 : p + 1]
        if not np.isfinite(ctx).all():
            n_skip += 1
            continue
        yhat = np.asarray(forecaster(ctx, horizon), dtype="float64")
        actual = vals[p + 1 : p + 1 + horizon]
        m = np.isfinite(actual) & np.isfinite(yhat)
        if not m.any():
            n_skip += 1
            continue
        preds.append(yhat[m])
        acts.append(actual[m])
        n_used += 1
        if return_origins:
            records.append({"origin_idx": int(p), "origin_ts": idx[p],
                            "context": ctx, "yhat": yhat, "actual": actual})

    info = {"n_origins": n_used, "n_skipped": n_skip, "horizon": horizon,
            "stride_hours": stride_hours, "context_len": context_len}
    p_arr = np.concatenate(preds) if preds else np.array([])
    a_arr = np.concatenate(acts) if acts else np.array([])
    if return_origins:
        return p_arr, a_arr, info, records
    return p_arr, a_arr, info

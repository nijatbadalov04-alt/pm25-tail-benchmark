"""
Leaderboard engine. Batched rolling-origin evaluation so foundation models predict ALL
origin windows in one GPU call (not one-at-a-time) — fast and keeps the GPU busy.

A "batch forecaster" maps contexts [N, context_len] -> point preds [N, horizon].
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from aq_fm_bench.data.processing import SPLITS, to_hourly_series
from aq_fm_bench.data.stations import Station
from aq_fm_bench.metrics.point import all_point_metrics
from aq_fm_bench.paths import RAW_AURN

BatchForecaster = Callable[[np.ndarray, int], np.ndarray]


# ---- data --------------------------------------------------------------
def load_station_series(station: Station, pollutant: str) -> pd.Series:
    if station.source == "aurn":
        df = pd.read_parquet(RAW_AURN / f"{station.code}_2019_2024.parquet")
        return to_hourly_series(df, pollutant)
    raise NotImplementedError(f"source {station.source} not wired yet")


def collect_windows(raw: pd.Series, horizon: int, *, context_len: int, stride_hours: int,
                    test_start: str, test_end: str):
    """Return contexts [N, context_len], actuals [N, horizon], origin_ts list — or None."""
    vals = raw.to_numpy("float64")
    ff = raw.ffill().to_numpy("float64")
    idx = raw.index
    ts0, ts1 = pd.Timestamp(test_start, tz="UTC"), pd.Timestamp(test_end, tz="UTC")
    origins = np.flatnonzero((idx >= ts0) & (idx < ts1))[::stride_hours]
    ctxs, acts, ots = [], [], []
    for p in origins:
        if p - context_len + 1 < 0 or p + 1 + horizon > len(idx):
            continue
        ctx = ff[p - context_len + 1 : p + 1]
        if not np.isfinite(ctx).all():
            continue
        ctxs.append(ctx)
        acts.append(vals[p + 1 : p + 1 + horizon])
        ots.append(idx[p])
    if not ctxs:
        return None
    return np.asarray(ctxs, "float64"), np.asarray(acts, "float64"), ots


# ---- batched baseline forecasters -------------------------------------
def fc_persistence(ctxs: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(ctxs[:, -1:], horizon, axis=1)


def make_fc_seasonal(period: int) -> BatchForecaster:
    def _f(ctxs: np.ndarray, horizon: int) -> np.ndarray:
        season = ctxs[:, -period:]
        reps = int(np.ceil(horizon / period))
        return np.tile(season, (1, reps))[:, :horizon]
    return _f


class ChronosBoltBatch:
    """Loads Chronos-Bolt-Base once; predicts a whole batch of windows per call."""

    def __init__(self, model_id: str = "amazon/chronos-bolt-base", max_batch: int = 256):
        import torch
        from chronos import BaseChronosPipeline
        self.torch = torch
        self.max_batch = max_batch
        t0 = time.time()
        self.pipe = BaseChronosPipeline.from_pretrained(
            model_id, device_map="cuda", dtype=torch.bfloat16,
        )
        print(f"[chronos] loaded {model_id} on CUDA in {time.time()-t0:.1f}s")

    def __call__(self, ctxs: np.ndarray, horizon: int) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(ctxs), self.max_batch):
            batch = torch.tensor(ctxs[i : i + self.max_batch], dtype=torch.float32)
            q, _ = self.pipe.predict_quantiles(
                batch, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9],
            )
            out.append(q[:, :, 1].float().cpu().numpy())  # median
        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
        return np.concatenate(out, axis=0)


# ---- runner ------------------------------------------------------------
def run_leaderboard(stations, *, pollutants=("NO2", "PM2.5"), horizons=(24, 72, 168),
                    context_len=512, stride_hours=168, models=None) -> pd.DataFrame:
    test_start, test_end = SPLITS["test"]
    if models is None:
        models = {
            "persistence": fc_persistence,
            "seasonal_naive_24h": make_fc_seasonal(24),
            "seasonal_naive_168h": make_fc_seasonal(168),
            "chronos_bolt_base": ChronosBoltBatch(),
        }
    rows = []
    for st in stations:
        for pol in pollutants:
            if pol not in st.pollutants:
                continue
            raw = load_station_series(st, pol)
            for h in horizons:
                win = collect_windows(raw, h, context_len=context_len,
                                      stride_hours=stride_hours,
                                      test_start=test_start, test_end=test_end)
                if win is None:
                    print(f"  [skip] {st.code} {pol} h={h}: no valid windows")
                    continue
                ctxs, acts, ots = win
                for mname, fc in models.items():
                    t0 = time.time()
                    preds = fc(ctxs, h)
                    m = all_point_metrics(acts.ravel(), preds.ravel())
                    rows.append({
                        "city": st.city, "station": st.code, "env_type": st.env_type,
                        "pollutant": pol, "horizon": h, "model": mname,
                        "mae": round(m["mae"], 3), "rmse": round(m["rmse"], 3),
                        "smape": round(m["smape"], 2), "bias": round(m["bias"], 3),
                        "n": m["n"], "n_origins": len(ots), "seconds": round(time.time()-t0, 2),
                    })
                print(f"  {st.code:5} {pol:5} h={h:3}: "
                      + "  ".join(f"{r['model'].split('_')[0][:4]}={r['mae']:.1f}"
                                  for r in rows[-len(models):]))
    return pd.DataFrame(rows)

"""
Content-hashed cache for XGBoost direct-trajectory predictions.

The same (station, pollutant, oracle, horizon, position-set) XGBoost forecast is recomputed in
run_significance_leaderboard, significance_pm25_tail, run_pm25_analysis and run_pm25_mmcp4d — and
re-fit on every invocation. Fitting a 400-tree model on ~1M (origin x lead) rows dominates the
post-bootstrap wall-clock. This caches the [n_test, horizon] prediction array under a key derived
from a BLAKE2b hash of every input that can change the result, so a stale cache is impossible:

    target column values, the weather columns, the timestamp index, train + test positions,
    horizon, the oracle-weather flag, and SPEC_VERSION (bump when build_features/LAGS/WEATHER change).

Cache hits return the array bit-for-bit, so results are unchanged (verified by
scripts/verify_results_unchanged.py). Cache lives on the SSD at data/processed/xgb_cache/.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from aq_fm_bench.models.xgb import LAGS, WEATHER, XGBDirectForecaster
from aq_fm_bench.paths import PROCESSED

SPEC_VERSION = "v1"   # bump if build_features(), LAGS, WEATHER, or the XGB params change
CACHE_DIR = PROCESSED / "xgb_cache"


def _arr_bytes(a) -> bytes:
    return np.ascontiguousarray(a).tobytes()


def _key(df: pd.DataFrame, col: str, train_pos, test_pos, horizon: int, oracle: bool,
         params: dict) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(SPEC_VERSION.encode())
    h.update(f"{col}|{horizon}|{int(oracle)}".encode())
    h.update(repr(sorted(params.items())).encode())
    h.update(repr(LAGS).encode()); h.update(repr(WEATHER).encode())
    h.update(_arr_bytes(df[col].to_numpy("float64")))
    h.update(_arr_bytes(pd.DatetimeIndex(df["timestamp_utc"]).asi8))
    for w in WEATHER:
        if w in df.columns:
            h.update(w.encode()); h.update(_arr_bytes(df[w].to_numpy("float64")))
    h.update(_arr_bytes(np.asarray(train_pos, dtype=np.int64)))
    h.update(_arr_bytes(np.asarray(test_pos, dtype=np.int64)))
    return h.hexdigest()


def xgb_traj_cached(df: pd.DataFrame, col: str, train_pos, test_pos, horizon: int,
                    *, oracle: bool, label: str, device: str = "cuda", verbose: bool = True,
                    **xgb_kwargs) -> np.ndarray:
    """Fit-and-predict an XGBDirectForecaster trajectory, memoised on the SSD by content hash.

    `label` is a human-readable filename prefix (e.g. "SHDG_no2_oracle"); correctness rests on the
    hash, not the label. Returns [len(test_pos), horizon] float array, identical to a fresh fit.
    """
    params = dict(oracle_weather=oracle, device=device, **xgb_kwargs)
    key = _key(df, col, train_pos, test_pos, horizon, oracle, params)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{label}_{key}.npy"
    if path.exists():
        if verbose:
            print(f"    [xgb-cache HIT ] {label} ({path.name[:28]}…)")
        return np.load(path)
    if verbose:
        print(f"    [xgb-cache MISS] {label} — fitting", flush=True)
    model = XGBDirectForecaster(device=device, oracle_weather=oracle, **xgb_kwargs)
    pred = model.fit(df, col, train_pos).predict_traj(df, col, test_pos, horizon)
    np.save(path, pred)
    return pred

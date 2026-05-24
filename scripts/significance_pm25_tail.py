"""
Significance test for finding F1: on the top-decile PM2.5 (episodes) at 24h, is ANY model
significantly better than persistence? Block bootstrap over forecast origins (resamples
station-origin units, respecting temporal structure) -> 95% CI on dMAE = MAE_persist - MAE_model.
If the CI includes 0, that model is NOT significantly better than persistence on the tail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aq_fm_bench.data.processing import SPLITS
from aq_fm_bench.data.stations import TIER1
from aq_fm_bench.experiments.leaderboard import ChronosBoltBatch, fc_persistence
from aq_fm_bench.models.timesfm_model import TimesFMBatch
from aq_fm_bench.models.xgb_cache import xgb_traj_cached
from aq_fm_bench.paths import PROCESSED
from aq_fm_bench.stats.bootstrap import count_matrix, dmae_ci, make_boots

WEATHER_COLS = ["u10", "v10", "t2m_c", "sp_hpa", "blh", "tp_mm"]
C, STRIDE, H, MAXH = 512, 168, 24, 168


def test_positions(idx, ff):
    ts0, ts1 = pd.Timestamp(SPLITS["test"][0], tz="UTC"), pd.Timestamp(SPLITS["test"][1], tz="UTC")
    cand = np.flatnonzero((idx >= ts0) & (idx < ts1))[::STRIDE]
    return np.array([p for p in cand if p - C + 1 >= 0 and p + MAXH < len(idx)
                     and np.isfinite(ff[p - C + 1: p + 1]).all()])


def train_positions(idx, stride=6):
    ts0, ts1 = pd.Timestamp(SPLITS["train"][0], tz="UTC"), pd.Timestamp(SPLITS["train"][1], tz="UTC")
    cand = np.flatnonzero((idx >= ts0) & (idx < ts1))[::stride]
    cap = int(np.flatnonzero(idx >= ts1)[0]) - MAXH
    return np.array([p for p in cand if 168 <= p < cap])


def main() -> int:
    chronos = ChronosBoltBatch()
    timesfm = TimesFMBatch()
    MODELS = ["chronos_bolt_base", "timesfm_2_wx", "xgboost", "xgboost_no_oracle"]

    units = []  # per (station,origin): {model: abs_errors over that origin's tail steps}
    for st in TIER1:
        if "PM2.5" not in st.pollutants:
            continue
        df = pd.read_parquet(PROCESSED / f"{st.code}_hourly.parquet")
        idx = pd.DatetimeIndex(df["timestamp_utc"])
        vals = df["pm25"].to_numpy("float64")
        ff = pd.Series(vals).ffill().to_numpy()
        wf = {w: pd.Series(df[w].to_numpy("float64")).ffill().bfill().to_numpy() for w in WEATHER_COLS}
        tpos = test_positions(idx, ff)
        if len(tpos) == 0:
            continue
        ctxs = np.stack([ff[p - C + 1: p + 1] for p in tpos])
        actual = np.stack([vals[p + 1: p + 1 + H] for p in tpos])
        q90 = np.nanpercentile(np.stack([vals[p + 1: p + 1 + MAXH] for p in tpos]), 90)
        covH = {w: np.stack([wf[w][p - C + 1: p + 1 + H] for p in tpos]) for w in WEATHER_COLS}
        tr = train_positions(idx)
        preds = {
            "persistence": fc_persistence(ctxs, H),
            "chronos_bolt_base": chronos(ctxs, H),
            "timesfm_2_wx": timesfm.predict(ctxs, H, covariates=covH),
            "xgboost": xgb_traj_cached(df, "pm25", tr, tpos, H, oracle=True, label=f"{st.code}_pm25_oracle"),
            "xgboost_no_oracle": xgb_traj_cached(df, "pm25", tr, tpos, H, oracle=False, label=f"{st.code}_pm25_noor"),
        }
        for i in range(len(tpos)):
            tail = np.isfinite(actual[i]) & (actual[i] >= q90)
            if tail.sum() == 0:
                continue
            u = {m: np.abs(actual[i][tail] - preds[m][i][tail]) for m in ["persistence", *MODELS]}
            units.append(u)
        print(f"  {st.code}: {len(tpos)} origins, q90={q90:.1f}", flush=True)

    n = len(units)
    print(f"\ntail origin-units pooled: {n}")
    base = np.concatenate([u["persistence"] for u in units]).mean()
    print(f"\nPM2.5 TOP-DECILE @24h — persistence MAE = {base:.3f}")
    print(f"{'model':20} {'MAE':>7} {'dMAE(pers-mdl)':>15} {'95% CI':>20} {'better?':>9}")
    # Vectorised block bootstrap (same default_rng(42) draws as the old loop).
    C = count_matrix(make_boots(n, 2000, np.random.default_rng(42)), n)
    pers = [u["persistence"] for u in units]
    for m in MODELS:
        mae_m = np.concatenate([u[m] for u in units]).mean()
        d_full, lo, hi, significant = dmae_ci(pers, [u[m] for u in units], C)  # MAE(pers)-MAE(m); >0 => m better
        sig = "YES" if significant else "no"
        print(f"{m:20} {mae_m:7.3f} {d_full:+15.3f}   [{lo:+.3f}, {hi:+.3f}]   {sig:>9}")
    print("\n(dMAE>0 => model better than persistence; 'better?'=YES only if 95% CI excludes 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

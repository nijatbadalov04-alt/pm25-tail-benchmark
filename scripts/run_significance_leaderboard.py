"""
Significance-complete leaderboard @24h: per-model MAE + 95% block-bootstrap CI (overall +
top-decile) for both pollutants, plus the pairwise tests backing F2-F6. GPU models computed live;
Moirai loaded from results/preds/moirai_test_preds.pkl.
Outputs: results/leaderboard/significance_h24.csv ; results/leaderboard/pairwise_tests_h24.csv
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aq_fm_bench.data.processing import SPLITS
from aq_fm_bench.data.stations import TIER1
from aq_fm_bench.experiments.leaderboard import ChronosBoltBatch, fc_persistence, make_fc_seasonal
from aq_fm_bench.models.linear import LinearForecaster
from aq_fm_bench.models.timesfm_model import TimesFMBatch
from aq_fm_bench.models.xgb_cache import xgb_traj_cached
from aq_fm_bench.paths import PROCESSED, RESULTS
from aq_fm_bench.stats.bootstrap import count_matrix, dmae_ci, mae_ci_multi, make_boots

COL = {"NO2": "no2", "PM2.5": "pm25"}
WEATHER = ["u10", "v10", "t2m_c", "sp_hpa", "blh", "tp_mm"]
C, STRIDE, H, MAXH = 512, 168, 24, 168
MODELS = ["persistence", "seasonal_naive_24h", "seasonal_naive_168h", "linear_no_oracle",
          "linear_oracle", "chronos_bolt_base", "timesfm_2_wx", "moirai_1_1_zeroshot",
          "moirai_1_1_wx", "xgboost_no_oracle", "xgboost"]
PAIRS = {  # (reference a, candidate b): finding — dMAE>0 means b better
    "NO2": [("chronos_bolt_base", "xgboost_no_oracle", "F2 Chronos vs XGB-no-weather"),
            ("chronos_bolt_base", "timesfm_2_wx", "F3a TimesFM+wx vs Chronos"),
            ("timesfm_2_wx", "xgboost", "F3b XGBoost vs TimesFM+wx"),
            ("linear_oracle", "xgboost", "F4 nonlin|weather (lin_orc vs xgb)"),
            ("linear_no_oracle", "xgboost_no_oracle", "F4 nonlin|no-weather"),
            ("moirai_1_1_zeroshot", "moirai_1_1_wx", "F6 Moirai wx vs zeroshot")],
    "PM2.5": [("xgboost", "timesfm_2_wx", "F5 TimesFM+wx vs XGBoost")],
}


def positions(idx, ff):
    a, b = pd.Timestamp(SPLITS["test"][0], tz="UTC"), pd.Timestamp(SPLITS["test"][1], tz="UTC")
    c = np.flatnonzero((idx >= a) & (idx < b))[::STRIDE]
    return np.array([p for p in c if p - C + 1 >= 0 and p + MAXH < len(idx)
                     and np.isfinite(ff[p - C + 1: p + 1]).all()])


def trainpos(idx, stride=6):
    a, b = pd.Timestamp(SPLITS["train"][0], tz="UTC"), pd.Timestamp(SPLITS["train"][1], tz="UTC")
    c = np.flatnonzero((idx >= a) & (idx < b))[::stride]
    cap = int(np.flatnonzero(idx >= b)[0]) - MAXH
    return np.array([p for p in c if 168 <= p < cap])


def main() -> int:
    moirai = pickle.load(open(RESULTS / "preds" / "moirai_test_preds.pkl", "rb"))
    chronos, timesfm = ChronosBoltBatch(), TimesFMBatch()
    s24, s168 = make_fc_seasonal(24), make_fc_seasonal(168)
    rng = np.random.default_rng(42)
    lb_rows, pair_rows = [], []
    loss_dump = {}  # per-origin mean abs error per model -> for the DM+BH test (run_dm_bh_h24.py)

    for pol in ("NO2", "PM2.5"):
        col = COL[pol]
        units = []
        for st in TIER1:
            if pol not in st.pollutants:
                continue
            df = pd.read_parquet(PROCESSED / f"{st.code}_hourly.parquet")
            idx = pd.DatetimeIndex(df["timestamp_utc"])
            vals = df[col].to_numpy("float64"); ff = pd.Series(vals).ffill().to_numpy()
            wf = {w: pd.Series(df[w].to_numpy("float64")).ffill().bfill().to_numpy() for w in WEATHER}
            tp = positions(idx, ff)
            ctx = np.stack([ff[p - C + 1: p + 1] for p in tp])
            cov = {w: np.stack([wf[w][p - C + 1: p + 1 + H] for p in tp]) for w in WEATHER}
            actual = np.stack([vals[p + 1: p + 1 + H] for p in tp])
            q90 = np.nanpercentile(np.stack([vals[p + 1: p + 1 + MAXH] for p in tp]), 90)
            tr = trainpos(idx)
            pred = {
                "persistence": fc_persistence(ctx, H), "seasonal_naive_24h": s24(ctx, H),
                "seasonal_naive_168h": s168(ctx, H), "chronos_bolt_base": chronos(ctx, H),
                "timesfm_2_wx": timesfm.predict(ctx, H, covariates=cov),
                "linear_oracle": LinearForecaster(True).fit(df, col, tr).predict_traj(df, col, tp, H),
                "linear_no_oracle": LinearForecaster(False).fit(df, col, tr).predict_traj(df, col, tp, H),
                "xgboost": xgb_traj_cached(df, col, tr, tp, H, oracle=True, label=f"{st.code}_{col}_oracle"),
                "xgboost_no_oracle": xgb_traj_cached(df, col, tr, tp, H, oracle=False, label=f"{st.code}_{col}_noor"),
                "moirai_1_1_wx": moirai[("wx", pol, H, st.code)]["test_pred"],
                "moirai_1_1_zeroshot": moirai[("zeroshot", pol, H, st.code)]["test_pred"],
            }
            for i in range(len(tp)):
                u = {"tail": actual[i] >= q90, "station": st.code}
                for m in MODELS:
                    u["ae_" + m] = np.abs(actual[i] - pred[m][i])
                units.append(u)

        # Vectorised block bootstrap (count-matrix matmul) — identical draws as the old
        # [rng.integers(0,n,n) for _ in 2000] loop (shared rng), so results are unchanged; ~100x faster.
        n = len(units)
        Cmat = count_matrix(make_boots(n, 2000, rng), n)                     # Cmat: bootstrap count matrix (C is context len)
        ov = {m: [u["ae_" + m] for u in units] for m in MODELS}              # overall errors per unit
        tl = {m: [u["ae_" + m][u["tail"]] for u in units] for m in MODELS}   # tail-only (var length)
        ov_ci, tl_ci = mae_ci_multi(ov, Cmat), mae_ci_multi(tl, Cmat)
        for m in MODELS:
            o, t = ov_ci[m], tl_ci[m]
            lb_rows.append(dict(pollutant=pol, model=m, mae=round(o[0], 3), mae_lo=round(o[1], 3),
                                mae_hi=round(o[2], 3), td_mae=round(t[0], 3), td_lo=round(t[1], 3),
                                td_hi=round(t[2], 3)))
        for a, b, name in PAIRS[pol]:
            pt, lo, hi, sig = dmae_ci(ov[a], ov[b], Cmat)
            pair_rows.append(dict(pollutant=pol, comparison=name, a=a, b=b, dMAE_a_minus_b=round(pt, 3),
                                  ci_lo=round(lo, 3), ci_hi=round(hi, 3), significant=sig))
        # per-origin mean abs error per model (the DM/BH unit) + station id, for run_dm_bh_h24.py
        tag = {"NO2": "no2", "PM2.5": "pm25"}[pol]
        loss_dump[f"{tag}_loss"] = np.array([[np.nanmean(u["ae_" + m]) for m in MODELS] for u in units])
        loss_dump[f"{tag}_station"] = np.array([u["station"] for u in units])
        print(f"[{pol}] {n} origin-units done")

    lb = pd.DataFrame(lb_rows); pr = pd.DataFrame(pair_rows)
    lb.to_csv(RESULTS / "leaderboard" / "significance_h24.csv", index=False)
    pr.to_csv(RESULTS / "leaderboard" / "pairwise_tests_h24.csv", index=False)
    np.savez(RESULTS / "leaderboard" / "per_origin_losses_h24.npz", models=np.array(MODELS), **loss_dump)
    pd.set_option("display.width", 170)
    for pol in ("NO2", "PM2.5"):
        print(f"\n=== {pol} @24h: MAE [95% CI] / top-decile MAE [95% CI] ===")
        for _, r in lb[lb.pollutant == pol].iterrows():
            print(f"  {r['model']:20} {r['mae']:6.2f} [{r['mae_lo']:5.2f},{r['mae_hi']:5.2f}]   "
                  f"tail {r['td_mae']:6.2f} [{r['td_lo']:5.2f},{r['td_hi']:5.2f}]")
        print(f"--- {pol} pairwise (dMAE=a-b; >0 ⇒ b better; sig=CI excl 0) ---")
        for _, r in pr[pr.pollutant == pol].iterrows():
            print(f"  {r['comparison']:34} {r['dMAE_a_minus_b']:+.2f} [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] sig={r['significant']}")
    print("\nsaved significance_h24.csv + pairwise_tests_h24.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

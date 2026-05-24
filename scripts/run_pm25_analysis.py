"""
PM2.5 @24h analysis (main venv):
  F1  — tail MAE vs persistence, block bootstrap, ALL models incl Moirai (note #1).
  F5  — TimesFM+wx vs XGBoost overall MAE, block bootstrap (reviewer note).
  MM-CP — split-CP vs MM-CP coverage on top-decile, paired coverage bootstrap (the method result).

GPU-model predictions computed here; Moirai loaded from results/preds/moirai_pm25_h24.pkl.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aq_fm_bench.conformal.coverage import covered, paired_coverage_bootstrap
from aq_fm_bench.conformal.mm_cp import fit_mm_cp
from aq_fm_bench.conformal.split_cp import conformal_quantile
from aq_fm_bench.data.processing import SPLITS
from aq_fm_bench.data.stations import TIER1
from aq_fm_bench.experiments.leaderboard import ChronosBoltBatch, fc_persistence
from aq_fm_bench.models.timesfm_model import TimesFMBatch
from aq_fm_bench.models.xgb_cache import xgb_traj_cached
from aq_fm_bench.paths import PROCESSED, RESULTS
from aq_fm_bench.stats.bootstrap import count_matrix, dmae_ci, make_boots

C, STRIDE, H, MAXH, ALPHA = 512, 168, 24, 168, 0.1
WEATHER = ["u10", "v10", "t2m_c", "sp_hpa", "blh", "tp_mm"]
MODELS = ["persistence", "chronos_bolt_base", "timesfm_2_wx", "moirai_1_1_wx", "xgboost"]


def positions(idx, ff, start, end):
    ts0, ts1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    cand = np.flatnonzero((idx >= ts0) & (idx < ts1))[::STRIDE]
    return np.array([p for p in cand if p - C + 1 >= 0 and p + MAXH < len(idx)
                     and np.isfinite(ff[p - C + 1: p + 1]).all()])


def train_positions(idx, stride=6):
    ts0, ts1 = pd.Timestamp(SPLITS["train"][0], tz="UTC"), pd.Timestamp(SPLITS["train"][1], tz="UTC")
    cand = np.flatnonzero((idx >= ts0) & (idx < ts1))[::stride]
    cap = int(np.flatnonzero(idx >= ts1)[0]) - MAXH
    return np.array([p for p in cand if 168 <= p < cap])


def boot_diff(units, key_a, key_b, n=2000, seed=42):
    """Block bootstrap over origins of mean(a) - mean(b) where units[i][key] are per-origin arrays.
    Vectorised count-matrix form (same default_rng(seed) draws as the prior loop); nanmean-safe."""
    m = len(units)
    C = count_matrix(make_boots(m, n, np.random.default_rng(seed)), m)
    return dmae_ci([u[key_a] for u in units], [u[key_b] for u in units], C)


def main() -> int:
    moirai = pickle.load(open(RESULTS / "preds" / "moirai_pm25_h24.pkl", "rb"))["wx"]
    chronos, timesfm = ChronosBoltBatch(), TimesFMBatch()

    cal_resid = {m: [] for m in MODELS}
    cal_cov = {"ws": [], "blh": [], "hour": []}
    units = []   # per test origin: abs err per model, covariates, actual, tail mask, split/mmcp coverage filled later
    test_rows = []  # flat per-point for MM-CP fitting convenience: dict per station

    def cov_targets(df, idx, pos):
        ws = df["ws10"].to_numpy("float64"); blh = df["blh"].to_numpy("float64")
        hour = idx.hour.to_numpy()
        WS = np.stack([ws[np.arange(p + 1, p + 1 + H)] for p in pos])
        BL = np.stack([blh[np.arange(p + 1, p + 1 + H)] for p in pos])
        HR = np.stack([hour[np.arange(p + 1, p + 1 + H)] for p in pos])
        return WS, BL, HR

    station_blobs = []
    for st in TIER1:
        if "PM2.5" not in st.pollutants:
            continue
        df = pd.read_parquet(PROCESSED / f"{st.code}_hourly.parquet")
        idx = pd.DatetimeIndex(df["timestamp_utc"])
        vals = df["pm25"].to_numpy("float64")
        ff = pd.Series(vals).ffill().to_numpy()
        wf = {w: pd.Series(df[w].to_numpy("float64")).ffill().bfill().to_numpy() for w in WEATHER}
        cal_pos, test_pos = positions(idx, ff, *SPLITS["cal"]), positions(idx, ff, *SPLITS["test"])
        ctx_cal = np.stack([ff[p - C + 1: p + 1] for p in cal_pos])
        ctx_test = np.stack([ff[p - C + 1: p + 1] for p in test_pos])
        cov_cal = {w: np.stack([wf[w][p - C + 1: p + 1 + H] for p in cal_pos]) for w in WEATHER}
        cov_test = {w: np.stack([wf[w][p - C + 1: p + 1 + H] for p in test_pos]) for w in WEATHER}
        a_cal = np.stack([vals[p + 1: p + 1 + H] for p in cal_pos])
        a_test = np.stack([vals[p + 1: p + 1 + H] for p in test_pos])
        q90 = np.nanpercentile(np.stack([vals[p + 1: p + 1 + MAXH] for p in test_pos]), 90)
        trp = train_positions(idx)
        pc = {  # predictions on cal
            "persistence": fc_persistence(ctx_cal, H), "chronos_bolt_base": chronos(ctx_cal, H),
            "timesfm_2_wx": timesfm.predict(ctx_cal, H, covariates=cov_cal),
            "moirai_1_1_wx": moirai[st.code]["cal_pred"],
            "xgboost": xgb_traj_cached(df, "pm25", trp, cal_pos, H, oracle=True, label=f"{st.code}_pm25_oracle_cal"),
        }
        pt = {  # predictions on test
            "persistence": fc_persistence(ctx_test, H), "chronos_bolt_base": chronos(ctx_test, H),
            "timesfm_2_wx": timesfm.predict(ctx_test, H, covariates=cov_test),
            "moirai_1_1_wx": moirai[st.code]["test_pred"],
            "xgboost": xgb_traj_cached(df, "pm25", trp, test_pos, H, oracle=True, label=f"{st.code}_pm25_oracle"),
        }
        WSc, BLc, HRc = cov_targets(df, idx, cal_pos)
        WSt, BLt, HRt = cov_targets(df, idx, test_pos)
        for m in MODELS:
            cal_resid[m].append((a_cal - pc[m]).ravel())
        cal_cov["ws"].append(WSc.ravel()); cal_cov["blh"].append(BLc.ravel()); cal_cov["hour"].append(HRc.ravel())
        station_blobs.append(dict(code=st.code, a_test=a_test, pt=pt, q90=q90,
                                  WSt=WSt, BLt=BLt, HRt=HRt, ntest=len(test_pos)))

    # fit conformal per model on pooled cal
    cw = {k: np.concatenate(v) for k, v in cal_cov.items()}
    split_q, mmcp = {}, {}
    for m in MODELS:
        r = np.concatenate(cal_resid[m])
        split_q[m] = conformal_quantile(r, ALPHA)
        mmcp[m] = fit_mm_cp(r, cw["ws"], cw["blh"], cw["hour"], alpha=ALPHA)

    # build per-origin units across stations
    oid = 0
    for blob in station_blobs:
        a = blob["a_test"]; n = blob["ntest"]
        tail_mask = a >= blob["q90"]
        for i in range(n):
            u = {"tail": tail_mask[i]}
            for m in MODELS:
                u[f"ae_{m}"] = np.abs(a[i] - blob["pt"][m][i])
            # MM-CP / split-CP coverage per point (for the chosen method model below)
            u["actual"] = a[i]; u["ws"] = blob["WSt"][i]; u["blh"] = blob["BLt"][i]; u["hour"] = blob["HRt"][i]
            u["pt"] = {m: blob["pt"][m][i] for m in MODELS}
            units.append(u); oid += 1

    # ---- F1: tail MAE vs persistence (all models) ----
    tail_units = [{"ae_" + m: u["ae_" + m][u["tail"]] for m in MODELS} for u in units if u["tail"].any()]
    # cache the F1 inputs so scripts/reproduce_f1.py can reproduce F1 in <1 min with no GPU/FMs
    with open(RESULTS / "leaderboard" / "f1_tail_inputs.pkl", "wb") as _f:
        pickle.dump({"models": MODELS, "tail_units": tail_units}, _f)
    print(f"\n=== F1: PM2.5 TOP-DECILE @24h, tail MAE vs persistence (block bootstrap, {len(tail_units)} origins) ===")
    base = np.concatenate([u["ae_persistence"] for u in tail_units]).mean()
    print(f"persistence tail MAE = {base:.3f}")
    for m in MODELS:
        if m == "persistence":
            continue
        pt_, lo, hi, sig = boot_diff(tail_units, "ae_persistence", "ae_" + m)
        mae_m = np.concatenate([u["ae_" + m] for u in tail_units]).mean()
        print(f"  {m:20} MAE {mae_m:6.3f}  dMAE {pt_:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  better-than-persist:{sig}")

    # ---- F5: TimesFM+wx vs XGBoost overall MAE ----
    all_units = [{"ae_timesfm_2_wx": u["ae_timesfm_2_wx"], "ae_xgboost": u["ae_xgboost"]} for u in units]
    pt_, lo, hi, sig = boot_diff(all_units, "ae_xgboost", "ae_timesfm_2_wx")
    print(f"\n=== F5: PM2.5 OVERALL @24h, XGBoost vs TimesFM+wx (block bootstrap) ===")
    print(f"  XGBoost {np.nanmean(np.concatenate([u['ae_xgboost'] for u in all_units])):.3f} vs "
          f"TimesFM {np.nanmean(np.concatenate([u['ae_timesfm_2_wx'] for u in all_units])):.3f} | "
          f"dMAE(xgb-tfm) {pt_:+.3f} CI[{lo:+.3f},{hi:+.3f}] distinguishable:{sig}")

    # ---- MM-CP: split-CP vs MM-CP top-decile coverage, per model ----
    print(f"\n=== MM-CP: top-decile PM2.5 coverage, split-CP vs MM-CP (target {1-ALPHA:.0%}) ===")
    print(f"{'model':20} {'split cov':>10} {'MM-CP cov':>10} {'diff':>7} {'95% CI':>18} {'sig':>5}")
    for m in MODELS:
        # per-point intervals on the tail, pooled, with origin ids for bootstrap
        ys, los, his, lom, him, oids = [], [], [], [], [], []
        for j, u in enumerate(units):
            t = u["tail"]
            if not t.any():
                continue
            p = u["pt"][m][t]
            ys.append(u["actual"][t])
            los.append(p - split_q[m]); his.append(p + split_q[m])
            lo_m, hi_m = mmcp[m].intervals(p, u["ws"][t], u["blh"][t], u["hour"][t])
            lom.append(lo_m); him.append(hi_m)
            oids.append(np.full(t.sum(), j))
        ys = np.concatenate(ys); los = np.concatenate(los); his = np.concatenate(his)
        lom = np.concatenate(lom); him = np.concatenate(him); oids = np.concatenate(oids)
        res = paired_coverage_bootstrap(ys, lom, him, los, his, oids, n_boot=2000)
        print(f"{m:20} {res['cov_b']:10.3f} {res['cov_a']:10.3f} {res['diff']:+7.3f} "
              f"[{res['ci_lo']:+.3f},{res['ci_hi']:+.3f}] {str(res['significant']):>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

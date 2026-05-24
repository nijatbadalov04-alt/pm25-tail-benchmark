"""
4D episode-conditioned MM-CP experiment (PM2.5 @24h). Compares split-CP vs 3D MM-CP vs 4D MM-CP
top-decile coverage, per base model. Bar: 4D must significantly beat BOTH split-CP and 3D MM-CP
(paired coverage bootstraps) AND reach >=80% FM coverage. Includes leakage audit + cell stats.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aq_fm_bench.conformal.coverage import coverage, covered, paired_coverage_bootstrap
from aq_fm_bench.conformal.mm_cp import (episode_cuts, episode_regime, fit_mm_cp, fit_mm_cp_4d)
from aq_fm_bench.conformal.split_cp import conformal_quantile
from aq_fm_bench.data.processing import SPLITS
from aq_fm_bench.data.stations import TIER1
from aq_fm_bench.experiments.leaderboard import ChronosBoltBatch, fc_persistence
from aq_fm_bench.models.timesfm_model import TimesFMBatch
from aq_fm_bench.models.xgb_cache import xgb_traj_cached
from aq_fm_bench.paths import PROCESSED, RESULTS

C, STRIDE, H, MAXH, ALPHA, LB = 512, 168, 24, 168, 0.1, 24
WEATHER = ["u10", "v10", "t2m_c", "sp_hpa", "blh", "tp_mm"]
MODELS = ["persistence", "chronos_bolt_base", "timesfm_2_wx", "moirai_1_1_wx", "xgboost"]
FMS = ["chronos_bolt_base", "timesfm_2_wx", "moirai_1_1_wx"]


def positions(idx, ff, s, e):
    a, b = pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")
    c = np.flatnonzero((idx >= a) & (idx < b))[::STRIDE]
    return np.array([p for p in c if p - C + 1 >= 0 and p + MAXH < len(idx)
                     and np.isfinite(ff[p - C + 1: p + 1]).all()])


def trainpos(idx, stride=6):
    a, b = pd.Timestamp(SPLITS["train"][0], tz="UTC"), pd.Timestamp(SPLITS["train"][1], tz="UTC")
    c = np.flatnonzero((idx >= a) & (idx < b))[::stride]
    cap = int(np.flatnonzero(idx >= b)[0]) - MAXH
    return np.array([p for p in c if 168 <= p < cap])


def tgt(arr, pos):
    return np.stack([arr[np.arange(p + 1, p + 1 + H)] for p in pos])


def main() -> int:
    moirai = pickle.load(open(RESULTS / "preds" / "moirai_pm25_h24.pkl", "rb"))["wx"]
    chronos, timesfm = ChronosBoltBatch(), TimesFMBatch()
    B = []  # station blobs

    for st in TIER1:
        if "PM2.5" not in st.pollutants:
            continue
        df = pd.read_parquet(PROCESSED / f"{st.code}_hourly.parquet")
        idx = pd.DatetimeIndex(df["timestamp_utc"])
        vals = df["pm25"].to_numpy("float64"); ff = pd.Series(vals).ffill().to_numpy()
        ws10 = df["ws10"].to_numpy("float64"); blh = df["blh"].to_numpy("float64")
        hour = idx.hour.to_numpy()
        wf = {w: pd.Series(df[w].to_numpy("float64")).ffill().bfill().to_numpy() for w in WEATHER}
        cal, test = positions(idx, ff, *SPLITS["cal"]), positions(idx, ff, *SPLITS["test"])
        cov_cal = {w: np.stack([wf[w][p - C + 1: p + 1 + H] for p in cal]) for w in WEATHER}
        cov_test = {w: np.stack([wf[w][p - C + 1: p + 1 + H] for p in test]) for w in WEATHER}
        trp = trainpos(idx)
        pc = {"persistence": fc_persistence(np.stack([ff[p - C + 1:p + 1] for p in cal]), H),
              "chronos_bolt_base": chronos(np.stack([ff[p - C + 1:p + 1] for p in cal]), H),
              "timesfm_2_wx": timesfm.predict(np.stack([ff[p - C + 1:p + 1] for p in cal]), H, covariates=cov_cal),
              "moirai_1_1_wx": moirai[st.code]["cal_pred"],
              "xgboost": xgb_traj_cached(df, "pm25", trp, cal, H, oracle=True, label=f"{st.code}_pm25_oracle_cal")}
        pt = {"persistence": fc_persistence(np.stack([ff[p - C + 1:p + 1] for p in test]), H),
              "chronos_bolt_base": chronos(np.stack([ff[p - C + 1:p + 1] for p in test]), H),
              "timesfm_2_wx": timesfm.predict(np.stack([ff[p - C + 1:p + 1] for p in test]), H, covariates=cov_test),
              "moirai_1_1_wx": moirai[st.code]["test_pred"],
              "xgboost": xgb_traj_cached(df, "pm25", trp, test, H, oracle=True, label=f"{st.code}_pm25_oracle")}
        cbar_cal = np.array([np.nanmean(ff[p - LB + 1:p + 1]) for p in cal])
        lb_blh = np.concatenate([blh[p - LB + 1:p + 1] for p in cal])
        lb_ws = np.concatenate([ws10[p - LB + 1:p + 1] for p in cal])
        B.append(dict(st=st, ff=ff, vals=vals, ws10=ws10, blh=blh, hour=hour, cal=cal, test=test,
                      pc=pc, pt=pt, a_cal=tgt(vals, cal), a_test=tgt(vals, test),
                      ws_cal=tgt(ws10, cal), blh_cal=tgt(blh, cal), hr_cal=tgt(hour, cal),
                      ws_t=tgt(ws10, test), blh_t=tgt(blh, test), hr_t=tgt(hour, test),
                      q90=np.nanpercentile(np.stack([vals[p + 1:p + 1 + MAXH] for p in test]), 90),
                      cbar_cal=cbar_cal, lb_blh=lb_blh, lb_ws=lb_ws))

    # episode cuts from pooled calibration
    C3, blh_lo, ws_lo = episode_cuts(np.concatenate([b["cbar_cal"] for b in B]),
                                     np.concatenate([b["lb_blh"] for b in B]),
                                     np.concatenate([b["lb_ws"] for b in B]))
    print(f"episode cuts: C3(conc 2/3)={C3:.1f}  blh_lo(1/3)={blh_lo:.0f}  ws_lo(1/3)={ws_lo:.2f}")

    # episode regime per origin + leakage audit
    fut_means, E_test_all = [], []
    for b in B:
        b["E_cal"] = episode_regime(b["ff"], b["blh"], b["ws10"], b["cal"], C3=C3, blh_lo=blh_lo, ws_lo=ws_lo)
        b["E_test"] = episode_regime(b["ff"], b["blh"], b["ws10"], b["test"], C3=C3, blh_lo=blh_lo, ws_lo=ws_lo)
        fut_means.append(np.nanmean(b["a_test"], axis=1)); E_test_all.append(b["E_test"])
    fm = np.concatenate(fut_means); Ea = np.concatenate(E_test_all)
    ok = np.isfinite(fm)
    print(f"LEAKAGE AUDIT: corr(E_origin, future-window mean conc) = {np.corrcoef(Ea[ok], fm[ok])[0,1]:.3f} "
          f"(want positive, not ~1) | episode rate test={Ea.mean():.2f}")

    # pooled calibration covariates/regime (model-independent)
    cw = dict(ws=np.concatenate([b["ws_cal"].ravel() for b in B]),
              blh=np.concatenate([b["blh_cal"].ravel() for b in B]),
              hour=np.concatenate([b["hr_cal"].ravel() for b in B]),
              reg=np.concatenate([np.repeat(b["E_cal"], H) for b in B]))

    print(f"\n{'model':18} {'split':>6} {'3D':>6} {'4D':>6}   {'4D-split CI':>20} {'4D-3D CI':>20}")
    rows = []
    for m in MODELS:
        r = np.concatenate([(b["a_cal"] - b["pc"][m]).ravel() for b in B])
        split_q = conformal_quantile(r, ALPHA)
        mm3 = fit_mm_cp(r, cw["ws"], cw["blh"], cw["hour"], alpha=ALPHA)
        mm4 = fit_mm_cp_4d(r, cw["ws"], cw["blh"], cw["hour"], cw["reg"], alpha=ALPHA)
        ys, ls, hs, l3, h3, l4, h4, oid = ([] for _ in range(8))
        o = 0
        for b in B:
            tail = b["a_test"] >= b["q90"]
            for i in range(len(b["test"])):
                t = tail[i]
                if not t.any():
                    o += 1; continue
                p = b["pt"][m][i][t]
                ws, bl, hr = b["ws_t"][i][t], b["blh_t"][i][t], b["hr_t"][i][t]
                reg = np.full(t.sum(), b["E_test"][i])
                ys.append(b["a_test"][i][t]); ls.append(p - split_q); hs.append(p + split_q)
                a3, c3_ = mm3.intervals(p, ws, bl, hr); l3.append(a3); h3.append(c3_)
                a4, c4_ = mm4.intervals(p, ws, bl, hr, reg); l4.append(a4); h4.append(c4_)
                oid.append(np.full(t.sum(), o)); o += 1
        ys = np.concatenate(ys); ls, hs = np.concatenate(ls), np.concatenate(hs)
        l3, h3, l4, h4, oid = map(np.concatenate, (l3, h3, l4, h4, oid))
        cs, c3, c4 = coverage(ys, ls, hs), coverage(ys, l3, h3), coverage(ys, l4, h4)
        v_split = paired_coverage_bootstrap(ys, l4, h4, ls, hs, oid, n_boot=2000)
        v_3d = paired_coverage_bootstrap(ys, l4, h4, l3, h3, oid, n_boot=2000)
        print(f"{m:18} {cs:6.3f} {c3:6.3f} {c4:6.3f}   "
              f"[{v_split['ci_lo']:+.3f},{v_split['ci_hi']:+.3f}]{'*' if v_split['significant'] else ' '}  "
              f"[{v_3d['ci_lo']:+.3f},{v_3d['ci_hi']:+.3f}]{'*' if v_3d['significant'] else ' '}")
        rows.append(dict(model=m, split=cs, mmcp3d=c3, mmcp4d=c4,
                         sig_vs_split=v_split["significant"], sig_vs_3d=v_3d["significant"]))

    # cell populations (from last model's 4D fit — same taxonomy for all)
    cc = mm4.cell_counts
    print(f"\n4D cell populations (36 cells): min={min(cc.values())} "
          f"median={int(np.median(list(cc.values())))} "
          f">=n_min(50)={sum(v >= 50 for v in cc.values())}/36")
    pd.DataFrame(rows).to_csv(RESULTS / "coverage" / "mmcp4d_pm25.csv", index=False)
    print("\nsaved results/coverage/mmcp4d_pm25.csv")
    print(f"\nBAR: FM 4D coverage >=80% AND significant vs BOTH split & 3D ('*' in both CI cols).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

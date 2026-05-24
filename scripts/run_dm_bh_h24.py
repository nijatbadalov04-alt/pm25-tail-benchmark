"""
Diebold-Mariano + Benjamini-Hochberg significance @24h: a second, parametric
significance method complementing the block-bootstrap CIs, on the overall-MAE pairwise claims.

Reads per-origin mean-abs-error losses dumped by run_significance_leaderboard.py
(results/leaderboard/per_origin_losses_h24.npz). For each pollutant, tests:
  - every model vs persistence (does it beat the naive floor?),
  - the named F2-F6 pairwise comparisons,
then BH-corrects the p-values over that family (per pollutant) and cross-checks that DM+BH agrees
with the bootstrap pairwise tests. Writes results/leaderboard/dm_bh_h24.csv.

DM truncation lag = 0 (7-day stride => non-overlapping 24h windows => per-origin losses not
mechanically autocorrelated; DM then coincides with a paired t-test). lag=1 reported as robustness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # import constants from the sibling script
from aq_fm_bench.paths import RESULTS
from aq_fm_bench.stats.dm_test import benjamini_hochberg, diebold_mariano
from run_significance_leaderboard import MODELS, PAIRS  # single source of truth for the comparison set

TAG = {"NO2": "no2", "PM2.5": "pm25"}
ALPHA = 0.05


def comparisons(pol):
    """(a, b, name): vs-persistence for every model, plus the named F2-F6 pairs. >0 mean_diff => b better."""
    out = [("persistence", m, f"{m} vs persistence") for m in MODELS if m != "persistence"]
    out += [(a, b, name) for a, b, name in PAIRS[pol]]
    return out


def main() -> int:
    npz = RESULTS / "leaderboard" / "per_origin_losses_h24.npz"
    if not npz.exists():
        print(f"missing {npz} — run scripts/run_significance_leaderboard.py first"); return 1
    d = np.load(npz, allow_pickle=True)
    models = list(d["models"])
    boot = pd.read_csv(RESULTS / "leaderboard" / "pairwise_tests_h24.csv")  # for the agreement cross-check

    rows = []
    for pol in ("NO2", "PM2.5"):
        loss = d[f"{TAG[pol]}_loss"]                       # [n_units, n_models]
        col = {m: loss[:, models.index(m)] for m in models}
        comps = comparisons(pol)
        recs = []
        for a, b, name in comps:
            r0 = diebold_mariano(col[a], col[b], lag=0)
            r1 = diebold_mariano(col[a], col[b], lag=1)
            recs.append(dict(pollutant=pol, comparison=name, a=a, b=b,
                             mean_diff_a_minus_b=round(r0["mean_diff"], 4), dm_stat=round(r0["dm"], 3),
                             p_raw=r0["p"], n=r0["n"], p_raw_lag1=r1["p"]))
        p = np.array([x["p_raw"] for x in recs])
        reject, q = benjamini_hochberg(p, ALPHA)
        for x, rj, qq in zip(recs, reject, q):
            x["q_bh"] = round(float(qq), 4); x["significant_bh"] = bool(rj)
            x["p_raw"] = round(x["p_raw"], 4); x["p_raw_lag1"] = round(x["p_raw_lag1"], 4)
        rows += recs

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "leaderboard" / "dm_bh_h24.csv", index=False)

    # ---- report + agreement cross-check vs bootstrap ----
    pd.set_option("display.width", 180)
    for pol in ("NO2", "PM2.5"):
        sub = out[out.pollutant == pol]
        print(f"\n=== {pol} @24h — Diebold-Mariano + Benjamini-Hochberg (FDR {ALPHA}; family n={len(sub)}) ===")
        print(f"{'comparison':34} {'mean_diff(a-b)':>14} {'DM':>8} {'p_raw':>8} {'q_BH':>8} {'sig':>5}")
        for _, r in sub.iterrows():
            print(f"  {r['comparison']:32} {r['mean_diff_a_minus_b']:+14.3f} {r['dm_stat']:+8.2f} "
                  f"{r['p_raw']:8.4f} {r['q_bh']:8.4f} {str(r['significant_bh']):>5}")

    print("\n=== Agreement: DM+BH vs block-bootstrap on the F2-F6 named pairs ===")
    agree = 0; total = 0
    for _, br in boot.iterrows():
        m = out[(out.pollutant == br["pollutant"]) & (out.a == br["a"]) & (out.b == br["b"])]
        if m.empty:
            continue
        total += 1
        dm_sig = bool(m.iloc[0]["significant_bh"]); bt_sig = bool(br["significant"])
        agree += int(dm_sig == bt_sig)
        flag = "OK " if dm_sig == bt_sig else "XX "
        print(f"  {flag}{br['comparison']:34} bootstrap_sig={bt_sig!s:5}  DM+BH_sig={dm_sig!s:5}")
    print(f"\nAgreement: {agree}/{total} named pairs concordant between the two methods.")
    print("saved results/leaderboard/dm_bh_h24.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

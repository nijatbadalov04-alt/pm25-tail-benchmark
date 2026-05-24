"""
Minimal replication of F1 — the headline finding — in <1 minute, with NO GPU and NO foundation-model
downloads. Reproduces:

    "Across naive, classical, and three foundation-model architectures, no model significantly beats
     persistence on top-decile PM2.5 at 24 h (all dMAE-vs-persistence 95% CIs span zero)."

from the cached per-origin tail errors in results/leaderboard/f1_tail_inputs.pkl (committed to the
repo). This is the clean-machine reproducibility entry point.

The FULL path that regenerates the cache from raw data + foundation-model inference is
scripts/run_pm25_analysis.py (needs the GPU env `.venv` plus the Moirai env `.venv-fm`).

Run:  python scripts/reproduce_f1.py    (only needs numpy + the aq_fm_bench package)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aq_fm_bench.paths import RESULTS
from aq_fm_bench.stats.bootstrap import count_matrix, dmae_ci, make_boots


def main() -> int:
    cache = RESULTS / "leaderboard" / "f1_tail_inputs.pkl"
    if not cache.exists():
        print(f"missing {cache}\nRegenerate it with the full pipeline: python scripts/run_pm25_analysis.py")
        return 1
    d = pickle.load(open(cache, "rb"))
    models, tail_units = d["models"], d["tail_units"]
    n = len(tail_units)

    # identical block bootstrap to run_pm25_analysis.py: default_rng(42), 2000 resamples of origins
    C = count_matrix(make_boots(n, 2000, np.random.default_rng(42)), n)
    pers = [u["ae_persistence"] for u in tail_units]
    base = float(np.concatenate(pers).mean())

    print(f"F1 — PM2.5 top-decile (episodes) @24h | {n} tail origin-units | 2000 bootstrap iterations")
    print(f"persistence tail MAE = {base:.3f} ug/m3\n")
    print(f"{'model':22}{'tail MAE':>10}{'dMAE(pers-mdl)':>16}{'95% CI':>20}{'beats persist?':>16}")
    any_beats = False
    for m in models:
        if m == "persistence":
            continue
        arrs = [u["ae_" + m] for u in tail_units]
        mae_m = float(np.concatenate(arrs).mean())
        diff, lo, hi, sig = dmae_ci(pers, arrs, C)        # MAE(pers) - MAE(model); >0 => model better
        beats = bool(sig and diff > 0)
        any_beats |= beats
        print(f"  {m:20}{mae_m:10.3f}{diff:+16.3f}   [{lo:+.3f},{hi:+.3f}]{('YES' if beats else 'no'):>14}")

    print()
    if any_beats:
        print("UNEXPECTED: a model significantly beats persistence — F1 was NOT reproduced.")
        return 1
    print("F1 REPRODUCED: no model significantly beats persistence on the PM2.5 episode tail "
          "(every dMAE 95% CI spans 0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

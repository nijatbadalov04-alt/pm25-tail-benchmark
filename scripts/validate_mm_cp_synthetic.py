"""
Validate MM-CP mechanics on synthetic heteroscedastic data: residual scale is large in a
'stagnation' regime (low wind x low BLH) and small elsewhere. Split-CP (one global quantile)
must under-cover in stagnation and over-cover in calm; MM-CP (per-cell) must restore ~nominal
coverage in both. Also exercises the paired coverage bootstrap.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aq_fm_bench.conformal.coverage import coverage, mean_width, paired_coverage_bootstrap
from aq_fm_bench.conformal.mm_cp import fit_mm_cp
from aq_fm_bench.conformal.split_cp import split_conformal

rng = np.random.default_rng(0)
N, alpha = 40000, 0.1
ws = rng.uniform(0, 10, N)
blh = rng.uniform(0, 2000, N)
hour = rng.integers(0, 24, N)
stagnation = (ws < 10 / 3) & (blh < 2000 / 3)          # low wind AND low BLH
scale = np.where(stagnation, 5.0, 1.0)                  # 5x noise in stagnation
y_pred = np.zeros(N)
resid = rng.normal(0, scale)                            # y_true - y_pred
y_true = y_pred + resid

half = N // 2
cal, test = slice(0, half), slice(half, N)

lo_s, hi_s, q = split_conformal(resid[cal], y_pred[test], alpha)
mm = fit_mm_cp(resid[cal], ws[cal], blh[cal], hour[cal], alpha)
lo_m, hi_m = mm.intervals(y_pred[test], ws[test], blh[test], hour[test])

yt = y_true[test]
st = stagnation[test]
print(f"target coverage = {1-alpha:.0%}  (split-CP global q={q:.2f})\n")
print(f"{'subset':16} {'split-CP cov':>12} {'MM-CP cov':>10} {'split-CP wid':>13} {'MM-CP wid':>10}")
for label, m in [("marginal", np.ones(len(yt), bool)), ("STAGNATION cell", st), ("calm cells", ~st)]:
    print(f"{label:16} {coverage(yt[m], lo_s[m], hi_s[m]):12.3f} {coverage(yt[m], lo_m[m], hi_m[m]):10.3f}"
          f" {mean_width(lo_s[m], hi_s[m]):13.2f} {mean_width(lo_m[m], hi_m[m]):10.2f}")

oid = np.arange(N)[test]
res = paired_coverage_bootstrap(yt[st], lo_m[st], hi_m[st], lo_s[st], hi_s[st], oid[st], n_boot=2000)
print(f"\nStagnation paired coverage bootstrap (MM-CP − split-CP):")
print(f"  MM-CP {res['cov_a']:.3f} vs split-CP {res['cov_b']:.3f} | diff {res['diff']:+.3f} "
      f"95% CI [{res['ci_lo']:+.3f}, {res['ci_hi']:+.3f}] | significant: {res['significant']}")
print("\nExpect: split-CP undercovers stagnation, overcovers calm; MM-CP ~nominal both; diff significant.")

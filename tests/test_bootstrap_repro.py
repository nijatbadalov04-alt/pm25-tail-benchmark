"""
Correctness gate: the vectorised count-matrix bootstrap must reproduce the naive
Python-loop estimator (np.nanmean over concatenated per-unit arrays) to floating-point
tolerance, given the SAME resample indices. Covers: overall MAE, NaN handling, variable-length
tail subsets, multi-model batching, and the paired dMAE difference.

Run:  .venv\Scripts\python.exe -m pytest tests/test_bootstrap_repro.py -q
or:   .venv\Scripts\python.exe tests/test_bootstrap_repro.py   (prints PASS/FAIL, no pytest needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aq_fm_bench.stats.bootstrap import (count_matrix, dmae_ci, mae_ci, mae_ci_multi, unit_sums)

H = 24
ATOL = 1e-9


def _make_units(n, rng, with_nan=True, tail=False):
    """n per-unit abs-error arrays of length H; optionally inject NaNs and take a tail subset."""
    arrs = []
    for _ in range(n):
        a = rng.gamma(2.0, 3.0, size=H)
        if with_nan and rng.random() < 0.3:
            a[rng.integers(0, H, rng.integers(1, 4))] = np.nan
        if tail:
            mask = a >= np.nanpercentile(a, 70)  # variable-length subset per unit
            a = a[mask]
        arrs.append(a)
    return arrs


def _naive_mae_dist(arrays, boots):
    pt = np.nanmean(np.concatenate(arrays))
    dist = np.array([np.nanmean(np.concatenate([arrays[i] for i in b])) for b in boots])
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def _naive_dmae_dist(a, b, boots):
    pt = np.nanmean(np.concatenate(a)) - np.nanmean(np.concatenate(b))
    dist = np.array([np.nanmean(np.concatenate([a[i] for i in s]))
                     - np.nanmean(np.concatenate([b[i] for i in s])) for s in boots])
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def _check(name, naive, fast):
    ok = np.allclose(naive, fast, atol=ATOL, rtol=0, equal_nan=True)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: naive={tuple(round(x,6) for x in naive)} "
          f"fast={tuple(round(x,6) for x in fast)}")
    assert ok, f"{name} mismatch: {naive} vs {fast}"


def run():
    rng = np.random.default_rng(7)
    n = 120
    boots = [rng.integers(0, n, n) for _ in range(500)]
    C = count_matrix(boots, n)

    # 1. overall MAE CI (no NaN)
    arr = _make_units(n, np.random.default_rng(1), with_nan=False)
    _check("overall MAE (no nan)", _naive_mae_dist(arr, boots), mae_ci(arr, C))

    # 2. overall MAE CI with NaN
    arr = _make_units(n, np.random.default_rng(2), with_nan=True)
    _check("overall MAE (with nan)", _naive_mae_dist(arr, boots), mae_ci(arr, C))

    # 3. variable-length tail subsets (some units may be empty)
    arr_t = _make_units(n, np.random.default_rng(3), with_nan=True, tail=True)
    _check("tail MAE (var-length)", _naive_mae_dist(arr_t, boots), mae_ci(arr_t, C))

    # 4. multi-model batched == per-model loop
    models = {f"m{j}": _make_units(n, np.random.default_rng(10 + j), with_nan=True) for j in range(5)}
    multi = mae_ci_multi(models, C)
    for m, a in models.items():
        _check(f"multi vs single [{m}]", mae_ci(a, C), multi[m])

    # 5. paired dMAE difference
    a = _make_units(n, np.random.default_rng(20), with_nan=True)
    b = _make_units(n, np.random.default_rng(21), with_nan=True)
    naive = _naive_dmae_dist(a, b, boots)
    pt, lo, hi, _ = dmae_ci(a, b, C)
    _check("paired dMAE", naive, (pt, lo, hi))

    # 6. count_matrix integrity: row sums == n (each resample draws n units)
    assert np.all(C.sum(1) == n), "count matrix row sums != n"
    print(f"  [PASS] count_matrix row-sums == n ({n}) for all {len(boots)} resamples")

    # 7. torch backend (if available) matches numpy
    try:
        import torch
        if torch.cuda.is_available():
            S, N = unit_sums(arr)
            d_np = (C @ S) / (C @ N)
            from aq_fm_bench.stats.bootstrap import _dist
            d_t = _dist(S, N, C, "torch")
            _check("torch-GPU backend", (float(np.percentile(d_np, 2.5)),),
                   (float(np.percentile(d_t, 2.5)),))
        else:
            print("  [skip] torch present but no CUDA")
    except ImportError:
        print("  [skip] torch not installed")

    print("\nALL BOOTSTRAP REPRO CHECKS PASSED")


def test_bootstrap_reproduces_naive():
    run()


if __name__ == "__main__":
    run()

"""Conformal evaluation: coverage, interval width, Winkler, and the paired coverage bootstrap."""
from __future__ import annotations

import numpy as np


def covered(y_true, lo, hi) -> np.ndarray:
    y = np.asarray(y_true, "float64")
    return (y >= np.asarray(lo, "float64")) & (y <= np.asarray(hi, "float64"))


def coverage(y_true, lo, hi) -> float:
    y = np.asarray(y_true, "float64")
    m = np.isfinite(y)
    return float(covered(y, lo, hi)[m].mean()) if m.any() else float("nan")


def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi, "float64") - np.asarray(lo, "float64")))


def winkler(y_true, lo, hi, alpha: float = 0.1) -> float:
    """Winkler/interval score @ (1-alpha): width + miscoverage penalty (lower is better)."""
    y = np.asarray(y_true, "float64"); lo = np.asarray(lo, "float64"); hi = np.asarray(hi, "float64")
    m = np.isfinite(y)
    y, lo, hi = y[m], lo[m], hi[m]
    w = hi - lo
    below = y < lo
    above = y > hi
    score = w + (2 / alpha) * (lo - y) * below + (2 / alpha) * (y - hi) * above
    return float(np.mean(score))


def paired_coverage_bootstrap(y_true, lo_a, hi_a, lo_b, hi_b, origin_ids,
                              n_boot: int = 2000, seed: int = 42):
    """
    Block bootstrap over origins on coverage(A) - coverage(B). Returns dict with point diff,
    95% CI, and whether the CI excludes 0 (A significantly different coverage from B).
    A = e.g. MM-CP, B = split-CP.

    Vectorised: each origin contributes (Sa,Sb,N) = (#covered_A, #covered_B, #finite-points);
    a resample with origin multiplicities m gives cov_X(resample) = (m@Sx)/(m@N). The origin
    resamples are drawn with the same rng.choice sequence as the original loop, so results are
    unchanged — only the per-iteration concatenation is replaced by two matmuls.
    """
    y = np.asarray(y_true, "float64")
    cov_a = covered(y, lo_a, hi_a).astype("float64")
    cov_b = covered(y, lo_b, hi_b).astype("float64")
    finite = np.isfinite(y)
    origin_ids = np.asarray(origin_ids)

    # group point indices by origin (finite only); sorted unique origin values
    uniq = np.unique(origin_ids)
    by_origin = {o: np.flatnonzero((origin_ids == o) & finite) for o in uniq}
    by_origin = {o: idx for o, idx in by_origin.items() if len(idx)}
    origins = np.array(list(by_origin.keys()))

    Sa = np.array([cov_a[by_origin[o]].sum() for o in origins], dtype="float64")
    Sb = np.array([cov_b[by_origin[o]].sum() for o in origins], dtype="float64")
    N = np.array([len(by_origin[o]) for o in origins], dtype="float64")
    point = Sa.sum() / N.sum() - Sb.sum() / N.sum()

    # count matrix from the same rng.choice draws (origins is sorted -> searchsorted gives index)
    rng = np.random.default_rng(seed)
    M = np.zeros((n_boot, len(origins)), dtype="float64")
    for i in range(n_boot):
        samp = rng.choice(origins, size=len(origins), replace=True)
        M[i] = np.bincount(np.searchsorted(origins, samp), minlength=len(origins))
    with np.errstate(invalid="ignore", divide="ignore"):
        den = M @ N
        diffs = (M @ Sa) / den - (M @ Sb) / den
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "significant": bool(lo > 0 or hi < 0),
            "cov_a": float(cov_a[finite].mean()), "cov_b": float(cov_b[finite].mean())}

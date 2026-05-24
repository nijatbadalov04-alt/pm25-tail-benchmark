"""
Vectorised grouped (block) bootstrap over forecast-origin units.

Replaces the per-script Python-loop pattern that dominated the significance wall-clock:

    boots = [rng.integers(0, n, n) for _ in range(n_boot)]
    dist  = np.array([np.nanmean(np.concatenate([arr[i] for i in b])) for b in boots])

with an algebraically *identical* count-matrix formulation. For a resample whose unit i
appears c[i] times,

    MAE(resample) = sum_i c[i]*nansum(arr[i]) / sum_i c[i]*count_finite(arr[i])
                  = (C @ S) / (C @ N)

where S[i]=nansum(arr[i]), N[i]=#finite(arr[i]), and C[b]=bincount of resample b's indices.
This is the SAME estimator as nanmean-over-concatenation (a unit contributing its values c
times contributes c× to both numerator and denominator), so given the SAME `boots` it
reproduces prior results to floating-point tolerance — but it turns n_boot Python
concatenations into two matmuls (~100-1000x faster). Equivalence is proven in
tests/test_bootstrap_repro.py.

Correctness is preserved by feeding in the caller's existing
`boots` so the RNG draw order is unchanged; only the reduction is vectorised. The GPU `backend`
is optional and off by default — at our problem sizes the matmul is sub-millisecond on the CPU,
so the GPU is reserved for where it actually pays (FM inference). It exists for honesty/scaling.
"""
from __future__ import annotations

import numpy as np

PCT = (2.5, 97.5)


# ---- resample bookkeeping ------------------------------------------------
def make_boots(n_unit: int, n_boot: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Canonical resample draw: `n_boot` resamples of `n_unit` indices, with replacement.
    Pass a seeded Generator so the draw sequence matches the original scripts exactly."""
    return [rng.integers(0, n_unit, n_unit) for _ in range(n_boot)]


def count_matrix(boots, n_unit: int) -> np.ndarray:
    """List of index resamples -> [n_boot, n_unit] float multiplicity matrix (for C @ S)."""
    C = np.zeros((len(boots), n_unit), dtype=np.float64)
    for b, idx in enumerate(boots):
        C[b] = np.bincount(np.asarray(idx, dtype=np.int64), minlength=n_unit)
    return C


def unit_sums(arrays) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit nansum S[i] and finite-count N[i] from a list of 1D arrays.

    Handles NaN (e.g. PM2.5 gaps) and variable length (e.g. tail-only subsets): nanmean over a
    concatenation of any subset of units equals (sum of their S) / (sum of their N)."""
    S = np.fromiter((np.nansum(a) for a in arrays), dtype=np.float64, count=len(arrays))
    N = np.fromiter((np.isfinite(np.asarray(a, dtype="float64")).sum() for a in arrays),
                    dtype=np.float64, count=len(arrays))
    return S, N


# ---- core reduction ------------------------------------------------------
def _dist(S: np.ndarray, N: np.ndarray, C: np.ndarray, backend: str) -> np.ndarray:
    """(C @ S) / (C @ N). S,N: [n_unit] or [n_unit, M]. Returns [n_boot] or [n_boot, M]."""
    if backend == "torch":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        Ct = torch.as_tensor(C, device=dev, dtype=torch.float64)
        num = Ct @ torch.as_tensor(S, device=dev, dtype=torch.float64)
        den = Ct @ torch.as_tensor(N, device=dev, dtype=torch.float64)
        return (num / den).cpu().numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (C @ S) / (C @ N)


# ---- public estimators ---------------------------------------------------
def mae_ci(arrays, C: np.ndarray, *, pct=PCT, backend: str = "numpy"):
    """Point MAE + bootstrap CI for one model. `arrays`: list of n_unit 1D error arrays."""
    S, N = unit_sums(arrays)
    point = S.sum() / N.sum()
    lo, hi = np.percentile(_dist(S, N, C, backend), pct)
    return float(point), float(lo), float(hi)


def mae_ci_multi(arrays_by_model: dict, C: np.ndarray, *, pct=PCT, backend: str = "numpy"):
    """All models in two matmuls. Returns {model: (point, lo, hi)}. Same C (paired across models)."""
    models = list(arrays_by_model)
    cols = [unit_sums(arrays_by_model[m]) for m in models]
    S = np.column_stack([c[0] for c in cols])   # [n_unit, M]
    N = np.column_stack([c[1] for c in cols])
    point = S.sum(0) / N.sum(0)                  # [M]
    dist = _dist(S, N, C, backend)               # [n_boot, M]
    lo, hi = np.percentile(dist, pct, axis=0)
    return {m: (float(point[j]), float(lo[j]), float(hi[j])) for j, m in enumerate(models)}


def dmae_ci(arrays_a, arrays_b, C: np.ndarray, *, pct=PCT, backend: str = "numpy"):
    """CI on dMAE = MAE(a) - MAE(b) on the SAME resamples (paired). >0 => b better.
    Returns (point, lo, hi, significant) where significant = CI excludes 0."""
    Sa, Na = unit_sums(arrays_a)
    Sb, Nb = unit_sums(arrays_b)
    point = Sa.sum() / Na.sum() - Sb.sum() / Nb.sum()
    dist = _dist(Sa, Na, C, backend) - _dist(Sb, Nb, C, backend)
    lo, hi = np.percentile(dist, pct)
    return float(point), float(lo), float(hi), bool(lo > 0 or hi < 0)

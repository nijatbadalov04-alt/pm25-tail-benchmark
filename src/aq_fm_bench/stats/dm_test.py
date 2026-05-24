"""
Diebold-Mariano forecast-comparison test (Harvey-Leybourne-Newbold small-sample correction)
+ Benjamini-Hochberg false-discovery-rate control. Complements the block-bootstrap CIs: a
second, parametric significance method on the overall-MAE pairwise claims.

Unit of analysis = the forecast origin (7-day stride); loss = per-origin mean absolute error.
The DM truncation lag defaults to 0 because the 7-day stride makes 24h-horizon forecast windows
*non-overlapping*, so per-origin losses are not mechanically autocorrelated — at lag 0 the DM
statistic coincides with a paired t-test on per-origin losses (we keep the DM/HLN machinery so the
multi-horizon extension, where windows can abut, is a one-line lag change). A Newey-West HAC lag
is available as a robustness option.
"""
from __future__ import annotations

import numpy as np


def _t_sf(x: np.ndarray | float, df: int) -> np.ndarray | float:
    """Upper-tail Student-t survival function; scipy if available, else a normal approximation."""
    try:
        from scipy import stats
        return stats.t.sf(x, df=df)
    except ImportError:  # graceful fallback (df large -> ~normal)
        from math import erfc, sqrt
        return 0.5 * erfc(np.asarray(x) / sqrt(2.0))


def diebold_mariano(loss_a, loss_b, *, lag: int = 0):
    """Paired DM test on two per-origin loss series (lower loss = better).

    Returns dict: dm (HLN-corrected statistic), p (two-sided), n, mean_diff = mean(loss_a-loss_b)
    (>0 => B better, matching dMAE_a_minus_b sign convention), and a sign of which is better.
    """
    a = np.asarray(loss_a, "float64"); b = np.asarray(loss_b, "float64")
    d = a - b
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3 or np.allclose(d, 0):
        return {"dm": 0.0, "p": 1.0, "n": int(n), "mean_diff": float(np.mean(d) if n else 0.0)}
    dbar = d.mean()
    dev = d - dbar
    gamma0 = np.mean(dev * dev)
    s = gamma0
    for k in range(1, lag + 1):
        s += 2.0 * np.mean(dev[k:] * dev[:-k])
    var_dbar = s / n
    if var_dbar <= 0:
        return {"dm": 0.0, "p": 1.0, "n": int(n), "mean_diff": float(dbar)}
    dm = dbar / np.sqrt(var_dbar)
    h = lag + 1                                   # forecast "horizon" for the HLN correction
    hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    dm_hln = dm * hln
    p = float(2.0 * _t_sf(abs(dm_hln), df=n - 1))
    return {"dm": float(dm_hln), "p": min(p, 1.0), "n": int(n), "mean_diff": float(dbar)}


def benjamini_hochberg(pvals, alpha: float = 0.05):
    """BH FDR control. Returns (reject[bool array], qvalues[array]) aligned to input order."""
    p = np.asarray(pvals, "float64")
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1)                       # ascending-sorted ranks
    q_raw = ranked * m / ranks                        # m/rank * p_(rank)
    q_sorted = np.minimum.accumulate(q_raw[::-1])[::-1]   # step-up: min over higher ranks
    q = np.empty(m); q[order] = np.minimum(q_sorted, 1.0)
    reject = q <= alpha
    return reject, q

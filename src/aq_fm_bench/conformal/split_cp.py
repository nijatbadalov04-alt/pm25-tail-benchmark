"""Split conformal prediction (Vovk et al., 2005). The baseline UQ method."""
from __future__ import annotations

import numpy as np


def conformal_quantile(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample conformal quantile of |residuals| for target coverage 1-alpha."""
    r = np.abs(np.asarray(residuals, dtype="float64"))
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return float("nan")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(r, level, method="higher"))


def split_conformal(cal_resid: np.ndarray, test_pred: np.ndarray, alpha: float = 0.1):
    """Return (lo, hi) symmetric intervals with marginal 1-alpha coverage under exchangeability."""
    q = conformal_quantile(cal_resid, alpha)
    test_pred = np.asarray(test_pred, dtype="float64")
    return test_pred - q, test_pred + q, q

"""
TimesFM-2.0 wrapper (GPU, main venv). Supports zero-shot and weather-as-known-future-covariates
(the fair counterpart to oracle-weather XGBoost). Exog schema documented in docs/EXOG_SCHEMA.md.

Covariate arrays span the FULL context+horizon window per series (TimesFM's covariate API
regresses covariates over context and applies over the horizon). Point forecast = median.
"""
from __future__ import annotations

import time

import numpy as np


class TimesFMBatch:
    # per_core_batch_size=128: throughput sweet spot from scripts/tune_gpu_batch.py (1.7x faster
    # than 32, plateaus by 256). Predictions are bitwise-identical across batch sizes (fp32), so
    # this changes no results. VRAM stays ~2 GB regardless — floored by the 500M fp32 weights,
    # not the batch — so enlarging further only wastes launches without filling the card.
    def __init__(self, context_len: int = 512, horizon_len: int = 256, per_core_batch_size: int = 128):
        import timesfm
        self._timesfm = timesfm
        t0 = time.time()
        self.tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu", context_len=context_len, horizon_len=horizon_len,
                num_layers=50, use_positional_embedding=False,
                point_forecast_mode="median", per_core_batch_size=per_core_batch_size,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )
        print(f"[timesfm] loaded TimesFM-2.0 on GPU in {time.time()-t0:.1f}s")

    def predict(self, contexts: np.ndarray, horizon: int, covariates: dict | None = None) -> np.ndarray:
        """contexts [N, C]; covariates {name: array [N, C+horizon]} (known future). Returns [N, horizon]."""
        inputs = [np.asarray(c, dtype="float32") for c in contexts]
        freq = [0] * len(inputs)
        if covariates is None:
            pt, _ = self.tfm.forecast(inputs, freq=freq)
            return np.asarray(pt)[:, :horizon]
        dyn = {name: [np.asarray(cov[i], dtype="float32") for i in range(len(inputs))]
               for name, cov in covariates.items()}
        res = self.tfm.forecast_with_covariates(
            inputs, dynamic_numerical_covariates=dyn, freq=freq,
            xreg_mode="xreg + timesfm", normalize_xreg_target_per_input=True,
        )
        # res[0] = per-series combined forecast (list of length-horizon arrays)
        pt = np.asarray(res[0] if isinstance(res, (tuple, list)) else res)
        return pt[:, :horizon]

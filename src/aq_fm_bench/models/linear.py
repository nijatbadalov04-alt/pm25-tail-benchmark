"""
Ridge-regression sanity baseline on the SAME features as XGBoost (median-imputed +
standardised). Purpose: isolate how much of XGBoost's skill is linear
meteorology vs nonlinearity. If linear_oracle ≈ xgboost_oracle, the NO₂ win is future-weather
information, not nonlinear modelling — and it pre-empts the "did you try a linear model?" review.
"""
from __future__ import annotations

import numpy as np

from aq_fm_bench.models.xgb import build_features


class LinearForecaster:
    def __init__(self, oracle_weather: bool = True, alpha: float = 1.0):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.oracle_weather = oracle_weather
        self.pipe = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)
        )

    def fit(self, df, target_col, train_positions, max_lead: int = 168):
        leads = np.arange(1, max_lead + 1)
        X, y, *_ = build_features(df, target_col, train_positions, leads,
                                  include_future_weather=self.oracle_weather)
        m = np.isfinite(y)
        self.pipe.fit(X[m], y[m])
        return self

    def predict_traj(self, df, target_col, test_positions, horizon: int) -> np.ndarray:
        leads = np.arange(1, horizon + 1)
        X, _, PP, LL, _, _ = build_features(df, target_col, test_positions, leads,
                                            include_future_weather=self.oracle_weather)
        pred = self.pipe.predict(X)
        tp = np.asarray(test_positions, dtype=np.int64)
        out = np.full((len(tp), horizon), np.nan)
        out[np.searchsorted(tp, PP), LL - 1] = pred
        return out

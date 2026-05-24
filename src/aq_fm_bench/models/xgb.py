"""
Weather-aware direct multi-horizon XGBoost (GPU). The serious trained baseline the FMs must beat.

Design: one model per (station, pollutant), trained on (origin, lead) pairs with the lead time
as a feature, so a single model forecasts the whole 1..H trajectory (matching how the FMs
output a full path). Features = pollutant lags + rolling stats at the origin, calendar of the
target hour, the lead, and ERA5 weather at BOTH origin and target time (oracle future weather —
a deliberately strong baseline; Chronos has no exog, so this asymmetry is part of RQ3).
XGBoost handles NaNs natively, so no imputation of missing lags/weather is needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 3, 6, 24, 168]
WEATHER = ["t2m_c", "blh", "ws10", "u10", "v10", "sp_hpa", "tp_mm"]


def build_features(df: pd.DataFrame, target_col: str, positions, leads,
                   include_future_weather: bool = True):
    """Vectorised (origin x lead) feature matrix. Returns X, y_target, origin_pos, lead, target_pos, names.

    include_future_weather=False -> `xgboost_no_oracle`: only weather up to the forecast
    origin (what a real deployment has); drops the target-time weather features.
    """
    N = len(df)
    y = df[target_col].to_numpy("float64")
    ts = pd.DatetimeIndex(df["timestamp_utc"])
    hour, dow, month = ts.hour.to_numpy(), ts.dayofweek.to_numpy(), ts.month.to_numpy()
    weather = {w: df[w].to_numpy("float64") for w in WEATHER if w in df.columns}
    s = df[target_col]
    rm3 = s.rolling(3).mean().to_numpy()
    rm24 = s.rolling(24).mean().to_numpy()
    rs24 = s.rolling(24).std().to_numpy()

    P = np.asarray(positions, dtype=np.int64)
    L = np.asarray(leads, dtype=np.int64)
    PP = np.repeat(P, len(L))
    LL = np.tile(L, len(P))
    T = PP + LL
    ok = T < N
    PP, LL, T = PP[ok], LL[ok], T[ok]

    def lag(arr, pos, k):
        out = np.full(pos.shape, np.nan)
        m = pos - k >= 0
        out[m] = arr[pos[m] - k]
        return out

    feats = {"y0": y[PP]}
    for k in LAGS:
        feats[f"lag{k}"] = lag(y, PP, k)
    feats["rm3"], feats["rm24"], feats["rs24"] = rm3[PP], rm24[PP], rs24[PP]
    feats["lead"] = LL.astype("float64")
    feats["hsin"] = np.sin(2 * np.pi * hour[T] / 24); feats["hcos"] = np.cos(2 * np.pi * hour[T] / 24)
    feats["dsin"] = np.sin(2 * np.pi * dow[T] / 7);   feats["dcos"] = np.cos(2 * np.pi * dow[T] / 7)
    feats["msin"] = np.sin(2 * np.pi * month[T] / 12); feats["mcos"] = np.cos(2 * np.pi * month[T] / 12)
    feats["wknd"] = (dow[T] >= 5).astype("float64")
    for w, arr in weather.items():
        feats[f"{w}_o"] = arr[PP]               # weather at origin (always available)
        if include_future_weather:
            feats[f"{w}_t"] = arr[T]            # weather at target (oracle future)
    names = list(feats)
    X = np.column_stack([feats[k] for k in names])
    return X, y[T], PP, LL, T, names


class XGBDirectForecaster:
    def __init__(self, device: str = "cuda", n_estimators: int = 400,
                 max_depth: int = 7, lr: float = 0.05, oracle_weather: bool = True):
        import xgboost as xgb
        self._xgb = xgb
        self.oracle_weather = oracle_weather
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
                           subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                           device=device, random_state=42)
        self.model = None

    def fit(self, df, target_col, train_positions, max_lead: int = 168):
        leads = np.arange(1, max_lead + 1)
        X, y, *_ = build_features(df, target_col, train_positions, leads,
                                  include_future_weather=self.oracle_weather)
        m = np.isfinite(y)
        self.model = self._xgb.XGBRegressor(**self.params)
        self.model.fit(X[m], y[m])
        return self

    def predict_traj(self, df, target_col, test_positions, horizon: int) -> np.ndarray:
        leads = np.arange(1, horizon + 1)
        X, _, PP, LL, _, _ = build_features(df, target_col, test_positions, leads,
                                            include_future_weather=self.oracle_weather)
        pred = self.model.predict(X)
        tp = np.asarray(test_positions, dtype=np.int64)
        out = np.full((len(tp), horizon), np.nan)
        row = np.searchsorted(tp, PP)
        out[row, LL - 1] = pred
        return out

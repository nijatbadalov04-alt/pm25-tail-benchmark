"""
FM-with-weather leaderboard. Identical test origins; trajectory MAE; OVERALL + TOP-DECILE for
both pollutants; wall-clock inference time per forecast (GPU/CPU asymmetry transparent).

Models: persistence, seasonal-naive(24/168), linear(±oracle weather), XGBoost(±oracle weather),
Chronos-Bolt (no exog), TimesFM-2.0 (+ 6 ERA5 known-future covariates).
Moirai (CPU, isolated venv) runs separately via subprocess — merged in afterwards.

    .venv\\Scripts\\python.exe scripts\\run_trained_leaderboard.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aq_fm_bench.data.processing import SPLITS  # noqa: E402
from aq_fm_bench.data.stations import TIER1  # noqa: E402
from aq_fm_bench.experiments.leaderboard import ChronosBoltBatch, fc_persistence, make_fc_seasonal  # noqa: E402
from aq_fm_bench.metrics.point import mae  # noqa: E402
from aq_fm_bench.models.linear import LinearForecaster  # noqa: E402
from aq_fm_bench.models.timesfm_model import TimesFMBatch  # noqa: E402
from aq_fm_bench.models.xgb import XGBDirectForecaster  # noqa: E402
from aq_fm_bench.paths import PROCESSED, RESULTS_LEADERBOARD  # noqa: E402

COL = {"NO2": "no2", "PM2.5": "pm25"}
WEATHER_COLS = ["u10", "v10", "t2m_c", "sp_hpa", "blh", "tp_mm"]   # the 6 ERA5 covariates (EXOG_SCHEMA)
CONTEXT_LEN, STRIDE, MAXH = 512, 168, 168
HORIZONS = (24, 72, 168)
pd.set_option("display.width", 175)


def test_positions(idx, ff, start, end):
    ts0, ts1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    cand = np.flatnonzero((idx >= ts0) & (idx < ts1))[::STRIDE]
    return np.array([p for p in cand
                     if p - CONTEXT_LEN + 1 >= 0 and p + MAXH < len(idx)
                     and np.isfinite(ff[p - CONTEXT_LEN + 1 : p + 1]).all()])


def train_positions(idx, stride=6):
    ts0 = pd.Timestamp(SPLITS["train"][0], tz="UTC")
    ts1 = pd.Timestamp(SPLITS["train"][1], tz="UTC")
    cand = np.flatnonzero((idx >= ts0) & (idx < ts1))[::stride]
    cap = int(np.flatnonzero(idx >= ts1)[0]) - MAXH
    return np.array([p for p in cand if p >= 168 and p < cap])


def main() -> int:
    chronos = ChronosBoltBatch()
    timesfm = TimesFMBatch()
    seas24, seas168 = make_fc_seasonal(24), make_fc_seasonal(168)
    rows = []

    for st in TIER1:
        df = pd.read_parquet(PROCESSED / f"{st.code}_hourly.parquet")
        idx = pd.DatetimeIndex(df["timestamp_utc"])
        wf = {w: pd.Series(df[w].to_numpy("float64")).ffill().bfill().to_numpy() for w in WEATHER_COLS}
        for pol in st.pollutants:
            col = COL[pol]
            vals = df[col].to_numpy("float64")
            ff = pd.Series(vals).ffill().to_numpy()
            tpos = test_positions(idx, ff, *SPLITS["test"])
            if len(tpos) == 0:
                continue
            ctxs = np.stack([ff[p - CONTEXT_LEN + 1 : p + 1] for p in tpos])
            actual_full = np.stack([vals[p + 1 : p + 1 + MAXH] for p in tpos])
            q90 = np.nanpercentile(actual_full, 90)
            cov_full = {w: np.stack([wf[w][p - CONTEXT_LEN + 1 : p + 1 + MAXH] for p in tpos])
                        for w in WEATHER_COLS}

            tr = train_positions(idx)
            xgbf = XGBDirectForecaster(device="cuda", oracle_weather=True).fit(df, col, tr, max_lead=MAXH)
            xgbf_no = XGBDirectForecaster(device="cuda", oracle_weather=False).fit(df, col, tr, max_lead=MAXH)
            linf = LinearForecaster(oracle_weather=True).fit(df, col, tr, max_lead=MAXH)
            linf_no = LinearForecaster(oracle_weather=False).fit(df, col, tr, max_lead=MAXH)

            h24 = {}
            for H in HORIZONS:
                actual = actual_full[:, :H]
                a = actual.ravel()
                top = a >= q90
                covH = {w: cov_full[w][:, :CONTEXT_LEN + H] for w in WEATHER_COLS}
                model_calls = {
                    "persistence": lambda H=H: fc_persistence(ctxs, H),
                    "seasonal_naive_24h": lambda H=H: seas24(ctxs, H),
                    "seasonal_naive_168h": lambda H=H: seas168(ctxs, H),
                    "linear_no_oracle": lambda H=H: linf_no.predict_traj(df, col, tpos, H),
                    "linear_oracle": lambda H=H: linf.predict_traj(df, col, tpos, H),
                    "chronos_bolt_base": lambda H=H: chronos(ctxs, H),
                    "timesfm_2_wx": lambda H=H, covH=covH: timesfm.predict(ctxs, H, covariates=covH),
                    "xgboost_no_oracle": lambda H=H: xgbf_no.predict_traj(df, col, tpos, H),
                    "xgboost": lambda H=H: xgbf.predict_traj(df, col, tpos, H),
                }
                for mname, fn in model_calls.items():
                    t0 = time.time()
                    P = fn()
                    secs = time.time() - t0
                    p = P.ravel()
                    o, td = round(mae(a, p), 3), round(mae(a[top], p[top]), 3)
                    rows.append({
                        "city": st.city, "station": st.code, "env_type": st.env_type,
                        "pollutant": pol, "horizon": H, "model": mname,
                        "mae": o, "mae_topdecile": td, "q90": round(float(q90), 1),
                        "n": int(np.isfinite(a).sum()), "n_origins": len(tpos),
                        "ms_per_forecast": round(secs / len(tpos) * 1000, 1),
                    })
                    if H == 24:
                        h24[mname] = o
            print(f"  {st.code:5} {pol:5} h24 MAE: chr={h24['chronos_bolt_base']:.1f} "
                  f"tfm={h24['timesfm_2_wx']:.1f} lin={h24['linear_oracle']:.1f} "
                  f"xgb={h24['xgboost']:.1f} | noOrc lin={h24['linear_no_oracle']:.1f} xgb={h24['xgboost_no_oracle']:.1f}")

    out = pd.DataFrame(rows)
    RESULTS_LEADERBOARD.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_LEADERBOARD / "uk_fm_weather_leaderboard.csv", index=False)
    out.to_parquet(RESULTS_LEADERBOARD / "uk_fm_weather_leaderboard.parquet", index=False)
    print(f"\nSaved {len(out)} rows -> results/leaderboard/uk_fm_weather_leaderboard.csv")

    order = ["persistence", "seasonal_naive_24h", "seasonal_naive_168h",
             "linear_no_oracle", "linear_oracle", "chronos_bolt_base", "timesfm_2_wx",
             "xgboost_no_oracle", "xgboost"]
    for pol in ("PM2.5", "NO2"):
        sub = out[out.pollutant == pol]
        if sub.empty:
            continue
        print(f"\n================ {pol}: MEDIAN MAE across UK stations ================")
        for metric in ("mae", "mae_topdecile"):
            piv = sub.pivot_table(index="model", columns="horizon", values=metric, aggfunc="median").reindex(order)
            lbl = "OVERALL" if metric == "mae" else "TOP-DECILE (episodes)"
            print(f"\n--- {pol} {lbl} ---")
            print(piv.round(2).to_string())
            print("best:", {h: piv[h].idxmin() for h in piv.columns})
    print("\n--- inference time (median ms/forecast) ---")
    print(out.pivot_table(index="model", values="ms_per_forecast", aggfunc="median").reindex(order).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

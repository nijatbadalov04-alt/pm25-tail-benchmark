"""
Moirai-1.1-R-base leaderboard pass — runs in .venv-fm (CPU). Produces zero-shot AND
weather-covariate forecasts on the SAME test origins as the main leaderboard, with overall +
top-decile MAE, both pollutants, all 3 horizons, and wall-clock ms/forecast.

Moirai uses native attention over known-future covariates (feat_dynamic_real spanning
context+horizon) — vs TimesFM's XReg ridge add-on — so this tests whether "FMs-with-weather"
is architecture-sensitive. Long-running (CPU); intended as a background job.

Output: results/leaderboard/moirai_results.csv  (merged with the GPU leaderboard afterwards).
Run with the .venv-fm interpreter:
    .venv-fm\\Scripts\\python.exe scripts\\run_moirai.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aq_fm_bench.data.processing import SPLITS  # noqa: E402
from aq_fm_bench.data.stations import TIER1  # noqa: E402
from aq_fm_bench.metrics.point import mae  # noqa: E402
from aq_fm_bench.paths import PROCESSED, RESULTS_LEADERBOARD  # noqa: E402

COL = {"NO2": "no2", "PM2.5": "pm25"}
WEATHER_COLS = ["u10", "v10", "t2m_c", "sp_hpa", "blh", "tp_mm"]
CONTEXT_LEN, STRIDE, MAXH = 512, 168, 168
HORIZONS = (24, 72, 168)


def test_positions(idx, ff, start, end):
    ts0, ts1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    cand = np.flatnonzero((idx >= ts0) & (idx < ts1))[::STRIDE]
    return np.array([p for p in cand
                     if p - CONTEXT_LEN + 1 >= 0 and p + MAXH < len(idx)
                     and np.isfinite(ff[p - CONTEXT_LEN + 1 : p + 1]).all()])


def main() -> int:
    print("loading Salesforce/moirai-1.1-R-base ...")
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base")
    rows = []

    # cache per-station arrays
    stations = []
    for st in TIER1:
        df = pd.read_parquet(PROCESSED / f"{st.code}_hourly.parquet")
        idx = pd.DatetimeIndex(df["timestamp_utc"])
        wf = {w: pd.Series(df[w].to_numpy("float64")).ffill().bfill().to_numpy() for w in WEATHER_COLS}
        stations.append((st, df, idx, wf))

    for variant in ("wx", "zeroshot"):           # with-weather first (the key comparison)
        feat_dim = len(WEATHER_COLS) if variant == "wx" else 0
        for H in HORIZONS:
            # GPU (torch 2.11+cu128 forced into .venv-fm) -> ~15x faster than CPU (87 vs 1300 ms/fc).
            # num_samples=20: point forecast = median; sufficient for MAE + residual-based MM-CP.
            model = MoiraiForecast(module=module, prediction_length=H, context_length=CONTEXT_LEN,
                                   patch_size="auto", num_samples=20, target_dim=1,
                                   feat_dynamic_real_dim=feat_dim, past_feat_dynamic_real_dim=0).to("cuda")
            predictor = model.create_predictor(batch_size=64, device="cuda")
            mname = f"moirai_1_1_{variant}"
            for st, df, idx, wf in stations:
                for pol in st.pollutants:
                    col = COL[pol]
                    vals = df[col].to_numpy("float64")
                    ff = pd.Series(vals).ffill().to_numpy()
                    tpos = test_positions(idx, ff, *SPLITS["test"])
                    if len(tpos) == 0:
                        continue
                    actual = np.stack([vals[p + 1 : p + 1 + H] for p in tpos])
                    q90 = np.nanpercentile(np.stack([vals[p + 1: p + 1 + MAXH] for p in tpos]), 90)
                    entries = []
                    for p in tpos:
                        e = {"target": ff[p - CONTEXT_LEN + 1 : p + 1].astype("float32"),
                             "start": pd.Period(pd.Timestamp(idx[p - CONTEXT_LEN + 1]).tz_convert(None), freq="h")}
                        if feat_dim:
                            e["feat_dynamic_real"] = np.stack(
                                [wf[w][p - CONTEXT_LEN + 1 : p + 1 + H] for w in WEATHER_COLS]).astype("float32")
                        entries.append(e)
                    t0 = time.time()
                    fcs = list(predictor.predict(ListDataset(entries, freq="h")))
                    secs = time.time() - t0
                    pred = np.stack([np.asarray(f.quantile(0.5)) for f in fcs])
                    a, p_ = actual.ravel(), pred.ravel()
                    top = a >= q90
                    rows.append({
                        "city": st.city, "station": st.code, "env_type": st.env_type,
                        "pollutant": pol, "horizon": H, "model": mname,
                        "mae": round(mae(a, p_), 3), "mae_topdecile": round(mae(a[top], p_[top]), 3),
                        "q90": round(float(q90), 1), "n": int(np.isfinite(a).sum()),
                        "n_origins": len(tpos), "ms_per_forecast": round(secs / len(tpos) * 1000, 1),
                    })
                    print(f"  {mname} {st.code:5} {pol:5} h{H:3}: MAE {rows[-1]['mae']:.2f} "
                          f"top {rows[-1]['mae_topdecile']:.2f} ({rows[-1]['ms_per_forecast']:.0f} ms/fc)",
                          flush=True)
                    pd.DataFrame(rows).to_csv(RESULTS_LEADERBOARD / "moirai_results.csv", index=False)

    print(f"\nDONE. {len(rows)} rows -> results/leaderboard/moirai_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

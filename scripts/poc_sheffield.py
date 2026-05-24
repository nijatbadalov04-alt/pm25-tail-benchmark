"""
Proof-of-concept: the motivating figure (a zero-shot FM interval under-covering an NO2 episode).

End-to-end on REAL Sheffield Devonshire Green NO2:
  - build hourly series + locked splits
  - rolling-origin (weekly stride) 24h forecasts: persistence, seasonal-naive(24h/168h),
    Chronos-Bolt-Base zero-shot
  - MAE/RMSE/sMAPE table on the test window
  - F-PoC figure (one forecast trajectory + Chronos 80% band) and metrics JSON

Run:
    .venv\\Scripts\\python.exe scripts\\poc_sheffield.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aq_fm_bench.data.processing import to_hourly_series, split_completeness, SPLITS  # noqa: E402
from aq_fm_bench.experiments.rolling import rolling_origin_eval  # noqa: E402
from aq_fm_bench.metrics.point import all_point_metrics  # noqa: E402
from aq_fm_bench.models.baselines import persistence, make_seasonal_naive  # noqa: E402
from aq_fm_bench.paths import RAW_AURN, FIGURES_EDA, FIGURES_PAPER, RESULTS_LEADERBOARD  # noqa: E402

POLLUTANT = "NO2"
HORIZON = 24
CONTEXT_LEN = 512
STRIDE = 168
TEST_START, TEST_END = SPLITS["test"]


def load_series() -> pd.Series:
    df = pd.read_parquet(RAW_AURN / "SHDG_2019_2024.parquet")
    s = to_hourly_series(df, POLLUTANT)
    print(f"Sheffield Devonshire Green {POLLUTANT}: {len(s):,} hourly slots "
          f"({s.notna().mean():.1%} non-null overall)")
    for name, c in split_completeness(s).items():
        print(f"  split {name:5}: completeness {c:6.1%}")
    return s


def build_chronos():
    """Load Chronos-Bolt-Base once; return (point_forecaster, quantile_fn) or (None, None)."""
    try:
        import torch
        from chronos import BaseChronosPipeline
    except Exception as e:  # noqa: BLE001
        print(f"[chronos] unavailable: {e!r}")
        return None, None

    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map="cuda", dtype=torch.bfloat16,
    )
    print(f"[chronos] loaded amazon/chronos-bolt-base on CUDA in {time.time() - t0:.1f}s")

    def quantile_fn(history, horizon, levels=(0.1, 0.5, 0.9)):
        ctx = torch.tensor(np.asarray(history, dtype="float32"))
        q, _mean = pipe.predict_quantiles(
            ctx, prediction_length=horizon, quantile_levels=list(levels),
        )
        return q[0].float().cpu().numpy()  # [horizon, n_levels]

    def point(history, horizon):
        return quantile_fn(history, horizon)[:, 1]  # median

    return point, quantile_fn


def find_plot_origin(series: pd.Series) -> int:
    """First test-window origin with full context + fully observed next HORIZON hours."""
    vals = series.to_numpy("float64")
    ff = series.ffill().to_numpy("float64")
    idx = series.index
    target = pd.Timestamp("2024-06-01", tz="UTC")
    cand = np.flatnonzero(idx >= target)
    for p in cand:
        if p - CONTEXT_LEN + 1 < 0 or p + 1 + HORIZON > len(idx):
            continue
        if np.isfinite(ff[p - CONTEXT_LEN + 1 : p + 1]).all() and \
           np.isfinite(vals[p + 1 : p + 1 + HORIZON]).all():
            return int(p)
    return int(cand[0])


def main() -> int:
    series = load_series()

    forecasters = {
        "persistence": persistence,
        "seasonal_naive_24h": make_seasonal_naive(24),
        "seasonal_naive_168h": make_seasonal_naive(168),
    }
    chronos_point, chronos_q = build_chronos()
    if chronos_point is not None:
        forecasters["chronos_bolt_base"] = chronos_point

    print(f"\n=== Rolling-origin eval: {POLLUTANT} H={HORIZON}h, stride={STRIDE}h, "
          f"test {TEST_START}..{TEST_END} ===")
    results = {}
    for name, fc in forecasters.items():
        t0 = time.time()
        preds, acts, info = rolling_origin_eval(
            series, fc, HORIZON, test_start=TEST_START, test_end=TEST_END,
            context_len=CONTEXT_LEN, stride_hours=STRIDE,
        )
        m = all_point_metrics(acts, preds)
        m.update(info)
        m["seconds"] = round(time.time() - t0, 1)
        results[name] = m
        print(f"  {name:22} MAE={m['mae']:6.2f}  RMSE={m['rmse']:6.2f}  "
              f"sMAPE={m['smape']:5.1f}%  n={m['n']:5d}  origins={info['n_origins']}  "
              f"({m['seconds']}s)")

    # ---- F-PoC figure: one 24h forecast trajectory + Chronos 80% band --------
    p = find_plot_origin(series)
    idx = series.index
    vals = series.to_numpy("float64")
    ff = series.ffill().to_numpy("float64")
    ctx = ff[p - CONTEXT_LEN + 1 : p + 1]
    hist_show = 48
    x_hist = np.arange(-hist_show, 0)
    x_fut = np.arange(0, HORIZON)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x_hist, vals[p - hist_show + 1 : p + 1], color="black", lw=1.5, label="observed (context)")
    ax.plot(x_fut, vals[p + 1 : p + 1 + HORIZON], color="black", lw=2.2, marker="o", ms=3,
            label="observed (target)")
    ax.axvline(0, color="grey", ls=":", lw=1)
    ax.plot(x_fut, persistence(ctx, HORIZON), ls="--", label=f"persistence (MAE {results['persistence']['mae']:.1f})")
    ax.plot(x_fut, make_seasonal_naive(24)(ctx, HORIZON), ls="--",
            label=f"seasonal-naive 24h (MAE {results['seasonal_naive_24h']['mae']:.1f})")
    if chronos_q is not None:
        q = chronos_q(ctx, HORIZON)  # [H, 3]
        ax.plot(x_fut, q[:, 1], color="tab:red", lw=2,
                label=f"Chronos-Bolt median (MAE {results['chronos_bolt_base']['mae']:.1f})")
        ax.fill_between(x_fut, q[:, 0], q[:, 2], color="tab:red", alpha=0.18, label="Chronos 80% interval")
    ax.set_xlabel("hours from forecast origin")
    ax.set_ylabel("NO$_2$ (µg/m³)")
    ax.set_title(f"AQ-FM-Bench PoC — Sheffield Devonshire Green NO$_2$, 24h forecast\n"
                 f"origin {idx[p]:%Y-%m-%d %H:%M} UTC")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    FIGURES_EDA.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_EDA / "poc_sheffield_no2_24h.png"
    fig.savefig(fig_path, dpi=150)
    FIGURES_PAPER.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_PAPER / "F_intro.pdf", dpi=300, bbox_inches="tight")  # paper Figure 1
    print(f"\nSaved figure: {fig_path} + {FIGURES_PAPER / 'F_intro.pdf'}")

    RESULTS_LEADERBOARD.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_LEADERBOARD / "poc_sheffield_no2_24h.json"
    out_json.write_text(json.dumps(
        {k: {kk: (str(vv) if isinstance(vv, pd.Timestamp) else vv) for kk, vv in v.items()}
         for k, v in results.items()}, indent=2, default=str))
    print(f"Saved metrics: {out_json}")

    print("\n=== W2 GATE: numbers + figure produced ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

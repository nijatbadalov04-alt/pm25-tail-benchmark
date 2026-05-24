"""
Exploratory / diagnostic figures for AQ-FM-Bench (300 DPI PDF+PNG -> figures/eda/).
Each figure motivates a paper finding and doubles as a data-quality check:

  F_eda_completeness — monthly data completeness per station (the data-quality milestone).
  F_eda_diurnal      — mean diurnal cycle by environment type (traffic NO2 rush peaks; motivates calendar feats).
  F_eda_seasonal     — monthly climatology + episode-hour rate (winter episodes; motivates F1 tail analysis).
  F_eda_ventilation  — pollutant vs boundary-layer height & wind speed (the stagnation story behind F2/F4).
  F_eda_distribution — pollutant distributions with the q90 episode threshold (defines the top-decile tail).

Reads data/processed/{code}_hourly.parquet. Run:
    .venv\Scripts\python.exe scripts\make_eda_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aq_fm_bench.data.stations import TIER1
from aq_fm_bench.paths import FIGURES_EDA, PROCESSED
from aq_fm_bench.viz.style import (ENVTYPE_COLOR, ENVTYPE_LABEL, OI, POLLUTANT_COLOR,
                                   POLLUTANT_LABEL, save, set_style)

import matplotlib.pyplot as plt

COL = {"NO2": "no2", "PM2.5": "pm25"}
POLS = ("NO2", "PM2.5")


def load_all() -> dict[str, pd.DataFrame]:
    out = {}
    for st in TIER1:
        p = PROCESSED / f"{st.code}_hourly.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["timestamp_utc"] = pd.DatetimeIndex(df["timestamp_utc"])
            out[st.code] = df
    print(f"loaded {len(out)} stations: {', '.join(out)}")
    return out


def fig_completeness(data):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, pol in zip(axes, POLS):
        c = COL[pol]
        rows, codes = [], []
        for st in TIER1:
            if st.code not in data or pol not in st.pollutants:
                continue
            df = data[st.code]
            mser = df.set_index("timestamp_utc")[c]
            monthly = mser.notna().groupby(mser.index.to_period("M")).mean() * 100
            rows.append(monthly); codes.append(f"{st.code} ({st.city[:3]},{ENVTYPE_LABEL[st.env_type][:4]})")
        M = pd.concat(rows, axis=1).T
        M.columns = [str(p) for p in M.columns]
        im = ax.imshow(M.to_numpy(dtype="float64"), aspect="auto", cmap="YlGnBu", vmin=0, vmax=100,
                       interpolation="nearest")
        ax.set_yticks(range(len(codes))); ax.set_yticklabels(codes, fontsize=7)
        step = max(1, M.shape[1] // 12)
        ax.set_xticks(range(0, M.shape[1], step)); ax.set_xticklabels(M.columns[::step], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{POLLUTANT_LABEL[pol]} — monthly data completeness (%)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="% valid")
    fig.suptitle("Data completeness by station-month (≥75% completeness gate)", y=1.0)
    fig.tight_layout()
    save(fig, FIGURES_EDA / "F_eda_completeness")


def fig_diurnal(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, pol in zip(axes, POLS):
        c = COL[pol]
        for env in ("urban_traffic", "urban_background", "suburban_industrial"):
            series = []
            for st in TIER1:
                if st.code not in data or st.env_type != env or pol not in st.pollutants:
                    continue
                df = data[st.code]
                series.append(df.assign(h=df["timestamp_utc"].dt.hour).groupby("h")[c].mean())
            if not series:
                continue
            m = pd.concat(series, axis=1).mean(axis=1)
            ax.plot(m.index, m.values, color=ENVTYPE_COLOR[env], lw=2, label=ENVTYPE_LABEL[env])
        ax.set_xlabel("hour of day (UTC)"); ax.set_ylabel(f"mean {POLLUTANT_LABEL[pol]} (µg/m³)")
        ax.set_title(f"{POLLUTANT_LABEL[pol]} diurnal cycle"); ax.set_xticks(range(0, 24, 4))
        ax.legend()
    fig.suptitle("Diurnal cycle by station environment (pooled within type)")
    fig.tight_layout()
    save(fig, FIGURES_EDA / "F_eda_diurnal")


def fig_seasonal(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, pol in zip(axes, POLS):
        c = COL[pol]
        allser, epi = [], []
        for st in TIER1:
            if st.code not in data or pol not in st.pollutants:
                continue
            df = data[st.code]; v = df[c]
            q90 = np.nanpercentile(v, 90)
            mon = df["timestamp_utc"].dt.month
            allser.append(df.assign(m=mon).groupby("m")[c].mean())
            epi.append((df.assign(m=mon, ep=(v >= q90)).groupby("m")["ep"].mean()) * 100)
        clim = pd.concat(allser, axis=1).mean(axis=1)
        ax.plot(clim.index, clim.values, color=POLLUTANT_COLOR[pol], lw=2, marker="o", label="mean conc.")
        ax.set_xlabel("month"); ax.set_ylabel(f"mean {POLLUTANT_LABEL[pol]} (µg/m³)", color=POLLUTANT_COLOR[pol])
        ax.set_xticks(range(1, 13))
        ax2 = ax.twinx(); ax2.grid(False)
        epirate = pd.concat(epi, axis=1).mean(axis=1)
        ax2.bar(epirate.index, epirate.values, color=OI["grey"], alpha=0.35, label="episode-hour rate")
        ax2.set_ylabel("episode-hour rate (% > q90)", color=OI["grey"])
        ax.set_title(f"{POLLUTANT_LABEL[pol]} seasonality")
    fig.suptitle("Monthly climatology and episode (>q90) seasonality")
    fig.tight_layout()
    save(fig, FIGURES_EDA / "F_eda_seasonal")


def fig_ventilation(data):
    """The stagnation story: pollution rises as boundary layer collapses and wind drops."""
    drivers = [("blh", "boundary-layer height (m)", (0, 1500)), ("ws10", "10 m wind speed (m/s)", (0, 12))]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, pol in enumerate(POLS):
        c = COL[pol]
        for j, (dcol, dlabel, xlim) in enumerate(drivers):
            ax = axes[i, j]
            xs, ys = [], []
            for st in TIER1:
                if st.code not in data or pol not in st.pollutants or dcol not in data[st.code]:
                    continue
                df = data[st.code]
                m = df[c].notna() & df[dcol].notna()
                xs.append(df[dcol][m].to_numpy()); ys.append(df[c][m].to_numpy())
            x = np.concatenate(xs); y = np.concatenate(ys)
            # binned median + IQR over deciles of the driver
            edges = np.nanpercentile(x, np.linspace(0, 100, 11))
            edges = np.unique(edges)
            idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
            cx = 0.5 * (edges[:-1] + edges[1:])
            med = np.array([np.median(y[idx == b]) if (idx == b).any() else np.nan for b in range(len(cx))])
            lo = np.array([np.percentile(y[idx == b], 25) if (idx == b).any() else np.nan for b in range(len(cx))])
            hi = np.array([np.percentile(y[idx == b], 75) if (idx == b).any() else np.nan for b in range(len(cx))])
            ax.fill_between(cx, lo, hi, color=POLLUTANT_COLOR[pol], alpha=0.2)
            ax.plot(cx, med, color=POLLUTANT_COLOR[pol], lw=2, marker="o", ms=4)
            ax.set_xlim(*xlim)
            ax.set_xlabel(dlabel); ax.set_ylabel(f"{POLLUTANT_LABEL[pol]} (µg/m³)")
            r = np.corrcoef(x, y)[0, 1]
            ax.set_title(f"{POLLUTANT_LABEL[pol]} vs {dlabel.split('(')[0].strip()}  (r={r:+.2f})", fontsize=9)
    fig.suptitle("Ventilation controls pollution: concentration vs boundary-layer height & wind (median, IQR band)")
    fig.tight_layout()
    save(fig, FIGURES_EDA / "F_eda_ventilation")


def fig_distribution(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, pol in zip(axes, POLS):
        c = COL[pol]
        v = np.concatenate([data[st.code][c].dropna().to_numpy()
                            for st in TIER1 if st.code in data and pol in st.pollutants])
        v = v[v > 0]
        ax.hist(v, bins=np.logspace(np.log10(max(v.min(), 0.1)), np.log10(v.max()), 60),
                color=POLLUTANT_COLOR[pol], alpha=0.8)
        q90 = np.percentile(v, 90)
        ax.axvline(q90, ls="--", color=OI["black"], lw=1.4, label=f"q90 = {q90:.1f} µg/m³ (episode threshold)")
        ax.set_xscale("log")
        ax.set_xlabel(f"{POLLUTANT_LABEL[pol]} (µg/m³, log)"); ax.set_ylabel("hours")
        ax.set_title(f"{POLLUTANT_LABEL[pol]} distribution (n={len(v):,})"); ax.legend()
    fig.suptitle("Pollutant distributions and the top-decile episode threshold")
    fig.tight_layout()
    save(fig, FIGURES_EDA / "F_eda_distribution")


def main() -> int:
    set_style()
    FIGURES_EDA.mkdir(parents=True, exist_ok=True)
    data = load_all()
    if not data:
        print("no processed parquets found — nothing to plot"); return 1
    for fn in (fig_completeness, fig_diurnal, fig_seasonal, fig_ventilation, fig_distribution):
        try:
            fn(data)
        except Exception as e:  # one bad figure shouldn't kill the rest
            print(f"  [WARN] {fn.__name__} failed: {type(e).__name__}: {e}")
    print(f"\nEDA figures -> {FIGURES_EDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Paper figures (300 DPI PDF, Okabe-Ito colourblind-safe) -> figures/paper/.
F_leaderboard: h24 MAE with 95% CI bars (NO2 + PM2.5).  F_coverage: split-CP vs 3D vs 4D MM-CP
top-decile PM2.5 coverage.  (F_intro is produced by poc_sheffield.py.)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aq_fm_bench.paths import FIGURES_PAPER, RESULTS_LEADERBOARD, RESULTS

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "grey": "#999999", "vermillion": "#D55E00"}
LABEL = {"persistence": "persistence", "seasonal_naive_24h": "seasonal-naive 24h",
         "seasonal_naive_168h": "seasonal-naive 168h", "linear_no_oracle": "linear (no wx)",
         "linear_oracle": "linear (+wx)", "chronos_bolt_base": "Chronos-Bolt",
         "timesfm_2_wx": "TimesFM-2 (+wx)", "moirai_1_1_zeroshot": "Moirai (zero-shot)",
         "moirai_1_1_wx": "Moirai (+wx)", "xgboost_no_oracle": "XGBoost (no wx)", "xgboost": "XGBoost (+wx)"}
FIGURES_PAPER.mkdir(parents=True, exist_ok=True)


def color(m):
    if m.startswith(("chronos", "timesfm", "moirai")):
        return OI["blue"]
    if m.startswith(("xgboost", "linear")):
        return OI["orange"]
    return OI["grey"]


def fig_leaderboard():
    lb = pd.read_csv(RESULTS_LEADERBOARD / "significance_h24.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, pol in zip(axes, ("NO2", "PM2.5")):
        d = lb[lb.pollutant == pol].sort_values("mae", ascending=True)
        y = range(len(d))
        err = [d["mae"] - d["mae_lo"], d["mae_hi"] - d["mae"]]
        ax.barh(y, d["mae"], xerr=err, color=[color(m) for m in d["model"]], alpha=0.85,
                error_kw=dict(ecolor="#333333", lw=1, capsize=2))
        ax.set_yticks(list(y)); ax.set_yticklabels([LABEL[m] for m in d["model"]], fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel("MAE (µg/m³), 24h  [95% CI]")
        ax.set_title(f"{pol} — 24h leaderboard")
        ax.grid(axis="x", ls=":", alpha=0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OI[c]) for c in ("blue", "orange", "grey")]
    fig.legend(handles, ["foundation models", "classical (XGBoost/linear)", "naive"],
               loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIGURES_PAPER / "F_leaderboard.pdf", dpi=300, bbox_inches="tight")
    print("saved F_leaderboard.pdf")


def fig_coverage():
    cov = pd.read_csv(RESULTS / "coverage" / "mmcp4d_pm25.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    import numpy as np
    x = np.arange(len(cov)); w = 0.26
    ax.bar(x - w, cov["split"], w, label="split-CP", color=OI["grey"])
    ax.bar(x, cov["mmcp3d"], w, label="MM-CP 3D", color=OI["orange"])
    ax.bar(x + w, cov["mmcp4d"], w, label="MM-CP 4D (episode)", color=OI["green"])
    ax.axhline(0.90, ls="--", color=OI["vermillion"], lw=1.2, label="nominal 90%")
    ax.set_xticks(x); ax.set_xticklabels([LABEL.get(m, m) for m in cov["model"]], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("top-decile PM₂.₅ coverage @24h"); ax.set_ylim(0, 1.0)
    ax.set_title("Conformal coverage on PM₂.₅ episodes: split-CP vs MM-CP (3D, 4D)")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES_PAPER / "F_coverage.pdf", dpi=300, bbox_inches="tight")
    print("saved F_coverage.pdf")


if __name__ == "__main__":
    fig_leaderboard()
    fig_coverage()

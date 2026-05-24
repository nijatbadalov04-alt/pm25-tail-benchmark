"""
UK Tier-1 leaderboard: persistence, seasonal-naive (24h/168h), Chronos-Bolt zero-shot
across all 8 AURN stations x {NO2, PM2.5} x {24,72,168}h. Saves CSV + prints summary.

    .venv\\Scripts\\python.exe scripts\\run_uk_leaderboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aq_fm_bench.data.stations import TIER1  # noqa: E402
from aq_fm_bench.experiments.leaderboard import run_leaderboard  # noqa: E402
from aq_fm_bench.paths import RESULTS_LEADERBOARD  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)


def main() -> int:
    print(f"=== UK Tier-1 leaderboard: {len(TIER1)} stations ===")
    df = run_leaderboard(TIER1, pollutants=("NO2", "PM2.5"), horizons=(24, 72, 168))

    RESULTS_LEADERBOARD.mkdir(parents=True, exist_ok=True)
    out = RESULTS_LEADERBOARD / "uk_tier1_leaderboard.csv"
    df.to_parquet(out.with_suffix(".parquet"), index=False)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows -> {out}")

    # summary: MEDIAN MAE across stations (median, not mean, to limit high-pollution-site dominance), per pollutant x horizon x model
    print("\n================ MEDIAN MAE across UK stations (µg/m³) ================")
    for pol in ("NO2", "PM2.5"):
        sub = df[df.pollutant == pol]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="model", columns="horizon", values="mae", aggfunc="median")
        piv = piv.reindex(["persistence", "seasonal_naive_24h", "seasonal_naive_168h", "chronos_bolt_base"])
        print(f"\n--- {pol} ---")
        print(piv.round(2).to_string())
        # who wins each horizon
        wins = {h: piv[h].idxmin() for h in piv.columns}
        print("best per horizon:", wins)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

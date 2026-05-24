"""Merge GPU leaderboard + Moirai results into the full UK leaderboard; print medians incl Moirai."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aq_fm_bench.paths import RESULTS_LEADERBOARD

pd.set_option("display.width", 175)
a = pd.read_parquet(RESULTS_LEADERBOARD / "uk_fm_weather_leaderboard.parquet")
b = pd.read_csv(RESULTS_LEADERBOARD / "moirai_results.csv")
full = pd.concat([a, b], ignore_index=True)
full.to_csv(RESULTS_LEADERBOARD / "uk_full_leaderboard.csv", index=False)
print(f"merged {len(a)} + {len(b)} = {len(full)} rows -> uk_full_leaderboard.csv")

order = ["persistence", "seasonal_naive_24h", "seasonal_naive_168h",
         "linear_no_oracle", "linear_oracle", "chronos_bolt_base",
         "moirai_1_1_zeroshot", "moirai_1_1_wx", "timesfm_2_wx",
         "xgboost_no_oracle", "xgboost"]
for pol in ("PM2.5", "NO2"):
    sub = full[full.pollutant == pol]
    for metric, lbl in (("mae", "OVERALL"), ("mae_topdecile", "TOP-DECILE")):
        piv = sub.pivot_table(index="model", columns="horizon", values=metric, aggfunc="median").reindex(order)
        print(f"\n=== {pol} {lbl} median MAE ===")
        print(piv.round(2).to_string())

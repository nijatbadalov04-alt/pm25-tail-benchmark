"""
Dump Moirai per-window TEST predictions for both pollutants x {24,72,168} x {wx, zeroshot},
so every leaderboard cell (incl. Moirai) gets a block-bootstrap CI and F6 is significance-testable.
Runs in .venv-fm (GPU). Output: results/preds/moirai_test_preds.pkl
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aq_fm_bench.data.processing import SPLITS
from aq_fm_bench.data.stations import TIER1
from aq_fm_bench.paths import PROCESSED, RESULTS

WEATHER = ["u10", "v10", "t2m_c", "sp_hpa", "blh", "tp_mm"]
COL = {"NO2": "no2", "PM2.5": "pm25"}
C, STRIDE, MAXH = 512, 168, 168
HORIZONS = (24,)  # headline horizon for all F-claims; h72/168 use aggregate moirai_results.csv


def test_positions(idx, ff):
    a, b = pd.Timestamp(SPLITS["test"][0], tz="UTC"), pd.Timestamp(SPLITS["test"][1], tz="UTC")
    c = np.flatnonzero((idx >= a) & (idx < b))[::STRIDE]
    return np.array([p for p in c if p - C + 1 >= 0 and p + MAXH < len(idx)
                     and np.isfinite(ff[p - C + 1: p + 1]).all()])


def main() -> int:
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base")
    out = {}
    for variant in ("wx", "zeroshot"):
        feat = len(WEATHER) if variant == "wx" else 0
        for H in HORIZONS:
            model = MoiraiForecast(module=module, prediction_length=H, context_length=C, patch_size="auto",
                                   num_samples=20, target_dim=1, feat_dynamic_real_dim=feat,
                                   past_feat_dynamic_real_dim=0).to("cuda")
            predictor = model.create_predictor(batch_size=64, device="cuda")
            for st in TIER1:
                df = pd.read_parquet(PROCESSED / f"{st.code}_hourly.parquet")
                idx = pd.DatetimeIndex(df["timestamp_utc"])
                wf = {w: pd.Series(df[w].to_numpy("float64")).ffill().bfill().to_numpy() for w in WEATHER}
                for pol in st.pollutants:
                    ff = pd.Series(df[COL[pol]].to_numpy("float64")).ffill().to_numpy()
                    tp = test_positions(idx, ff)
                    entries = []
                    for p in tp:
                        e = {"target": ff[p - C + 1: p + 1].astype("float32"),
                             "start": pd.Period(pd.Timestamp.utcfromtimestamp(0), freq="h")}
                        if feat:
                            e["feat_dynamic_real"] = np.stack(
                                [wf[w][p - C + 1: p + 1 + H] for w in WEATHER]).astype("float32")
                        entries.append(e)
                    fcs = list(predictor.predict(ListDataset(entries, freq="h")))
                    pred = np.stack([np.asarray(f.quantile(0.5)) for f in fcs])
                    out[(variant, pol, H, st.code)] = {"test_pos": tp, "test_pred": pred}
            print(f"  moirai_{variant} h{H}: done", flush=True)
    (RESULTS / "preds").mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "preds" / "moirai_test_preds.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"saved {len(out)} (variant,pol,H,station) blocks -> results/preds/moirai_test_preds.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

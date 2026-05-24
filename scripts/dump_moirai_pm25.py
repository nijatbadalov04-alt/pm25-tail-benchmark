"""
Dump Moirai per-window PM2.5 h24 predictions (cal + test origins, both variants) so the main-venv
analysis can include Moirai in the F1 tail-significance bootstrap and the MM-CP experiment.
Runs in .venv-fm (GPU). Output: results/preds/moirai_pm25_h24.pkl
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
C, STRIDE, H, MAXH = 512, 168, 24, 168


def positions(idx, ff, start, end):
    ts0, ts1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    cand = np.flatnonzero((idx >= ts0) & (idx < ts1))[::STRIDE]
    return np.array([p for p in cand if p - C + 1 >= 0 and p + MAXH < len(idx)
                     and np.isfinite(ff[p - C + 1: p + 1]).all()])


def predict(predictor, ff, wf, pos, feat):
    entries = []
    for p in pos:
        e = {"target": ff[p - C + 1: p + 1].astype("float32"),
             "start": pd.Period(pd.Timestamp.utcfromtimestamp(0), freq="h")}
        if feat:
            e["feat_dynamic_real"] = np.stack([wf[w][p - C + 1: p + 1 + H] for w in WEATHER]).astype("float32")
        entries.append(e)
    fcs = list(predictor.predict(ListDataset(entries, freq="h")))
    return np.stack([np.asarray(f.quantile(0.5)) for f in fcs])


def main() -> int:
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base")
    out = {}
    for variant in ("wx", "zeroshot"):
        feat = len(WEATHER) if variant == "wx" else 0
        model = MoiraiForecast(module=module, prediction_length=H, context_length=C, patch_size="auto",
                               num_samples=20, target_dim=1, feat_dynamic_real_dim=feat,
                               past_feat_dynamic_real_dim=0).to("cuda")
        predictor = model.create_predictor(batch_size=64, device="cuda")
        out[variant] = {}
        for st in TIER1:
            if "PM2.5" not in st.pollutants:
                continue
            df = pd.read_parquet(PROCESSED / f"{st.code}_hourly.parquet")
            idx = pd.DatetimeIndex(df["timestamp_utc"])
            ff = pd.Series(df["pm25"].to_numpy("float64")).ffill().to_numpy()
            wf = {w: pd.Series(df[w].to_numpy("float64")).ffill().bfill().to_numpy() for w in WEATHER}
            cal = positions(idx, ff, *SPLITS["cal"])
            test = positions(idx, ff, *SPLITS["test"])
            out[variant][st.code] = {
                "cal_pos": cal, "test_pos": test,
                "cal_pred": predict(predictor, ff, wf, cal, feat),
                "test_pred": predict(predictor, ff, wf, test, feat),
            }
            print(f"  {variant} {st.code}: cal {len(cal)} test {len(test)}", flush=True)
    (RESULTS / "preds").mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "preds" / "moirai_pm25_h24.pkl", "wb") as f:
        pickle.dump(out, f)
    print("saved results/preds/moirai_pm25_h24.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

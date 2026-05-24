"""
MM-CP — Meteorology-Mondrian Conformal Prediction (the AQ-FM-Bench method contribution).

Mondrian conformal predictor with a physically-defined taxonomy: cells = wind-speed tercile ×
BLH tercile × hour-of-day bucket (3×3×4 = 36). Per-cell conformal quantiles target the
heteroscedasticity that single-quantile split-CP mishandles (esp. low-BLH × low-wind stagnation).
Spec: docs/MM_CP_METHOD.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aq_fm_bench.conformal.split_cp import conformal_quantile

N_HOD = 4  # hour buckets: 0-5, 6-11, 12-17, 18-23


def _ws_blh_hod_bins(ws, blh, hour, ws_edges, blh_edges):
    ws_bin = np.digitize(np.asarray(ws, "float64"), ws_edges)   # 0,1,2
    blh_bin = np.digitize(np.asarray(blh, "float64"), blh_edges)
    hod_bin = np.clip(np.asarray(hour, "int64") // 6, 0, N_HOD - 1)
    return ws_bin, blh_bin, hod_bin


@dataclass
class MMCPModel:
    ws_edges: np.ndarray
    blh_edges: np.ndarray
    q_cell: dict          # full cell (ws,blh,hod) -> qhat
    q_wsblh: dict         # ws×blh marginal -> qhat  (fallback 1)
    q_global: float       # split-CP global -> qhat  (fallback 2)
    n_min: int

    def _qhat_for(self, ws_bin, blh_bin, hod_bin) -> np.ndarray:
        out = np.empty(len(ws_bin), dtype="float64")
        for i in range(len(ws_bin)):
            cell = (int(ws_bin[i]), int(blh_bin[i]), int(hod_bin[i]))
            if cell in self.q_cell:
                out[i] = self.q_cell[cell]
            elif (cell[0], cell[1]) in self.q_wsblh:
                out[i] = self.q_wsblh[(cell[0], cell[1])]
            else:
                out[i] = self.q_global
        return out

    def intervals(self, test_pred, ws, blh, hour):
        wb, bb, hb = _ws_blh_hod_bins(ws, blh, hour, self.ws_edges, self.blh_edges)
        q = self._qhat_for(wb, bb, hb)
        test_pred = np.asarray(test_pred, "float64")
        return test_pred - q, test_pred + q


# ---------------------------------------------------------------------------
# 4D extension: episode-conditioned MM-CP (iteration 2). See docs/MM_CP_METHOD.md.
# ---------------------------------------------------------------------------
def episode_cuts(cal_conc_means, cal_blh_lookback, cal_ws_lookback):
    """Calibration cuts: C3 = 2/3-quantile of 24h-mean conc; blh_lo, ws_lo = 1/3-quantiles."""
    return (float(np.nanquantile(cal_conc_means, 2 / 3)),
            float(np.nanquantile(cal_blh_lookback, 1 / 3)),
            float(np.nanquantile(cal_ws_lookback, 1 / 3)))


def episode_regime(conc_ff, blh_arr, ws_arr, origins, *, C3, blh_lo, ws_lo, lookback: int = 24):
    """E(t0)=1 if (24h-mean conc >= C3) OR (>=6 of the 24 lookback hours have blh<blh_lo & ws<ws_lo)."""
    E = np.zeros(len(origins), dtype="int64")
    for i, p in enumerate(origins):
        a, b = p - lookback + 1, p + 1
        cbar = np.nanmean(conc_ff[a:b])
        nstag = int(np.sum((blh_arr[a:b] < blh_lo) & (ws_arr[a:b] < ws_lo)))
        E[i] = 1 if (cbar >= C3 or nstag >= 6) else 0
    return E


def _hod2(hour):
    h = np.asarray(hour, "int64")
    return np.where((h >= 6) & (h < 18), 0, 1)  # day / night


@dataclass
class MMCPModel4D:
    ws_edges: np.ndarray
    blh_edges: np.ndarray
    q_cell: dict       # (ws,blh,hod2,reg) -> q
    q_3d: dict         # (ws,blh,hod2) -> q   (regime backoff)
    q_wsblh: dict
    q_global: float
    n_min: int
    cell_counts: dict

    def _qhat(self, wb, bb, hb, rb):
        out = np.empty(len(wb), "float64")
        for i in range(len(wb)):
            k4 = (int(wb[i]), int(bb[i]), int(hb[i]), int(rb[i]))
            if k4 in self.q_cell:
                out[i] = self.q_cell[k4]
            elif k4[:3] in self.q_3d:
                out[i] = self.q_3d[k4[:3]]
            elif (k4[0], k4[1]) in self.q_wsblh:
                out[i] = self.q_wsblh[(k4[0], k4[1])]
            else:
                out[i] = self.q_global
        return out

    def intervals(self, pred, ws, blh, hour, regime):
        wb = np.digitize(np.asarray(ws, "float64"), self.ws_edges)
        bb = np.digitize(np.asarray(blh, "float64"), self.blh_edges)
        q = self._qhat(wb, bb, _hod2(hour), np.asarray(regime, "int64"))
        pred = np.asarray(pred, "float64")
        return pred - q, pred + q


def fit_mm_cp_4d(cal_resid, ws, blh, hour, regime, alpha: float = 0.1, n_min: int = 50) -> MMCPModel4D:
    cal_resid = np.asarray(cal_resid, "float64")
    ws_edges = np.nanquantile(ws, [1 / 3, 2 / 3])
    blh_edges = np.nanquantile(blh, [1 / 3, 2 / 3])
    wb = np.digitize(np.asarray(ws, "float64"), ws_edges)
    bb = np.digitize(np.asarray(blh, "float64"), blh_edges)
    hb = _hod2(hour)
    rb = np.asarray(regime, "int64")
    finite = np.isfinite(cal_resid)
    q_cell, q_3d, q_wsblh, cell_counts = {}, {}, {}, {}
    for w in range(3):
        for b in range(3):
            mwb = finite & (wb == w) & (bb == b)
            if mwb.sum() >= n_min:
                q_wsblh[(w, b)] = conformal_quantile(cal_resid[mwb], alpha)
            for h in range(2):
                m3 = mwb & (hb == h)
                if m3.sum() >= n_min:
                    q_3d[(w, b, h)] = conformal_quantile(cal_resid[m3], alpha)
                for r in range(2):
                    m4 = m3 & (rb == r)
                    cell_counts[(w, b, h, r)] = int(m4.sum())
                    if m4.sum() >= n_min:
                        q_cell[(w, b, h, r)] = conformal_quantile(cal_resid[m4], alpha)
    q_global = conformal_quantile(cal_resid[finite], alpha)
    return MMCPModel4D(ws_edges, blh_edges, q_cell, q_3d, q_wsblh, q_global, n_min, cell_counts)


def fit_mm_cp(cal_resid, cal_ws, cal_blh, cal_hour, alpha: float = 0.1, n_min: int = 50) -> MMCPModel:
    cal_resid = np.asarray(cal_resid, "float64")
    ws_edges = np.nanquantile(cal_ws, [1 / 3, 2 / 3])
    blh_edges = np.nanquantile(cal_blh, [1 / 3, 2 / 3])
    wb, bb, hb = _ws_blh_hod_bins(cal_ws, cal_blh, cal_hour, ws_edges, blh_edges)
    finite = np.isfinite(cal_resid)

    q_cell, q_wsblh = {}, {}
    for w in range(3):
        for b in range(3):
            mwb = finite & (wb == w) & (bb == b)
            if mwb.sum() >= n_min:
                q_wsblh[(w, b)] = conformal_quantile(cal_resid[mwb], alpha)
            for h in range(N_HOD):
                m = mwb & (hb == h)
                if m.sum() >= n_min:
                    q_cell[(w, b, h)] = conformal_quantile(cal_resid[m], alpha)
    q_global = conformal_quantile(cal_resid[finite], alpha)
    return MMCPModel(ws_edges, blh_edges, q_cell, q_wsblh, q_global, n_min)

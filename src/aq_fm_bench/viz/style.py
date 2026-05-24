"""
Shared publication figure style (Okabe-Ito colourblind-safe palette, 300 DPI).
Keeps every figure in the project visually consistent with the paper figures.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe palette
OI = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#999999",
}
POLLUTANT_COLOR = {"NO2": OI["blue"], "PM2.5": OI["vermillion"]}
POLLUTANT_LABEL = {"NO2": "NO$_2$", "PM2.5": "PM$_{2.5}$"}
ENVTYPE_COLOR = {
    "urban_traffic": OI["vermillion"], "urban_background": OI["blue"],
    "suburban_industrial": OI["green"],
}
ENVTYPE_LABEL = {
    "urban_traffic": "urban traffic", "urban_background": "urban background",
    "suburban_industrial": "suburban industrial",
}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.4,
        "legend.frameon": False, "legend.fontsize": 8,
        "figure.facecolor": "white",
    })


def save(fig, path, also_png: bool = True) -> None:
    """Save a figure as PDF (vector, for the paper) and optionally PNG (for quick viewing/notes)."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    if also_png:
        fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  saved {path.stem}.pdf" + (" + .png" if also_png else ""))

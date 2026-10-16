"""Shared chart style and a save helper for the report's figures.

Every figure is both embedded in the rendered report (Quarto captures it when
the chunk ends) and written to outputs/figures/ as a PNG via ``save``.
"""
from pathlib import Path

import matplotlib.pyplot as plt

PLANT = "#2a78d6"    # plant protein (blue)
ANIMAL = "#eb6834"   # animal protein (orange)
SECOND = "#1baf7a"   # second measure, e.g. weight next to dollars (aqua)

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def save(fig: plt.Figure, name: str) -> Path:
    """Write ``fig`` to outputs/figures/<name>.png and return the path.

    Call this at the end of a chunk; leave the figure open so Quarto also
    embeds it in the report.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path

"""Shared figure style for spike_fm: colors, sizes, rcParams and saving.

Every figure script calls `apply()` first and saves with `save(fig, name)`.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
RESULTS_DIR = REPO / "results"

# Figure widths in inches (typical journal column sizes)
SINGLE_COL = 3.5
DOUBLE_COL = 7.2

# One fixed color per model or quantity, used in every figure.
# Model colors are validated for colorblind separation and contrast on white.
COLORS = {
    "true": "#3a3a38",  # recorded behavior: neutral ink, not a series hue
    "poyo": "#2a78d6",  # blue
    "ridge": "#eb6834",  # orange
}
LABELS = {"true": "Recorded", "poyo": "POYO", "ridge": "Ridge"}
INK = "#0b0b0b"
INK_MUTED = "#6b6a66"
GRID = "#e4e3df"


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_MUTED,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # keep text editable in Illustrator / Inkscape
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save(fig, name, formats=("pdf", "png")):
    """Save to figures/<name>.<ext> and close the figure."""
    FIG_DIR.mkdir(exist_ok=True)
    for ext in formats:
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path)
        print(f"Saved {path.relative_to(REPO)}")
    plt.close(fig)

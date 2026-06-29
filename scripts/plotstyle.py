"""Shared matplotlib style for the committed figures: light surface, hairline grid, one
categorical order (blue, orange, aqua; validated for CVD separation) that always maps to the
same entity: aimd = blue, fixed:32 = orange, fixed:1 = aqua, fixed:8 = yellow. Any other mode
takes the next unused slot of the reference order."""

import matplotlib
import matplotlib.patches  # noqa: F401  (used by callers as plotstyle.matplotlib.patches)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BAND = "#e8f0fb"  # a wash of the blue ramp, for acceptance bands
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA, GREEN, VIOLET = "#eda100", "#e87ba4", "#008300", "#4a3aa7"
MODE_COLORS = {"aimd": BLUE, "fixed:32": ORANGE, "fixed:1": AQUA, "fixed:8": YELLOW}
_SPARE = [MAGENTA, GREEN, VIOLET]


def mode_color(mode: str) -> str:
    if mode not in MODE_COLORS:
        MODE_COLORS[mode] = _SPARE[(len(MODE_COLORS) - 4) % len(_SPARE)]
    return MODE_COLORS[mode]


def modes_in(runs: list[dict]) -> list[str]:
    """Modes in the order they first appear in a results file."""
    seen: list[str] = []
    for r in runs:
        m = r.get("mode")
        if m and m not in seen:
            seen.append(m)
    return seen

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "axes.titlesize": 11,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "font.size": 9.5,
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "legend.labelcolor": INK2,
    "lines.solid_capstyle": "round",
})

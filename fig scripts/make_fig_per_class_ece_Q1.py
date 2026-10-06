import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ---------------------------------------------------------------------
# Publication-quality settings
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": 9.0,

    "axes.titlesize": 10.2,
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.8,

    "axes.linewidth": 0.75,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",

    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#222222",

    "figure.facecolor": "white",
    "savefig.facecolor": "white",

    # Preserve editable/vector text in PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------
# Source values: Table S8
# Numerical values are unchanged.
# ---------------------------------------------------------------------
rows = [
    ("agricultural_farmland", 0.50, 12.46),
    ("baseball_field", 0.30, 10.58),
    ("dense_residential", 1.73, 9.50),
    ("river", 0.69, 7.92),
    ("sparse_residential", 0.96, 7.20),
    ("parking_lot", 0.20, 4.49),
    ("harbor_port", 0.19, 4.28),
    ("medium_residential", 1.09, 5.06),
    ("forest", 0.70, 3.49),
    ("beach", 0.62, 3.00),
    ("storage_tanks", 0.12, 0.59),
    ("overpass_viaduct", 0.13, 0.24),
]

# Sort by LODO inflation
rows.sort(key=lambda r: r[2] - r[1])

classes = [r[0] for r in rows]
pooled = np.asarray([r[1] for r in rows], dtype=float)
lodo = np.asarray([r[2] for r in rows], dtype=float)

inflation = lodo - pooled

# ---------------------------------------------------------------------
# Publication palette
# ---------------------------------------------------------------------
COL_POOLED = "#3B6EA8"       # muted scientific blue
COL_LODO = "#C56A3A"         # muted vermilion
COL_TEXT = "#222222"
COL_GRID = "#D4D7D9"
COL_EDGE = "#4A4A4A"

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(6.25, 4.25),
    dpi=600
)

y = np.arange(len(classes))

# LODO = background/full bar
ax.barh(
    y,
    lodo,
    height=0.58,
    color=COL_LODO,
    edgecolor="white",
    linewidth=0.7,
    zorder=2
)

# Pooled = narrower overlay
ax.barh(
    y,
    pooled,
    height=0.31,
    color=COL_POOLED,
    edgecolor="white",
    linewidth=0.7,
    zorder=3
)

# ---------------------------------------------------------------------
# Difference annotations
# ---------------------------------------------------------------------
for i, (l, diff) in enumerate(zip(lodo, inflation)):

    ax.text(
        l + 0.18,
        i,
        f"+{diff:.1f} pp",
        va="center",
        ha="left",
        fontsize=7.3,
        color=COL_TEXT,
        zorder=5
    )

# ---------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------
ax.set_yticks(y)
ax.set_yticklabels(
    classes,
    fontsize=8.0
)

ax.set_xlabel(
    "Expected Calibration Error (%)",
    labelpad=6
)

# Compact but does NOT clip the 12.46% observation
ax.set_xlim(0, 13.2)

# Clean integer tick positions
ax.set_xticks(
    np.arange(0, 14, 2)
)

# ---------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------
ax.xaxis.grid(
    True,
    linestyle="-",
    linewidth=0.45,
    color=COL_GRID,
    zorder=0
)

ax.set_axisbelow(True)

# ---------------------------------------------------------------------
# Spines
# ---------------------------------------------------------------------
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_color(COL_EDGE)
ax.spines["bottom"].set_color(COL_EDGE)

ax.tick_params(
    axis="both",
    which="major",
    length=3.0,
    width=0.6,
    pad=3
)

# ---------------------------------------------------------------------
# Legend
# Slightly inset from lower-right corner
# ---------------------------------------------------------------------
handles = [
    Patch(
        facecolor=COL_POOLED,
        edgecolor="white",
        label="In-distribution (pooled)"
    ),
    Patch(
        facecolor=COL_LODO,
        edgecolor="white",
        label="LODO (cross-dataset)"
    ),
]

ax.legend(
    handles=handles,
    loc="lower right",
    bbox_to_anchor=(0.97, 0.04),
    frameon=False,
    ncol=1,
    handlelength=1.5,
    handleheight=0.9,
    handletextpad=0.55,
    labelspacing=0.7
)

 

# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
fig.subplots_adjust(
    left=0.245,
    right=0.985,
    bottom=0.14,
    top=0.89
)

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
outdir = Path(
    r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\figures"
)

outdir.mkdir(
    parents=True,
    exist_ok=True
)

png_path = outdir / "fig_s8_per_class_ece_Q1.png"
pdf_path = outdir / "fig_s8_per_class_ece_Q1.pdf"

# 600-dpi raster
fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.055
)

# Vector PDF
fig.savefig(
    pdf_path,
    bbox_inches="tight",
    pad_inches=0.055
)

plt.close(fig)

print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")
print("Figure size before tight bounding-box adjustment: 6.25 × 4.25 inches")
print("PNG resolution: 600 dpi")
print("PDF: vector output")
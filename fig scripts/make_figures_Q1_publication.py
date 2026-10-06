import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ================================================================
# Publication-quality global style
# ================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": 9.0,

    "axes.titlesize": 10.5,
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 7.8,

    "axes.linewidth": 0.75,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#222222",

    "figure.facecolor": "white",
    "savefig.facecolor": "white",

    # Keep text and vector elements editable in PDF.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ================================================================
# Consistent architecture palette used across the manuscript
# ================================================================
ARCH_COLORS = {
    "ResNet-18": "#176C67",
    "CompactCNN": "#C46A24",
    "ConvNeXt-Tiny": "#4F8174",
    "ViT-Small/16": "#3D6EA8",
    "Swin-Tiny": "#8C5A4A",
}

GRID_COLOR = "#D3D7D9"
TEXT_COLOR = "#222222"
EDGE_COLOR = "#444444"

OUTDIR = Path(
    r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\figures"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# Figure 1: LODO accuracy by architecture and held-out dataset
# ================================================================
datasets = ["RESISC45", "UCMerced", "AID"]
archs = [
    "ResNet-18",
    "CompactCNN",
    "ConvNeXt-Tiny",
    "ViT-Small/16",
    "Swin-Tiny",
]

acc = {
    "ResNet-18":       [86.3, 84.3, 95.1],
    "CompactCNN":      [81.3, 65.8, 85.5],
    "ConvNeXt-Tiny":   [83.4, 86.8, 95.6],
    "ViT-Small/16":    [89.0, 86.9, 89.9],
    "Swin-Tiny":       [90.8, 88.0, 91.6],
}

ci_lo = {
    "ResNet-18":       [85.6, 82.2, 94.5],
    "CompactCNN":      [80.5, 63.1, 84.4],
    "ConvNeXt-Tiny":   [82.6, 84.8, 95.0],
    "ViT-Small/16":    [88.3, 84.9, 89.0],
    "Swin-Tiny":       [90.1, 86.1, 90.7],
}

ci_hi = {
    "ResNet-18":       [87.0, 86.4, 95.7],
    "CompactCNN":      [82.1, 68.5, 86.5],
    "ConvNeXt-Tiny":   [84.1, 88.7, 96.2],
    "ViT-Small/16":    [89.6, 88.8, 90.8],
    "Swin-Tiny":       [91.3, 89.8, 92.4],
}

fig, ax = plt.subplots(
    figsize=(6.65, 4.15),
    dpi=600
)

n_arch = len(archs)
x = np.arange(len(datasets))
width = 0.145

for i, arch in enumerate(archs):

    vals = np.asarray(acc[arch], dtype=float)
    lo = vals - np.asarray(ci_lo[arch], dtype=float)
    hi = np.asarray(ci_hi[arch], dtype=float) - vals

    offset = (i - (n_arch - 1) / 2) * width

    ax.bar(
        x + offset,
        vals,
        width=width,
        label=arch,
        color=ARCH_COLORS[arch],
        edgecolor="white",
        linewidth=0.65,
        yerr=[lo, hi],
        capsize=2.5,
        error_kw={
            "elinewidth": 0.8,
            "capthick": 0.8,
            "ecolor": "#333333",
        },
        zorder=3,
    )

ax.set_xticks(x)
ax.set_xticklabels(
    [f"{d}\nheld out" for d in datasets],
    fontsize=8.2
)

ax.set_ylabel(
    "Accuracy (%)",
    labelpad=6
)

# Compact but preserves the full 65.8–96.2% range.
ax.set_ylim(60, 100)

ax.set_yticks(np.arange(60, 101, 5))

ax.yaxis.grid(
    True,
    linestyle="-",
    linewidth=0.45,
    color=GRID_COLOR,
    zorder=0
)

ax.set_axisbelow(True)

# Reference line at 100%, subtle and non-dominant.
ax.axhline(
    100,
    color="#777777",
    linewidth=0.55,
    linestyle="--",
    zorder=1
)

# Clean frame.
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(EDGE_COLOR)
ax.spines["bottom"].set_color(EDGE_COLOR)

ax.tick_params(
    axis="both",
    which="major",
    length=3.0,
    width=0.6,
    pad=3
)

# More breathing room between legend entries.
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=5,
    frameon=False,
    fontsize=7.7,
    columnspacing=1.35,
    handlelength=1.25,
    handletextpad=0.45,
    labelspacing=0.65,
    borderaxespad=0,
)

fig.subplots_adjust(
    left=0.105,
    right=0.985,
    bottom=0.22,
    top=0.965
)

fig.savefig(
    OUTDIR / "fig5_lodo_accuracy_Q1.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.045,
)

fig.savefig(
    OUTDIR / "fig5_lodo_accuracy_Q1.pdf",
    bbox_inches="tight",
    pad_inches=0.045,
)

plt.close(fig)


# ================================================================
# Figure 2: In-distribution vs cross-dataset gap
# ================================================================
archs2 = [
    "Swin-Tiny",
    "ViT-Small/16",
    "ConvNeXt-Tiny",
    "ResNet-18",
    "CompactCNN",
]

pooled_mean = [97.25, 97.46, 97.52, 98.13, 96.59]
lodo_mean = [90.11, 88.60, 88.59, 88.59, 77.49]

gap = [
    p - l
    for p, l in zip(pooled_mean, lodo_mean)
]

fig2, ax2 = plt.subplots(
    figsize=(6.25, 3.55),
    dpi=600
)

y = np.arange(len(archs2))

# Use the same palette as Figure 1.
# Pooled is shown as a light neutral blue-gray so that the
# cross-dataset result remains visually prominent.
POOLED_COLOR = "#AFC1CF"
LODO_COLOR = "#176C67"

ax2.barh(
    y,
    pooled_mean,
    height=0.48,
    color=POOLED_COLOR,
    edgecolor="white",
    linewidth=0.7,
    zorder=2,
)

ax2.barh(
    y,
    lodo_mean,
    height=0.23,
    color=LODO_COLOR,
    edgecolor="white",
    linewidth=0.7,
    zorder=3,
)

# Difference annotations.
for i, (p, l, g) in enumerate(
    zip(pooled_mean, lodo_mean, gap)
):

    ax2.text(
        p + 0.35,
        i,
        f"−{g:.1f} pp",
        va="center",
        ha="left",
        fontsize=7.5,
        color=TEXT_COLOR,
        zorder=5
    )

ax2.set_yticks(y)
ax2.set_yticklabels(
    archs2,
    fontsize=8.2
)

ax2.set_xlabel(
    "Accuracy (%)",
    labelpad=6
)

ax2.set_xlim(70, 101.5)

ax2.set_xticks(
    np.arange(70, 102, 5)
)

ax2.xaxis.grid(
    True,
    linestyle="-",
    linewidth=0.45,
    color=GRID_COLOR,
    zorder=0
)

ax2.set_axisbelow(True)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_color(EDGE_COLOR)
ax2.spines["bottom"].set_color(EDGE_COLOR)

ax2.tick_params(
    axis="both",
    which="major",
    length=3.0,
    width=0.6,
    pad=3
)

handles = [
    Patch(
        facecolor=POOLED_COLOR,
        edgecolor="white",
        label="In-distribution pooled mean (5-fold CV)"
    ),
    Patch(
        facecolor=LODO_COLOR,
        edgecolor="white",
        label="Cross-dataset mean (LODO)"
    ),
]

ax2.legend(
    handles=handles,
    loc="lower right",
    bbox_to_anchor=(0.975, 0.035),
    frameon=False,
    ncol=1,
    fontsize=7.6,
    handlelength=1.4,
    handleheight=0.9,
    handletextpad=0.5,
    labelspacing=0.65,
)

ax2.set_title(
    "In-distribution versus cross-dataset accuracy by architecture",
    fontsize=10.2,
    fontweight="normal",
    pad=9
)

fig2.subplots_adjust(
    left=0.22,
    right=0.985,
    bottom=0.13,
    top=0.88
)

fig2.savefig(
    OUTDIR / "fig6_indist_vs_crossdataset_gap_Q1.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.045,
)

fig2.savefig(
    OUTDIR / "fig6_indist_vs_crossdataset_gap_Q1.pdf",
    bbox_inches="tight",
    pad_inches=0.045,
)

plt.close(fig2)

print("Publication-quality figures saved.")
print(f"Output directory: {OUTDIR}")
print("PNG: 600 dpi")
print("PDF: vector")

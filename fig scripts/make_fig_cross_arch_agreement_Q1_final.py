import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

# ============================================================
# Figure S7 — Cross-architecture agreement on LODO errors
# Publication-quality version
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 8.8,
    "axes.titlesize": 10.2,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 7.4,

    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",

    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,

    "savefig.dpi": 600,
})

# ============================================================
# Output directory
# ============================================================

FIG_DIR = Path(
    r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\figures"
)

FIG_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = FIG_DIR / "fig_s7_cross_arch_agreement_Q1.png"
PDF_OUT = FIG_DIR / "fig_s7_cross_arch_agreement_Q1.pdf"


# ============================================================
# Table S7 values
# ============================================================

folds = [
    "RESISC45",
    "UCMerced",
    "AID",
]

data = {
    "RESISC45": [
        67.53, 15.40, 6.73,
        4.11, 3.05, 3.19
    ],

    "UCMerced": [
        57.50, 19.83, 10.33,
        4.83, 4.25, 3.25
    ],

    "AID": [
        79.05, 10.71, 3.71,
        3.55, 1.36, 1.62
    ],
}


# ============================================================
# Category labels
# ============================================================

seg_labels = [
    "0 of 5\n(all correct)",
    "1 of 5",
    "2 of 5",
    "3 of 5",
    "4 of 5",
    "5 of 5\n(all wrong)",
]


# ============================================================
# Sequential publication palette
#
# Light:
#   fewer architectures wrong
#
# Dark:
#   stronger cross-architecture failure agreement
# ============================================================

colors = [
    "#E3EEF5",
    "#C4DDEB",
    "#9BC5DA",
    "#6FA7C0",
    "#4B819D",
    "#2F6078",
]


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.45, 2.95),
    dpi=600,
)


# ============================================================
# Horizontal stacked bars
# ============================================================

y = np.arange(len(folds))

left = np.zeros(len(folds))

height = 0.58


for i, seg in enumerate(seg_labels):

    vals = np.asarray(
        [data[f][i] for f in folds],
        dtype=float,
    )

    ax.barh(
        y,
        vals,
        left=left,
        height=height,
        color=colors[i],
        edgecolor="white",
        linewidth=0.65,
        zorder=3,
        label=seg,
    )

    # --------------------------------------------------------
    # Value labels
    # Avoid clutter in very narrow segments.
    # --------------------------------------------------------

    for j, v in enumerate(vals):

        if v >= 3.0:

            text_color = (
                "white"
                if i >= 3
                else "#222222"
            )

            ax.text(
                left[j] + v / 2.0,
                y[j],
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="medium",
                color=text_color,
                zorder=5,
            )

    left += vals


# ============================================================
# Axes
# ============================================================

ax.set_yticks(y)

ax.set_yticklabels(
    [
        f"{dataset}\nheld out"
        for dataset in folds
    ]
)

ax.set_xlabel(
    "Images (%)",
    labelpad=4,
)

ax.set_xlim(
    0,
    100,
)

ax.xaxis.set_major_locator(
    MultipleLocator(20)
)


# ============================================================
# Subtle quantitative grid
# ============================================================

ax.grid(
    axis="x",
    linestyle=":",
    linewidth=0.5,
    color="#999999",
    alpha=0.75,
    zorder=0,
)

ax.set_axisbelow(True)


# ============================================================
# Clean frame
# ============================================================

for spine in [
    "top",
    "right",
]:
    ax.spines[spine].set_visible(False)

ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")

ax.tick_params(
    axis="y",
    pad=4,
)


 


# ============================================================
# Legend below the figure
# ============================================================

legend = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.255),
    ncol=6,
    frameon=False,
    columnspacing=0.9,
    handlelength=1.15,
    handleheight=0.75,
    handletextpad=0.35,
    borderaxespad=0.0,
    title="Architectures misclassifying the image (of 5)",
)

legend.get_title().set_fontsize(7.6)
legend.get_title().set_color("#222222")


# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.145,
    right=0.995,
    top=0.83,
    bottom=0.30,
)

fig.patch.set_facecolor("white")
ax.set_facecolor("white")


# ============================================================
# Export
# ============================================================

fig.savefig(
    PNG_OUT,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
)

plt.close(fig)


print(f"Saved PNG: {PNG_OUT}")
print(f"Saved PDF: {PDF_OUT}")
print("PNG resolution: 600 dpi")
print("PDF: vector output")

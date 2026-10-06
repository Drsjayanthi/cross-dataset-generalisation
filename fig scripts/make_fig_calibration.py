import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Figure 9 — Calibration
# Figure 6 visual language:
#   In-distribution pooled = blue
#   Cross-dataset LODO    = vermilion
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.2,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.0,
})

COL_POOLED = "#3B6EA8"
COL_LODO = "#C56A3A"

archs = [
    "Swin-Tiny",
    "ViT-Small/16",
    "ConvNeXt-Tiny",
    "ResNet-18",
    "CompactCNN",
]

pooled_ece = [1.08, 0.97, 0.91, 0.42, 0.73]
lodo_ece = [4.76, 6.05, 5.22, 6.01, 11.15]

fig, ax = plt.subplots(figsize=(6.25, 3.65), dpi=600)
y = np.arange(len(archs))

# Wider LODO/background bars
ax.barh(
    y, lodo_ece,
    height=0.58,
    color=COL_LODO,
    label="Cross-dataset (LODO) ECE",
    edgecolor="white",
    linewidth=0.7,
    zorder=3,
)

# Narrower pooled/foreground bars
ax.barh(
    y, pooled_ece,
    height=0.32,
    color=COL_POOLED,
    label="In-distribution pooled ECE",
    edgecolor="white",
    linewidth=0.7,
    zorder=4,
)

# Difference annotations
for i, (p, l) in enumerate(zip(pooled_ece, lodo_ece)):
    difference = l - p
    ax.text(
        l + 0.25, i,
        f"+{difference:.1f} pp",
        va="center",
        ha="left",
        fontsize=7.8,
        fontweight="bold",
        color="#222222",
    )

ax.set_yticks(y)
ax.set_yticklabels(archs)

ax.set_xlabel(
    "Expected Calibration Error (%)",
    labelpad=6,
)

ax.set_xlim(0, 13.5)
ax.set_xticks(np.arange(0, 14, 2))

ax.xaxis.grid(
    True,
    linestyle=":",
    linewidth=0.55,
    color="#999999",
    alpha=0.8,
    zorder=0,
)
ax.set_axisbelow(True)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")



handles = [
    plt.Rectangle(
        (0, 0), 1, 1,
        facecolor=COL_POOLED,
        edgecolor="white",
        linewidth=0.7,
    ),
    plt.Rectangle(
        (0, 0), 1, 1,
        facecolor=COL_LODO,
        edgecolor="white",
        linewidth=0.7,
    ),
]

ax.legend(
    handles,
    [
        "In-distribution pooled ECE",
        "Cross-dataset (LODO) ECE",
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.19),
    ncol=2,
    frameon=False,
    fontsize=7.7,
    handlelength=1.4,
    handleheight=0.9,
    handletextpad=0.55,
    columnspacing=1.5,
    borderaxespad=0,
)

fig.subplots_adjust(
    left=0.18,
    right=0.98,
    bottom=0.25,
    top=0.88,
)

PNG_PATH = (
    r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study"
    r"\figures\fig9_calibration_Fig6colors.png"
)

PDF_PATH = (
    r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study"
    r"\figures\fig9_calibration_Fig6colors.pdf"
)

plt.savefig(
    PNG_PATH,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.10,
    facecolor="white",
)

plt.savefig(
    PDF_PATH,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.10,
    facecolor="white",
)

plt.close()

print("Saved PNG:")
print(PNG_PATH)
print("Saved PDF:")
print(PDF_PATH)

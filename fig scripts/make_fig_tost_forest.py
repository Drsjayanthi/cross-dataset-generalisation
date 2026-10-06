import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Publication-quality style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": 9,

    "axes.titlesize": 10.2,
    "axes.labelsize": 9.2,
    "axes.linewidth": 0.8,

    # Stronger, readable typography
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "text.color": "#222222",

    "xtick.labelsize": 8.2,
    "ytick.labelsize": 7.8,

    "savefig.facecolor": "white",
})


# ============================================================
# Architecture order
# ============================================================
ARCHS = [
    "ResNet-18",
    "ConvNeXt-Tiny",
    "ViT-Small/16",
    "Swin-Tiny",
    "CompactCNN"
]


# ============================================================
# Brighter publication-safe architecture colors
# ============================================================
COLORS = {
    "ResNet-18": "#00796B",       # vivid teal
    "ConvNeXt-Tiny": "#2E7D6E",   # green-teal
    "ViT-Small/16": "#1976B8",    # strong blue
    "Swin-Tiny": "#8A4F3D",       # warm brown-red
    "CompactCNN": "#D06420",      # orange
}


# ============================================================
# TOST equivalence margins
# ============================================================
MARGIN = {
    "ResNet-18": 0.50,
    "ConvNeXt-Tiny": 0.23,
    "ViT-Small/16": 0.42,
    "Swin-Tiny": 0.58,
    "CompactCNN": 0.39,
}


# ============================================================
# Data
# (architecture, held-out dataset, difference, CI lower, CI upper)
# ============================================================
DATA = [

    ("ResNet-18", "AID",
     -3.01, -3.56, -2.49),

    ("ResNet-18", "UCMerced",
     -13.82, -15.63, -12.05),

    ("ResNet-18", "RESISC45",
     -11.82, -12.41, -11.24),

    ("ConvNeXt-Tiny", "AID",
     -1.88, -2.41, -1.36),

    ("ConvNeXt-Tiny", "UCMerced",
     -10.76, -12.44, -9.11),

    ("ConvNeXt-Tiny", "RESISC45",
     -14.16, -14.80, -13.50),

    ("ViT-Small/16", "AID",
     -7.55, -8.32, -6.77),

    ("ViT-Small/16", "UCMerced",
     -10.56, -12.21, -8.96),

    ("ViT-Small/16", "RESISC45",
     -8.49, -9.03, -7.98),

    ("Swin-Tiny", "AID",
     -5.67, -6.39, -4.94),

    ("Swin-Tiny", "UCMerced",
     -9.26, -10.83, -7.75),

    ("Swin-Tiny", "RESISC45",
     -6.49, -7.00, -5.98),

    ("CompactCNN", "AID",
     -11.13, -12.04, -10.21),

    ("CompactCNN", "UCMerced",
     -30.81, -33.09, -28.50),

    ("CompactCNN", "RESISC45",
     -15.33, -15.99, -14.67),
]


# ============================================================
# Create figure
# ============================================================
fig, (ax_wide, ax_zoom) = plt.subplots(
    1,
    2,
    figsize=(9.25, 4.85),
    dpi=600,
    gridspec_kw={
        "width_ratios": [2.15, 1],
        "wspace": 0.08
    }
)


# ============================================================
# Build y positions
# ============================================================
y_labels = []
yticks = []
group_bounds = {}

y = 0

for arch in ARCHS:

    start_y = y

    rows = [
        d for d in DATA
        if d[0] == arch
    ]

    for _, dataset, diff, ci_lo, ci_hi in rows:

        y_labels.append(
            f"{arch} – {dataset}"
        )

        yticks.append(y)

        color = COLORS[arch]

        # --------------------------------------------
        # Confidence interval
        # --------------------------------------------
        for ax in (ax_wide, ax_zoom):

            ax.plot(
                [ci_lo, ci_hi],
                [y, y],
                color=color,
                linewidth=2.0,
                solid_capstyle="round",
                zorder=3
            )

            # ----------------------------------------
            # Point estimate
            # ----------------------------------------
            ax.plot(
                diff,
                y,
                "o",
                color=color,
                markersize=5.0,
                markeredgecolor="white",
                markeredgewidth=0.9,
                zorder=4
            )

        y -= 1

    group_bounds[arch] = (
        start_y,
        y + 1
    )

    # Gap between architecture groups
    y -= 0.62


# ============================================================
# Common axis formatting
# ============================================================
for ax in (ax_wide, ax_zoom):

    # Zero difference reference line
    ax.axvline(
        0,
        color="#555555",
        linewidth=0.9,
        zorder=1
    )

    # Subtle vertical grid
    ax.grid(
        axis="x",
        color="#C8C8C8",
        linewidth=0.55,
        alpha=0.8
    )

    ax.set_axisbelow(True)

    # Remove unnecessary borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")


# ============================================================
# Equivalence-margin bands
# ============================================================
for arch in ARCHS:

    top, bottom = group_bounds[arch]

    margin = MARGIN[arch]

    ax_zoom.fill_betweenx(
        [
            bottom - 0.5,
            top + 0.5
        ],
        -margin,
        margin,
        color="#D0D0D0",
        alpha=0.65,
        zorder=0
    )


# ============================================================
# LEFT PANEL — Full range
# ============================================================
ax_wide.set_yticks(yticks)

ax_wide.set_yticklabels(
    y_labels
)


# Color the dataset labels according to architecture
for tick, (arch, _) in zip(
    ax_wide.get_yticklabels(),
    [(d[0], d[1]) for d in DATA]
):

    tick.set_color(
        COLORS[arch]
    )

    tick.set_fontweight(
        "bold"
    )


ax_wide.set_xlabel(
    "LODO accuracy − in-distribution pooled accuracy "
    "(percentage points)",
    labelpad=6,
    color="#222222"
)

ax_wide.set_title(
    "(a) Full range",
    fontsize=10.2,
    fontweight="bold",
    color="#222222",
    pad=7
)

ax_wide.set_xlim(
    -35,
    3
)

ax_wide.set_xticks(
    np.arange(-35, 4, 5)
)


# ============================================================
# RIGHT PANEL — Zoomed near zero
# ============================================================
ax_zoom.set_yticks(
    yticks
)

ax_zoom.set_yticklabels([])

ax_zoom.set_xlabel(
    "Difference (percentage points)",
    labelpad=6,
    color="#222222"
)

ax_zoom.set_title(
    "(b) Near-zero differences\n"
    "shaded = equivalence margin",
    fontsize=9.2,
    fontweight="bold",
    color="#222222",
    pad=7
)

ax_zoom.set_xlim(
    -3,
    3
)

ax_zoom.set_xticks(
    np.arange(-3, 4, 1)
)


# ============================================================
# Final layout
# ============================================================
fig.subplots_adjust(
    left=0.235,
    right=0.985,
    bottom=0.18,
    top=0.94
)


# ============================================================
# Output paths
# ============================================================
PNG_PATH = (
    r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study"
    r"\figures\fig7_tost_forest_Q1_v2_bright.png"
)

PDF_PATH = (
    r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study"
    r"\figures\fig7_tost_forest_Q1_v2_bright.pdf"
)


# ============================================================
# Save 600-dpi PNG and vector PDF
# ============================================================
plt.savefig(
    PNG_PATH,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.12,
    facecolor="white"
)

plt.savefig(
    PDF_PATH,
    bbox_inches="tight",
    pad_inches=0.12,
    facecolor="white"
)

plt.close()

print("Saved PNG:")
print(PNG_PATH)

print("\nSaved vector PDF:")
print(PDF_PATH)
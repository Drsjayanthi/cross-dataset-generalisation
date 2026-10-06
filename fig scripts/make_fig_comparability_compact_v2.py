import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({
    "font.family": "Times New Roman",
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

INK = "black"

COLS = ["RESISC45", "UCMerced", "AID", "EuroSAT-RGB"]
FILES = {
    "RESISC45": r"D:\SJ\UVMDATA\NWPU-RESISC45\forest\forest_001.jpg",
    "UCMerced": r"D:\SJ\UVMDATA\UCMerced_LandUse\UCMerced_landuse\forest\forest00.tif",
    "AID": r"D:\SJ\UVMDATA\AID\AID\Forest\forest_1.jpg",
    "EuroSAT-RGB": r"D:\SJ\UVMDATA\EuroSAT-RGB\EuroSAT_RGB\Forest\Forest_1.jpg",
}

# Compact single-row layout: dataset labels sit immediately above each image.
fig, axes = plt.subplots(
    1, 4,
    figsize=(4.5, 1.32),
    dpi=600,
    gridspec_kw={"wspace": 0.055}
)

for ax, name in zip(axes, COLS):
    im = Image.open(FILES[name]).convert("RGB")
    ax.imshow(im, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    # Minimal label-to-image spacing.
    ax.set_title(
        name,
        fontsize=8.7,
        fontweight="bold",
        color=INK,
        pad=2.0
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")
        spine.set_linewidth(0.45)

# Remove outer margins while retaining a tiny breathing space.
fig.subplots_adjust(left=0.008, right=0.992, bottom=0.015, top=0.84)

out_png = r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\figures\fig2_comparability_compact.png"
out_pdf = r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\figures\fig2_comparability_compact.pdf"

plt.savefig(
    out_png,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.025,
    facecolor="white"
)
plt.savefig(
    out_pdf,
    bbox_inches="tight",
    pad_inches=0.025,
    facecolor="white"
)
plt.close()

print("Compact comparability figure saved.")
print(out_png)
print(out_pdf)

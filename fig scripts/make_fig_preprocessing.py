import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

plt.rcParams.update({
    "font.family": "Times New Roman",
    "text.color": "black", "axes.labelcolor": "black",
    "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black",
})

INK = "black"
MUTED = "#56635C"

# Same normalisation constants as common_train.py (ImgDataset)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_SIZE = 224

SRC_PATH = r"D:\SJ\UVMDATA\NWPU-RESISC45\baseball_diamond\baseball_diamond_001.jpg"
OUT_DIR = r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\figures"

include_hist = "--with-hist" in sys.argv

orig = Image.open(SRC_PATH).convert("RGB")
resized = TF.resize(orig, [TARGET_SIZE, TARGET_SIZE])
hflip = TF.hflip(resized)
vflip = TF.vflip(resized)
rotated = TF.rotate(resized, angle=15, fill=0)

panels = [
    (f"Original\n({orig.size[0]}\u00d7{orig.size[1]})", orig),
    (f"Resized\n({TARGET_SIZE}\u00d7{TARGET_SIZE})", resized),
    ("Horizontal flip", hflip),
    ("Vertical flip", vflip),
    ("Rotation (+15\u00b0)", rotated),
]

if include_hist:
    fig = plt.figure(figsize=(7.2, 2.05), dpi=600) 
    gs = gridspec.GridSpec( nrows=1, ncols=5, figure=fig, left=0.01, right=0.995, top=0.88, bottom=0.04, wspace=0.03)
 
else:
    fig = plt.figure(figsize=(8.5, 2.05), dpi=600)
    gs = gridspec.GridSpec(nrows=1, ncols=5, figure=fig, wspace=0.03)

for j, (title, im) in enumerate(panels):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(im, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#999999")
        spine.set_linewidth(0.5)
    ax.set_title(title, fontsize=7.6, fontweight="bold", color=INK, pad=4, linespacing=1.2)

if include_hist:
    # ToTensor-equivalent: scale to [0,1]; Normalize-equivalent: per-channel (x-mean)/std
    arr01 = np.asarray(resized).astype(np.float32) / 255.0  # H,W,3 in [0,1]
    arr_norm = np.empty_like(arr01)
    for c in range(3):
        arr_norm[..., c] = (arr01[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    chan_names = ["Red", "Green", "Blue"]
    chan_colors = ["#B0413E", "#3B7A57", "#3B6FA0"]
    for c in range(3):
        ax = fig.add_subplot(gs[1, c + 1])
        vals01 = arr01[..., c].ravel()
        valsn = arr_norm[..., c].ravel()
        ax.hist(vals01, bins=40, range=(0, 1), color="#B8BEB8", alpha=0.85,
                label="After ToTensor  [0,1]", density=True)
        ax2 = ax.twiny()
        ax2.hist(valsn, bins=40, color=chan_colors[c], alpha=0.55,
                 label="After Normalise", density=True)
        ax.set_yticks([])
        ax2.set_yticks([])
        ax.tick_params(axis="x", labelsize=5.6, colors=MUTED)
        ax2.tick_params(axis="x", labelsize=5.6, colors=chan_colors[c])
        ax.set_xlim(0, 1)
        ax.set_title(f"{chan_names[c]}  (\u03bc={arr01[...,c].mean():.2f}\u2192{arr_norm[...,c].mean():.2f})",
                     fontsize=6.6, color=INK, pad=14)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
            ax2.spines[spine].set_visible(False)

    # caption-style axis for the two remaining slots + a shared legend
    ax_leg = fig.add_subplot(gs[1, 0])
    ax_leg.axis("off")
    ax_leg.text(0.0, 0.75, "Pixel-value\ndistribution", ha="left", va="top", fontsize=7.2,
                weight="bold", color=INK, linespacing=1.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#B8BEB8", alpha=0.85),
               plt.Rectangle((0, 0), 1, 1, color="#808080", alpha=0.55)]
    ax_leg.legend(handles, ["After ToTensor [0,1]", "After Normalise"],
                  loc="lower left", fontsize=5.8, frameon=False, handlelength=1.0,
                  bbox_to_anchor=(0.0, 0.0))
    ax_dummy = fig.add_subplot(gs[1, 4])
    ax_dummy.axis("off")

# Figure 4 in the manuscript is the basic (no-histogram) version, finalised 2026-09-06.
# The histogram variant was evaluated and dropped; re-running --with-hist regenerates it
# into unused/ for reference rather than overwriting the published Figure 4.
if include_hist:
    out_path = f"{OUT_DIR}\\unused\\fig4_preprocessing_ALT_with_histogram.png"
else:
    out_path = f"{OUT_DIR}\\fig4_preprocessing.png"
plt.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
plt.close()
print(f"Saved: {out_path}")

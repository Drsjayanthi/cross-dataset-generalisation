import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 8,
    "text.color": "black", "axes.labelcolor": "black",
    "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black",
})

BASE = r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results\studyA_lodo"
HELD_OUTS = ["RESISC45", "UCMerced", "AID"]
ARCH = "resnet18"  # reference architecture, consistent with Table 7's McNemar comparisons

SHORT = {
    "agricultural_farmland": "agri.\nfarmland", "baseball_field": "baseball\nfield", "beach": "beach",
    "dense_residential": "dense\nresid.", "forest": "forest", "harbor_port": "harbor\nport",
    "medium_residential": "medium\nresid.", "overpass_viaduct": "overpass\nviaduct",
    "parking_lot": "parking\nlot", "river": "river", "sparse_residential": "sparse\nresid.",
    "storage_tanks": "storage\ntanks",
}

fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6), dpi=600,
                          gridspec_kw={"wspace": 0.38})

for ax, ds in zip(axes, HELD_OUTS):
    with open(os.path.join(BASE, f"lodo_heldout-{ds}_{ARCH}", "metrics.json")) as f:
        m = json.load(f)
    classes = m["classes"]
    cm = np.array(m["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / row_sums  # row-normalised: recall-style, each row sums to 1

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    labels = [SHORT[c] for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.6)
    ax.set_yticklabels(labels, fontsize=5.6)
    ax.set_title(f"{ds} held out", fontsize=9, fontweight="bold", pad=6)
    if ds == HELD_OUTS[0]:
        ax.set_ylabel("True class", fontsize=7.5)
    ax.set_xlabel("Predicted class", fontsize=7.5)
    # annotate cells with >= 10% off-diagonal confusion
    n = len(classes)
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if (i != j and v >= 0.10) or (i == j and v < 0.85):
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                        fontsize=4.8, color="white" if v > 0.5 else "black")

cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.01)
cbar.set_label("Row-normalised proportion (recall)", fontsize=7.5)
cbar.ax.tick_params(labelsize=6)

plt.savefig(
    r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\figures\figS3_confusion_matrices_lodo.png",
    dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
plt.close()
print("Saved figS3_confusion_matrices_lodo.png")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.2,

    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
})

# Same order/values as Table 11 (corrected, per-fold-averaged methodology; see
# benchmark_study/scripts/08_calibration_analysis.py)
archs = ["Swin-Tiny", "ViT-Small/16", "ConvNeXt-Tiny", "ResNet-18", "CompactCNN"]
pooled = [31.3, 35.9, 27.6, 29.5, 25.3]
lodo = [41.6, 36.8, 30.1, 36.9, 31.7]

fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=600)
y = np.arange(len(archs))

ax.barh(y, lodo, height=0.55, color="#C56A3A", label="LODO (cross-dataset)",
        edgecolor="black", linewidth=0.4, zorder=3)
ax.barh(y, pooled, height=0.30, color="#3B6EA8", label="In-distribution (pooled)",
        edgecolor="black", linewidth=0.4, zorder=4)

for i, (p, l) in enumerate(zip(pooled, lodo)):
    ax.text(l + 0.6, i, f"+{l-p:.1f} pp", va="center", fontsize=7.5, color="#222222")

ax.set_yticks(y)
ax.set_yticklabels(archs)
ax.set_xlabel("Misclassified predictions with confidence \u2265 90% (%)")
ax.set_xlim(0, 58)
ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="#999999", zorder=0)
ax.set_axisbelow(True)
handles = [plt.Rectangle((0, 0), 1, 1, color="#3B6EA8", ec="black", lw=0.4),
           plt.Rectangle((0, 0), 1, 1, color="#C56A3A", ec="black", lw=0.4)]
ax.legend(
    handles,
    ["In-distribution (pooled)", "LODO (cross-dataset)"],
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

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.subplots_adjust(left=0.18, right=0.98, bottom=0.25, top=0.88)
plt.savefig(
    r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\figures\fig10_confidence_stratification_Fig6colors.png",
    dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
plt.close()
print("Saved fig10_confidence_stratification_Fig6colors.png")

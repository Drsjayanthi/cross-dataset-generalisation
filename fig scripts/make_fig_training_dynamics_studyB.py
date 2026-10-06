import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
})

BASE = r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results\studyB_eurosat"
RUN_PREFIX = "final_helodtest_"  # Design B2: the single final model fit reported in Table 8

ARCHS = ["resnet18", "compactcnn", "convnext_tiny", "vit_small_patch16_224", "swin_tiny_patch4_window7_224"]
LABELS = {
    "resnet18": "ResNet-18",
    "compactcnn": "CompactCNN",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_small_patch16_224": "ViT-Small/16",
    "swin_tiny_patch4_window7_224": "Swin-Tiny",
}
COLORS = {
    "resnet18": "#1F6E64",
    "compactcnn": "#C06B2C",
    "convnext_tiny": "#4C7A6E",
    "vit_small_patch16_224": "#3B6FA0",
    "swin_tiny_patch4_window7_224": "#8E5C4E",
}

histories = {}
for a in ARCHS:
    p = os.path.join(BASE, f"{RUN_PREFIX}{a}", "history.json")
    with open(p) as f:
        histories[a] = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.15), dpi=600)

for a in ARCHS:
    h = histories[a]
    epochs = [e["epoch"] + 1 for e in h]
    loss = [e["train_loss"] for e in h]
    acc = [e["val_acc"] * 100 for e in h]
    c = COLORS[a]
    ax1.plot(epochs, loss, color=c, linewidth=1.3, marker="o", markersize=2.4, label=LABELS[a])
    ax1.plot(epochs[-1], loss[-1], marker="s", markersize=4.2, color=c, markeredgecolor="black",
              markeredgewidth=0.4, zorder=5)
    ax2.plot(epochs, acc, color=c, linewidth=1.3, marker="o", markersize=2.4, label=LABELS[a])
    ax2.plot(epochs[-1], acc[-1], marker="s", markersize=4.2, color=c, markeredgecolor="black",
              markeredgewidth=0.4, zorder=5)

ax1.set_yscale("log")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training loss (log scale)")

ax1.yaxis.grid(True, which="both", linestyle=":", linewidth=0.4, color="#bbbbbb", zorder=0)
ax1.set_axisbelow(True)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation accuracy (%)")

ax2.yaxis.grid(True, linestyle=":", linewidth=0.4, color="#bbbbbb", zorder=0)
ax2.set_axisbelow(True)

for ax in (ax1, ax2):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7.6,
           bbox_to_anchor=(0.5, -0.01), columnspacing=1.2, handlelength=1.4, handletextpad=0.4)

plt.tight_layout(rect=[0, 0.06, 1, 1])
out_path = r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\figures\figS2_training_dynamics_studyB.png"
plt.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
plt.close()
print(f"Saved: {out_path}")
for a in ARCHS:
    print(LABELS[a], "-> epochs run:", len(histories[a]))

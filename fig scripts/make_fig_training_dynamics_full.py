import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8.5,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
})

BASE = r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results\studyA_lodo"
HELD_OUTS = ["RESISC45", "UCMerced", "AID"]

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

fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.6), dpi=600)

for col, held_out in enumerate(HELD_OUTS):
    histories = {}
    for a in ARCHS:
        p = os.path.join(BASE, f"lodo_heldout-{held_out}_{a}", "history.json")
        with open(p) as f:
            histories[a] = json.load(f)

    ax_loss = axes[0, col]
    ax_acc = axes[1, col]

    for a in ARCHS:
        h = histories[a]
        epochs = [e["epoch"] + 1 for e in h]
        loss = [e["train_loss"] for e in h]
        acc = [e["val_acc"] * 100 for e in h]
        c = COLORS[a]
        ax_loss.plot(epochs, loss, color=c, linewidth=1.1, marker="o", markersize=1.8, label=LABELS[a])
        ax_loss.plot(epochs[-1], loss[-1], marker="s", markersize=3.4, color=c,
                     markeredgecolor="black", markeredgewidth=0.35, zorder=5)
        ax_acc.plot(epochs, acc, color=c, linewidth=1.1, marker="o", markersize=1.8, label=LABELS[a])
        ax_acc.plot(epochs[-1], acc[-1], marker="s", markersize=3.4, color=c,
                    markeredgecolor="black", markeredgewidth=0.35, zorder=5)

    ax_loss.set_yscale("log")
    ax_loss.set_title(f"{held_out} held out", fontsize=9, pad=5, fontweight="bold")
    ax_loss.yaxis.grid(True, which="both", linestyle=":", linewidth=0.35, color="#bbbbbb", zorder=0)
    ax_loss.set_axisbelow(True)
    ax_acc.yaxis.grid(True, linestyle=":", linewidth=0.35, color="#bbbbbb", zorder=0)
    ax_acc.set_axisbelow(True)
    ax_acc.set_xlabel("Epoch", fontsize=8)

    if col == 0:
        ax_loss.set_ylabel("Training loss\n(log scale)", fontsize=8.3)
        ax_acc.set_ylabel("Validation\naccuracy (%)", fontsize=8.3)

    for ax in (ax_loss, ax_acc):
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=7.2)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.2,
           bbox_to_anchor=(0.5, -0.02), columnspacing=1.3, handlelength=1.5, handletextpad=0.4)

plt.tight_layout(rect=[0, 0.05, 1, 1])
out_path = r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\figures\figS1_training_dynamics_all_folds.png"
plt.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
plt.close()
print(f"Saved: {out_path}")

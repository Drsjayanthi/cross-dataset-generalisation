"""
Stage 8: Calibration analysis for Section 3.8 and Supplementary Section S8.

Produces, from the predictions.csv already written by every training run:
  - Table 9  : per-architecture Expected Calibration Error (ECE), pooled
               in-distribution (Design A2) vs cross-dataset (Design A1/LODO).
  - Figure 8 : bar-chart data underlying the above (see
               manuscript/fig_scripts/make_fig_calibration.py, which
               consumes the hardcoded values printed here).
  - Table 11 : proportion of misclassified predictions made with softmax
               confidence >= 90%, in-distribution vs LODO.
  - Table S8 : per-class ECE, pooled across all five architectures,
               in-distribution vs LODO.

Two different aggregation conventions are used, and BOTH are load-bearing
for the numbers actually published in the manuscript (Section 2.6):

  * Architecture-level statistics (Tables 9 and 11) are computed PER FOLD,
    then averaged across folds with a simple (unweighted) mean. This
    matches the fold-averaging convention used throughout the study for
    accuracy and macro-F1 (e.g. Section 2.4, Table 6's pooled means).
    IMPORTANT: pooling all folds' raw predictions together and computing
    one ECE over the pooled set gives a NUMERICALLY DIFFERENT (and wrong,
    for reproducing the manuscript) result, because ECE is a non-linear
    (bin-weighted-absolute-difference) statistic and the LODO folds are
    very unequal in size (RESISC45 n=9100, UCMerced n=1200, AID n=4200).
    This was verified empirically while writing this script: pooling
    LODO predictions first gives ResNet-18 LODO ECE = 6.33%, whereas the
    published, correct value (per-fold ECE averaged) is 6.01%.

  * Class-level ECE (Table S8) pools predictions across ALL FIVE
    architectures AND all folds before computing one ECE per class. This
    is a deliberate departure from the architecture-level convention:
    an individual (architecture, fold, class) subset can be as small as
    100 images (e.g. any UCMerced class), which is too few to support a
    stable per-fold ECE estimate under ten-bin partitioning. Pooling
    across architectures and folds gives each class several thousand
    supporting predictions instead.

ECE definition (Guo et al., 2017): predictions are partitioned into ten
equal-width confidence bins by their maximum softmax probability; ECE is
the weighted average, across bins, of |mean confidence - empirical
accuracy| in that bin, weighted by the bin's share of all predictions.
"""
import os
import numpy as np
import pandas as pd

RESULTS = r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results"
BASE_LODO = os.path.join(RESULTS, "studyA_lodo")
BASE_POOLED = os.path.join(RESULTS, "studyA_pooled_cv")

ARCHS = ["resnet18", "compactcnn", "convnext_tiny", "vit_small_patch16_224", "swin_tiny_patch4_window7_224"]
LABELS = {
    "resnet18": "ResNet-18", "compactcnn": "CompactCNN", "convnext_tiny": "ConvNeXt-Tiny",
    "vit_small_patch16_224": "ViT-Small/16", "swin_tiny_patch4_window7_224": "Swin-Tiny",
}
LODO_FOLDS = ["RESISC45", "UCMerced", "AID"]
CLASSES = [
    "agricultural_farmland", "baseball_field", "beach", "dense_residential", "forest",
    "harbor_port", "medium_residential", "overpass_viaduct", "parking_lot", "river",
    "sparse_residential", "storage_tanks",
]


def ece(df, n_bins=10):
    """Expected Calibration Error over one set of predictions (needs 'confidence', 'true_label', 'pred_label')."""
    if len(df) < 10:
        return None
    conf = df["confidence"].values
    correct = (df["true_label"] == df["pred_label"]).values.astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(conf)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return total


def frac_high_conf_errors(df, threshold=0.9):
    wrong = df[df["true_label"] != df["pred_label"]]
    if len(wrong) == 0:
        return None
    return (wrong["confidence"] > threshold).mean()


def load_pooled_fold(arch, fold):
    suf = "" if arch == "resnet18" else "_" + arch
    return pd.read_csv(os.path.join(BASE_POOLED, f"pooled_fold{fold}{suf}", "predictions.csv"))


def load_lodo_fold(arch, held_out):
    return pd.read_csv(os.path.join(BASE_LODO, f"lodo_heldout-{held_out}_{arch}", "predictions.csv"))


def main():
    pooled_by_fold = {a: [load_pooled_fold(a, f) for f in range(5)] for a in ARCHS}
    lodo_by_fold = {a: [load_lodo_fold(a, ds) for ds in LODO_FOLDS] for a in ARCHS}

    print("=== Table 9 / Figure 8: per-architecture ECE (per-fold, averaged), pooled vs LODO ===")
    for a in ARCHS:
        e_pool = np.mean([ece(df) for df in pooled_by_fold[a]])
        e_lodo = np.mean([ece(df) for df in lodo_by_fold[a]])
        print(f"{LABELS[a]:16s} pooled_ECE={e_pool*100:5.2f}%  LODO_ECE={e_lodo*100:5.2f}%  "
              f"inflation=+{100*(e_lodo-e_pool):5.2f}pp")

    print()
    print("=== Table 11: proportion of misclassified predictions with confidence >= 90% (per-fold, averaged) ===")
    for a in ARCHS:
        p_pool = np.mean([frac_high_conf_errors(df) for df in pooled_by_fold[a]])
        p_lodo = np.mean([frac_high_conf_errors(df) for df in lodo_by_fold[a]])
        print(f"{LABELS[a]:16s} pooled={p_pool*100:5.1f}%  LODO={p_lodo*100:5.1f}%  "
              f"change=+{100*(p_lodo-p_pool):5.1f}pp")

    print()
    print("=== Context: overall mean confidence, all predictions (per-fold, averaged; used in Section 3.8) ===")
    for a in ARCHS:
        m_pool = np.mean([df["confidence"].mean() for df in pooled_by_fold[a]])
        m_lodo = np.mean([df["confidence"].mean() for df in lodo_by_fold[a]])
        print(f"{LABELS[a]:16s} mean_conf_pooled={m_pool:.3f}  mean_conf_LODO={m_lodo:.3f}  "
              f"reduction={m_pool-m_lodo:.5f}")

    print()
    print("=== Table S8: per-class ECE, POOLED across all 5 architectures and all folds (see module docstring) ===")
    pooled_all = {a: pd.concat(pooled_by_fold[a], ignore_index=True) for a in ARCHS}
    lodo_all = {a: pd.concat(lodo_by_fold[a], ignore_index=True) for a in ARCHS}
    for cls in CLASSES:
        pool_cls = pd.concat([pooled_all[a][pooled_all[a]["true_label"] == cls] for a in ARCHS], ignore_index=True)
        lodo_cls = pd.concat([lodo_all[a][lodo_all[a]["true_label"] == cls] for a in ARCHS], ignore_index=True)
        e_pool = ece(pool_cls)
        e_lodo = ece(lodo_cls)
        print(f"{cls:22s} pooled_ECE={e_pool*100:5.2f}%  LODO_ECE={e_lodo*100:5.2f}%  "
              f"inflation=+{100*(e_lodo-e_pool):5.2f}pp")


if __name__ == "__main__":
    main()

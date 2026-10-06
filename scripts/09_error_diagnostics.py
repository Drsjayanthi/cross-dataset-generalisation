"""
Stage 9: Error-structure diagnostics for Sections 3.3, 3.5, and Supplementary
Sections S6 and S8.

Produces:
  - Table S4  : per-class F1, ResNet-18, by leave-one-dataset-out fold
                (from metrics.json's per_class_report; reference architecture
                used for the paired comparisons in Section 3.5).
  - Table S5  : per-class F1, all five architectures, averaged across the
                three leave-one-dataset-out folds.
  - Figure S3 : row-normalised confusion matrices, ResNet-18, all three
                LODO folds (see manuscript/fig_scripts/make_fig_confusion_matrices.py,
                which reads metrics.json's confusion_matrix field directly).
  - The dense_residential <-> medium_residential confusion-rate numbers
    (both directions) quoted in Section S6, and the chi-square test of
    independence confirming that pattern is not attributable to
    UCMerced's smaller held-out sample size for this class.
  - The agricultural_farmland <-> baseball_field confusion-rate numbers
    (both directions, RESISC45- and UCMerced-held-out folds) and the
    river -> beach confusion rate (UCMerced held out), quoted in Section
    S6's second paragraph.
  - The omnibus Cochran's Q test (Section 3.5) confirming per-sample
    correctness differs across the five architectures within each fold,
    justifying the pairwise McNemar comparisons that follow it.
  - Table S7  : cross-architecture error-agreement distribution (Section
    S8) - for each fold, what fraction of images were misclassified by
    0..5 of the five architectures - plus the per-class breakdown of the
    "wrong for all 5" subset quoted in Section S8's discussion.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import cochrans_q

RESULTS = r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results"
BASE_LODO = os.path.join(RESULTS, "studyA_lodo")

ARCHS = ["resnet18", "compactcnn", "convnext_tiny", "vit_small_patch16_224", "swin_tiny_patch4_window7_224"]
LABELS = {
    "resnet18": "ResNet-18", "compactcnn": "CompactCNN", "convnext_tiny": "ConvNeXt-Tiny",
    "vit_small_patch16_224": "ViT-Small/16", "swin_tiny_patch4_window7_224": "Swin-Tiny",
}
LODO_FOLDS = ["RESISC45", "UCMerced", "AID"]


def load_metrics(held_out, arch):
    with open(os.path.join(BASE_LODO, f"lodo_heldout-{held_out}_{arch}", "metrics.json")) as f:
        return json.load(f)


def load_predictions(held_out, arch):
    return pd.read_csv(os.path.join(BASE_LODO, f"lodo_heldout-{held_out}_{arch}", "predictions.csv"))


def table_s4_s5():
    print("=== Table S4: per-class F1, ResNet-18, by fold ===")
    data = {}
    classes = None
    for ds in LODO_FOLDS:
        m = load_metrics(ds, "resnet18")
        classes = m["classes"]
        for c in classes:
            data.setdefault(c, {})[ds] = m["per_class_report"][c]["f1-score"]
    for c in classes:
        vals = [data[c][ds] for ds in LODO_FOLDS]
        print(f"  {c:22s}", [round(v, 3) for v in vals], " mean=", round(sum(vals) / 3, 3))

    print()
    print("=== Table S5: mean per-class F1, all 5 architectures (averaged across the 3 LODO folds) ===")
    per_arch = {a: {} for a in ARCHS}
    for a in ARCHS:
        for ds in LODO_FOLDS:
            m = load_metrics(ds, a)
            for c in m["classes"]:
                per_arch[a].setdefault(c, []).append(m["per_class_report"][c]["f1-score"])
    for c in classes:
        row = [round(sum(per_arch[a][c]) / 3, 3) for a in ARCHS]
        overall_mean = round(sum(sum(per_arch[a][c]) / 3 for a in ARCHS) / 5, 3)
        print(f"  {c:22s}", dict(zip([LABELS[a] for a in ARCHS], row)), " mean=", overall_mean)


def dense_residential_confusion():
    print()
    print("=== Section S6: dense_residential <-> medium_residential confusion, by fold (ResNet-18) ===")
    print("(both directions checked: manuscript claims the forward direction dominates only for the")
    print(" UCMerced-held-out fold, and that RESISC45/AID show comparatively balanced confusion)")
    counts = {}
    for ds in LODO_FOLDS:
        m = load_metrics(ds, "resnet18")
        classes = m["classes"]
        cm = m["confusion_matrix"]
        i_dense = classes.index("dense_residential")
        i_med = classes.index("medium_residential")
        fwd_row = cm[i_dense]
        rev_row = cm[i_med]
        fwd_total = sum(fwd_row)
        rev_total = sum(rev_row)
        fwd_n = fwd_row[i_med]
        rev_n = rev_row[i_dense]
        counts[ds] = (fwd_n, fwd_total)
        print(f"  {ds:10s} dense->medium: {fwd_n}/{fwd_total} = {100*fwd_n/fwd_total:.1f}%   "
              f"medium->dense: {rev_n}/{rev_total} = {100*rev_n/rev_total:.1f}%")

    table = np.array([[counts[ds][0], counts[ds][1] - counts[ds][0]] for ds in LODO_FOLDS])
    chi2, p, dof, expected = chi2_contingency(table)
    print(f"  chi-square test of independence (fold x forward-direction misclassification rate): "
          f"chi2={chi2:.1f}, df={dof}, p={p:.3e}")


def farmland_baseball_and_river_beach_confusion():
    print()
    print("=== Section S6 (2nd paragraph): agricultural_farmland <-> baseball_field confusion, "
          "and river -> beach (ResNet-18) ===")
    for ds in LODO_FOLDS:
        m = load_metrics(ds, "resnet18")
        classes = m["classes"]
        cm = m["confusion_matrix"]
        i_ag = classes.index("agricultural_farmland")
        i_bb = classes.index("baseball_field")
        fwd_n = cm[i_ag][i_bb]; fwd_total = sum(cm[i_ag])
        rev_n = cm[i_bb][i_ag]; rev_total = sum(cm[i_bb])
        print(f"  {ds:10s} agricultural_farmland->baseball_field: {fwd_n}/{fwd_total} = {100*fwd_n/fwd_total:.1f}%   "
              f"baseball_field->agricultural_farmland: {rev_n}/{rev_total} = {100*rev_n/rev_total:.1f}%")

    m = load_metrics("UCMerced", "resnet18")
    classes = m["classes"]
    cm = m["confusion_matrix"]
    i_riv = classes.index("river"); i_beach = classes.index("beach")
    fwd_n = cm[i_riv][i_beach]; fwd_total = sum(cm[i_riv])
    print(f"  UCMerced   river->beach: {fwd_n}/{fwd_total} = {100*fwd_n/fwd_total:.1f}%")


def cochrans_q_omnibus():
    print()
    print("=== Section 3.5: omnibus Cochran's Q test, per LODO fold ===")
    for ds in LODO_FOLDS:
        mat_cols = []
        for a in ARCHS:
            df = load_predictions(ds, a).sort_values("filepath").reset_index(drop=True)
            mat_cols.append((df["true_label"] == df["pred_label"]).astype(int).values)
        lens = [len(c) for c in mat_cols]
        assert len(set(lens)) == 1, f"mismatched prediction counts across architectures for {ds}: {lens}"
        mat = np.column_stack(mat_cols)
        res = cochrans_q(mat)
        print(f"  {ds:10s} held out: Q={res.statistic:.1f}  df={mat.shape[1]-1}  p={res.pvalue:.3e}  n={mat.shape[0]}")


def table_s7_cross_arch_agreement():
    print()
    print("=== Table S7 / Section S8: cross-architecture error-agreement distribution ===")
    for ds in LODO_FOLDS:
        merged = None
        for a in ARCHS:
            df = load_predictions(ds, a)[["filepath", "true_label", "pred_label"]].copy()
            df[f"wrong_{a}"] = (df["true_label"] != df["pred_label"]).astype(int)
            keep = ["filepath", f"wrong_{a}"] if merged is not None else ["filepath", "true_label", f"wrong_{a}"]
            df = df[keep]
            merged = df if merged is None else merged.merge(df, on="filepath", how="inner")
        wrong_cols = [c for c in merged.columns if c.startswith("wrong_")]
        merged["n_wrong"] = merged[wrong_cols].sum(axis=1)
        total = len(merged)
        dist = merged["n_wrong"].value_counts().sort_index()
        print(f"  {ds} held out (n={total}):")
        for k in range(6):
            cnt = dist.get(k, 0)
            print(f"    wrong for {k} of 5: {cnt:5d}  ({100*cnt/total:.2f}%)")
        all_wrong = merged[merged["n_wrong"] == 5]
        print(f"    'wrong for all 5' by true class (n={len(all_wrong)}):")
        print("     ", dict(all_wrong["true_label"].value_counts().head(6)))


def main():
    table_s4_s5()
    dense_residential_confusion()
    farmland_baseball_and_river_beach_confusion()
    cochrans_q_omnibus()
    table_s7_cross_arch_agreement()


if __name__ == "__main__":
    main()

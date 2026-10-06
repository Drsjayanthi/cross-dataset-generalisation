"""
Stage 4: Statistical analysis over the completed experiment results.

1. Bootstrap 95% CIs (2000 resamples, stratified by true class) for accuracy / macro-F1 on
   every run's test-set predictions -- point estimates alone are not reported without
   uncertainty.
2. McNemar's exact test: resnet18 vs compactcnn on the SAME held-out test set, per LODO fold
   (paired design -- both models scored on identical images) -- is the architecture choice
   itself a significant factor in cross-dataset generalization?
3. Equivalence testing (TOST) for the central scientific question: is LODO (cross-dataset,
   fully unseen source) performance statistically EQUIVALENT to in-distribution performance?
   Equivalence margin is DATA-DRIVEN: 1 SD of the A2 pooled-5-fold in-distribution accuracy
   (i.e. the natural fold-to-fold variability you'd see even with no domain shift at all).
   Two one-sided bootstrap tests: if the 90% CI of (LODO_acc - pooled_mean_acc) lies entirely
   within [-margin, +margin], the datasets are equivalent at alpha=0.05; if it lies entirely
   outside, they are significantly different; otherwise the test is inconclusive at this
   sample size.
4. Holm-Bonferroni correction across the family of LODO-vs-pooled equivalence tests.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

RES_ROOT = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results")
OUT_DIR = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
N_BOOT = 2000


def bootstrap_ci(true, pred, metric_fn, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    true = np.asarray(true)
    pred = np.asarray(pred)
    n = len(true)
    point = metric_fn(true, pred)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = metric_fn(true[idx], pred[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, boots


def load_preds(run_dir):
    p = pd.read_csv(run_dir / "predictions.csv")
    return p["true_label"].values, p["pred_label"].values


def acc_fn(t, p):
    return accuracy_score(t, p)


def macrof1_fn(t, p):
    return f1_score(t, p, average="macro", zero_division=0)


report = {"bootstrap_ci": {}, "mcnemar": {}, "tost_equivalence": {}}

# ---------------- 1. Bootstrap CIs for every run ----------------
all_run_dirs = []
for sub in ["studyA_lodo", "studyA_pooled_cv", "studyA_native_cv", "studyB_eurosat"]:
    d = RES_ROOT / sub
    if d.exists():
        all_run_dirs += [p for p in d.iterdir() if p.is_dir() and (p / "predictions.csv").exists()]

boot_cache = {}  # run_tag -> (acc_point, acc_boots, f1_point, f1_boots, true, pred)
for run_dir in all_run_dirs:
    tag = run_dir.name
    true, pred = load_preds(run_dir)
    acc_p, acc_lo, acc_hi, acc_boots = bootstrap_ci(true, pred, acc_fn)
    f1_p, f1_lo, f1_hi, f1_boots = bootstrap_ci(true, pred, macrof1_fn)
    boot_cache[tag] = dict(true=true, pred=pred, acc_boots=acc_boots, f1_boots=f1_boots)
    report["bootstrap_ci"][tag] = {
        "n_test": len(true),
        "accuracy": {"point": acc_p, "ci95_lo": acc_lo, "ci95_hi": acc_hi},
        "macro_f1": {"point": f1_p, "ci95_lo": f1_lo, "ci95_hi": f1_hi},
    }
    print(f"{tag:45s} acc={acc_p:.4f} [{acc_lo:.4f},{acc_hi:.4f}]  "
          f"macroF1={f1_p:.4f} [{f1_lo:.4f},{f1_hi:.4f}]")

# ---------------- 2. McNemar: every architecture vs the resnet18 baseline, per LODO fold ----------------
lodo_dir = RES_ROOT / "studyA_lodo"
CHALLENGER_ARCHS = ["compactcnn", "convnext_tiny", "vit_small_patch16_224",
                    "swin_tiny_patch4_window7_224"]
if lodo_dir.exists():
    for held_out in ["RESISC45", "UCMerced", "AID"]:
        r18_dir = lodo_dir / f"lodo_heldout-{held_out}_resnet18"
        if not r18_dir.exists():
            continue
        p18 = pd.read_csv(r18_dir / "predictions.csv").sort_values("filepath").reset_index(drop=True)
        correct18 = (p18["true_label"] == p18["pred_label"]).values
        for arch in CHALLENGER_ARCHS:
            chal_dir = lodo_dir / f"lodo_heldout-{held_out}_{arch}"
            if not chal_dir.exists():
                continue
            pchal = pd.read_csv(chal_dir / "predictions.csv").sort_values("filepath").reset_index(drop=True)
            assert (p18["filepath"].values == pchal["filepath"].values).all(), \
                "test sets must match for McNemar"
            correct_chal = (pchal["true_label"] == pchal["pred_label"]).values
            # 2x2 contingency: [both correct, r18 correct/chal wrong; r18 wrong/chal correct, both wrong]
            both_c = int(np.sum(correct18 & correct_chal))
            r18_only = int(np.sum(correct18 & ~correct_chal))
            chal_only = int(np.sum(~correct18 & correct_chal))
            both_w = int(np.sum(~correct18 & ~correct_chal))
            table = [[both_c, r18_only], [chal_only, both_w]]
            res = mcnemar(table, exact=(r18_only + chal_only < 25), correction=True)
            key = f"{held_out}__resnet18_vs_{arch}"
            report["mcnemar"][key] = {
                "held_out": held_out, "challenger": arch,
                "table_both_correct": both_c, "resnet18_only_correct": r18_only,
                "challenger_only_correct": chal_only, "both_wrong": both_w,
                "statistic": float(res.statistic), "p_value": float(res.pvalue),
            }
            print(f"McNemar (held_out={held_out}, resnet18 vs {arch}): "
                  f"resnet18-only={r18_only}, {arch}-only={chal_only}, p={res.pvalue:.4g}")

# Holm-Bonferroni correction across the full family of McNemar comparisons (Table 7),
# for consistency with the correction applied to the TOST equivalence tests below.
mcnemar_keys = list(report["mcnemar"].keys())
if mcnemar_keys:
    mcnemar_pvals = [report["mcnemar"][k]["p_value"] for k in mcnemar_keys]
    rej, p_adj, _, _ = multipletests(mcnemar_pvals, alpha=0.05, method="holm")
    for k, padj, r in zip(mcnemar_keys, p_adj, rej):
        report["mcnemar"][k]["holm_adjusted_p"] = float(padj)
        report["mcnemar"][k]["significant_after_correction"] = bool(r)
        print(f"  Holm-adjusted p for {k}: {padj:.4g} (significant: {r})")

# ---------------- 3. TOST equivalence: LODO vs in-distribution pooled, per architecture ----------------
# Every architecture that has its own pooled 5-fold reference gets its own equivalence test
# (margin is that architecture's own fold-to-fold SD -- CompactCNN has no pooled run, so it's
# skipped here; the McNemar comparisons above already cover it).
pooled_dir = RES_ROOT / "studyA_pooled_cv"
POOLED_ARCHS = {
    "resnet18": "pooled_fold[0-4]",  # original run: no arch suffix in the tag
    "compactcnn": "pooled_fold*_compactcnn",
    "convnext_tiny": "pooled_fold*_convnext_tiny",
    "vit_small_patch16_224": "pooled_fold*_vit_small_patch16_224",
    "swin_tiny_patch4_window7_224": "pooled_fold*_swin_tiny_patch4_window7_224",
}
if pooled_dir.exists() and lodo_dir.exists():
    pvals, keys = [], []
    for arch, glob_pat in POOLED_ARCHS.items():
        run_dirs = sorted(pooled_dir.glob(glob_pat))
        if not run_dirs:
            continue
        pooled_accs = []
        for run_dir in run_dirs:
            with open(run_dir / "metrics.json") as f:
                pooled_accs.append(json.load(f)["test_accuracy"])
        pooled_accs = np.array(pooled_accs)
        pooled_mean, pooled_sd = pooled_accs.mean(), pooled_accs.std(ddof=1)
        margin = pooled_sd  # data-driven equivalence margin: 1 SD of in-distribution fold variability
        print(f"\n[{arch}] in-distribution pooled {len(run_dirs)}-fold accuracy: "
              f"mean={pooled_mean:.4f} sd={pooled_sd:.4f} -> equivalence margin = +/-{margin:.4f}")

        for held_out in ["RESISC45", "UCMerced", "AID"]:
            tag = f"lodo_heldout-{held_out}_{arch}"
            if tag not in boot_cache:
                continue
            acc_boots = boot_cache[tag]["acc_boots"]
            diff_boots = acc_boots - pooled_mean  # bootstrap distribution of (LODO - in-distribution)
            lo90, hi90 = np.percentile(diff_boots, [5, 95])  # 90% CI -> two one-sided tests @ alpha=0.05
            point_diff = acc_boots.mean() - pooled_mean
            equivalent = (lo90 > -margin) and (hi90 < margin)
            significant_diff = (lo90 > margin) or (hi90 < -margin)
            verdict = ("EQUIVALENT" if equivalent else
                       "SIGNIFICANTLY DIFFERENT" if significant_diff else "INCONCLUSIVE")
            p_low = float(np.mean(diff_boots < -margin))
            p_high = float(np.mean(diff_boots > margin))
            p_tost = max(p_low, p_high)  # TOST p-value = max of the two one-sided p-values
            key = f"{arch}__{held_out}"
            report["tost_equivalence"][key] = {
                "arch": arch, "held_out": held_out,
                "lodo_accuracy": float(acc_boots.mean()),
                "in_distribution_pooled_mean_accuracy": float(pooled_mean),
                "equivalence_margin": float(margin),
                "diff_90ci": [float(lo90), float(hi90)],
                "verdict": verdict,
                "tost_p_value_uncorrected": p_tost,
            }
            pvals.append(p_tost)
            keys.append(key)
            print(f"TOST [{arch}] held_out={held_out}: diff={point_diff:+.4f}, "
                  f"90% CI=[{lo90:+.4f},{hi90:+.4f}], margin=+/-{margin:.4f} -> {verdict} (p={p_tost:.4g})")

    if pvals:
        rej, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
        for k, padj, r in zip(keys, p_adj, rej):
            report["tost_equivalence"][k]["holm_adjusted_p"] = float(padj)
            report["tost_equivalence"][k]["equivalence_significant_after_correction"] = bool(r)
            print(f"  Holm-adjusted p for {k}: {padj:.4g} (equivalence significant: {r})")

with open(OUT_DIR / "statistical_analysis.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nFull statistical report -> {OUT_DIR / 'statistical_analysis.json'}")

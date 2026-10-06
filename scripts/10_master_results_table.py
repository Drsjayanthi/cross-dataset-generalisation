"""
Stage 10: Master per-run results table for Supplementary Table S6.

Parses reports/all_results_with_ci.csv (produced by 04_stats_analysis.py)
into the structured (Design, Detail, Architecture) format used in Table S6,
underlying every summary statistic reported in the main text and in
Supplementary Tables S1-S3.

run_tag naming conventions produced by the driver scripts:
  lodo_heldout-{DATASET}[_{arch}]   -> Design A1 (unsuffixed run_tag = resnet18)
  pooled_fold{i}[_{arch}]           -> Design A2
  {DATASET}_fold{i}[_{arch}]        -> Design A3
  cv_fold{i}_{arch}                 -> Design B1 (always suffixed)
  final_helodtest_{arch}            -> Design B2 (always suffixed)
"""
import re
import pandas as pd

REPORT = r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\reports\all_results_with_ci.csv"

ARCH_LABELS = {
    "resnet18": "ResNet-18", "compactcnn": "CompactCNN", "convnext_tiny": "ConvNeXt-Tiny",
    "vit_small_patch16_224": "ViT-Small/16", "swin_tiny_patch4_window7_224": "Swin-Tiny",
}
# longest first, so e.g. "swin_tiny_patch4_window7_224" matches before any shorter false-prefix could
ARCH_SUFFIXES = sorted(ARCH_LABELS.keys(), key=len, reverse=True)

DESIGN_ORDER = {"A1 (LODO)": 0, "A2 (Pooled CV)": 1, "A3 (Native-taxonomy CV)": 2,
                "B1 (EuroSAT CV)": 3, "B2 (EuroSAT final test)": 4}
ARCH_ORDER = {"ResNet-18": 0, "ConvNeXt-Tiny": 1, "ViT-Small/16": 2, "Swin-Tiny": 3, "CompactCNN": 4}


def parse_run_tag(tag):
    arch = "resnet18"
    base = tag
    for suf in ARCH_SUFFIXES:
        if tag == suf or tag.endswith("_" + suf):
            arch = suf
            base = tag[: -(len(suf) + 1)] if tag.endswith("_" + suf) else ""
            break

    if base.startswith("lodo_heldout-"):
        ds = base[len("lodo_heldout-"):]
        return "A1 (LODO)", f"{ds} held out", arch
    if base.startswith("pooled_fold"):
        i = base[len("pooled_fold"):]
        return "A2 (Pooled CV)", f"Fold {i}", arch
    if base.startswith("cv_fold"):
        i = base[len("cv_fold"):]
        return "B1 (EuroSAT CV)", f"Fold {i}", arch
    if base.startswith("final_helodtest"):
        return "B2 (EuroSAT final test)", "Final test", arch
    m = re.match(r"(RESISC45|UCMerced|AID)_fold(\d+)", base)
    if m:
        return "A3 (Native-taxonomy CV)", f"{m.group(1)}, Fold {m.group(2)}", arch
    raise ValueError(f"Unrecognised run_tag pattern: {tag!r}")


def main():
    df = pd.read_csv(REPORT)
    rows = []
    for _, r in df.iterrows():
        design, detail, arch = parse_run_tag(r["run_tag"])
        rows.append((design, detail, ARCH_LABELS[arch],
                      r["accuracy"], r["acc_ci95_lo"], r["acc_ci95_hi"],
                      r["macro_f1"], r["f1_ci95_lo"], r["f1_ci95_hi"]))

    rows.sort(key=lambda x: (DESIGN_ORDER.get(x[0], 9), x[1], ARCH_ORDER.get(x[2], 9)))

    print(f"Total rows: {len(rows)} (expected 145: A1=15, A2=25, A3=75, B1=25, B2=5)")
    print()
    print(f"{'Design':<25s} {'Detail':<20s} {'Architecture':<15s} {'Accuracy':>9s} {'95% CI (Acc.)':>15s} "
          f"{'Macro-F1':>9s} {'95% CI (F1)':>15s}")
    for design, detail, arch, acc, alo, ahi, f1, flo, fhi in rows:
        acc_s = f"{acc*100:.2f}%"
        ci_acc_s = f"{alo*100:.1f}-{ahi*100:.1f}%"
        f1_s = f"{f1:.3f}"
        ci_f1_s = f"{flo:.3f}-{fhi:.3f}"
        print(f"{design:<25s} {detail:<20s} {arch:<15s} {acc_s:>9s} {ci_acc_s:>15s} {f1_s:>9s} {ci_f1_s:>15s}")

    # sanity check: per-design row counts should be 15/25/75/25/5
    counts = {}
    for design, *_ in rows:
        counts[design] = counts.get(design, 0) + 1
    print()
    print("Per-design row counts:", counts)


if __name__ == "__main__":
    main()

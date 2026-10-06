"""
Stage 6: Extend the study with modern architectures spanning different paradigms --
does the cross-dataset generalization gap found with ResNet-18 hold up, or is it an
artifact of one model family?

  convnext_tiny                 pure convolutional (modern, 2022-era design)
  vit_small_patch16_224         pure global self-attention (no conv locality prior)
  swin_tiny_patch4_window7_224  hierarchical windowed self-attention

Scope per architecture (proportionate to the question -- generalization, not re-deriving
already-established ceilings):
  - Study A / LODO, all 3 folds (the core generalization test)
  - Study A / pooled 5-fold (in-distribution reference, needed for TOST)
  - Study B / EuroSAT final held-out test only (one comparable number, not a full 5-fold CV)
Native per-dataset 5-fold CV is NOT re-run per architecture -- that ceiling is already
established (ResNet-18) and re-deriving it wouldn't change the generalization conclusion.
"""
import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_train import run_experiment

MAN_DIR = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\manifests")
RES_LODO = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results\studyA_lodo")
RES_POOLED = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results\studyA_pooled_cv")
RES_EUROSAT = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results\studyB_eurosat")
for d in (RES_LODO, RES_POOLED, RES_EUROSAT):
    d.mkdir(parents=True, exist_ok=True)

NEW_ARCHS = {
    "convnext_tiny": dict(img_size=224, max_epochs=25, patience=4, batch_size=64),
    "vit_small_patch16_224": dict(img_size=224, max_epochs=25, patience=4, batch_size=64),
    "swin_tiny_patch4_window7_224": dict(img_size=224, max_epochs=25, patience=4, batch_size=48),
}


def main():
    all_new_results = {"lodo": [], "pooled": [], "eurosat_final": []}

    # ---------------- LODO ----------------
    lodo = pd.read_csv(MAN_DIR / "splits" / "studyA_lodo_splits.csv")
    for held_out in ["RESISC45", "UCMerced", "AID"]:
        fold = lodo[lodo["lodo_fold"] == held_out]
        train = fold[fold["lodo_split"] == "train"].reset_index(drop=True)
        test = fold[fold["lodo_split"] == "test"].reset_index(drop=True)
        for arch, cfg in NEW_ARCHS.items():
            run_tag = f"lodo_heldout-{held_out}_{arch}"
            out_path = RES_LODO / run_tag
            print(f"\n=== LODO {run_tag} === train={len(train)} test={len(test)}")
            m = run_experiment(train, test, "harmonized_label", arch, out_path,
                                run_tag=run_tag, **cfg)
            m["held_out_dataset"] = held_out
            all_new_results["lodo"].append(m)
            with open(RES_LODO / "_extended_summary_so_far.json", "w") as f:
                json.dump(all_new_results, f, indent=2)

    # ---------------- Pooled 5-fold ----------------
    pooled = pd.read_csv(MAN_DIR / "splits" / "studyA_pooled_5fold.csv")
    for arch, cfg in NEW_ARCHS.items():
        for fold_i in sorted(pooled["pooled_fold"].unique()):
            train = pooled[pooled["pooled_fold"] != fold_i].reset_index(drop=True)
            test = pooled[pooled["pooled_fold"] == fold_i].reset_index(drop=True)
            run_tag = f"pooled_fold{fold_i}_{arch}"
            out_path = RES_POOLED / run_tag
            print(f"\n=== POOLED {run_tag} === train={len(train)} test={len(test)}")
            m = run_experiment(train, test, "harmonized_label", arch, out_path,
                                run_tag=run_tag, **cfg)
            m["fold"] = int(fold_i)
            all_new_results["pooled"].append(m)
            with open(RES_POOLED / "_extended_summary_so_far.json", "w") as f:
                json.dump(all_new_results, f, indent=2)

    # ---------------- EuroSAT final held-out test ----------------
    eu = pd.read_csv(MAN_DIR / "splits" / "studyB_splits.csv")
    trainval = eu[eu["studyB_split"] == "trainval"].reset_index(drop=True)
    held_out_test = eu[eu["studyB_split"] == "held_out_test"].reset_index(drop=True)
    for arch, cfg in NEW_ARCHS.items():
        run_tag = f"final_helodtest_{arch}"
        out_path = RES_EUROSAT / run_tag
        print(f"\n=== EUROSAT {run_tag} === train={len(trainval)} test={len(held_out_test)}")
        m = run_experiment(trainval, held_out_test, "native_label", arch, out_path,
                            run_tag=run_tag, **cfg)
        all_new_results["eurosat_final"].append(m)
        with open(RES_EUROSAT / "_extended_summary_so_far.json", "w") as f:
            json.dump(all_new_results, f, indent=2)

    print("\n\n=== EXTENDED ARCHITECTURE SUMMARY ===")
    for m in all_new_results["lodo"]:
        print(f"LODO    {m['held_out_dataset']:>10s} | {m['arch']:>28s} | "
              f"acc={m['test_accuracy']:.4f} macroF1={m['test_macro_f1']:.4f}")
    for m in all_new_results["pooled"]:
        print(f"POOLED  fold{m['fold']} | {m['arch']:>28s} | "
              f"acc={m['test_accuracy']:.4f} macroF1={m['test_macro_f1']:.4f}")
    for m in all_new_results["eurosat_final"]:
        print(f"EUROSAT | {m['arch']:>28s} | acc={m['test_accuracy']:.4f} macroF1={m['test_macro_f1']:.4f}")

    with open(RES_LODO.parent / "extended_architectures_summary.json", "w") as f:
        json.dump(all_new_results, f, indent=2)
    print(f"\nSaved -> {RES_LODO.parent / 'extended_architectures_summary.json'}")


if __name__ == "__main__":
    main()

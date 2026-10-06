"""
Stage 3c: Study B -- EuroSAT-RGB, standalone (not pooled with Study A; see dataset-validity
rationale in reports/). Native 10-class taxonomy.

5-fold group-stratified CV on the trainval portion (both architectures, for the same
pretrained-vs-scratch robustness check as Study A), then a single final fit on the FULL
trainval set evaluated once on the untouched held_out_test split (20%, set aside in
02_make_splits.py before any of this ran) -- the honest "final external-style number" for
this dataset, even though (as noted in the design doc) it is a same-distribution held-out
test rather than a second independent sensor.

Images are native 64x64; resnet18 input is upsampled to 128px (moderate, not to 224 --
avoids overstating information content that the sensor doesn't have).
"""
import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_train import run_experiment

MAN_DIR = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\manifests")
RES_DIR = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results\studyB_eurosat")
RES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(MAN_DIR / "splits" / "studyB_splits.csv")
    trainval = df[df["studyB_split"] == "trainval"]
    held_out_test = df[df["studyB_split"] == "held_out_test"].reset_index(drop=True)

    ARCH_CFG = {
        "resnet18": dict(img_size=128, max_epochs=25, patience=4, batch_size=128),
        "compactcnn": dict(img_size=64, max_epochs=35, patience=6, batch_size=128),
    }

    # ---------------- B1: 5-fold CV within trainval ----------------
    cv_results = []
    for fold_i in sorted(trainval["cv_fold"].dropna().unique()):
        fold_i = int(fold_i)
        train = trainval[trainval["cv_fold"] != fold_i].reset_index(drop=True)
        val_test = trainval[trainval["cv_fold"] == fold_i].reset_index(drop=True)
        for arch, cfg in ARCH_CFG.items():
            run_tag = f"cv_fold{fold_i}_{arch}"
            out_path = RES_DIR / run_tag
            print(f"\n=== B1 {run_tag} === train={len(train)} test={len(val_test)}")
            m = run_experiment(train, val_test, "native_label", arch, out_path,
                                run_tag=run_tag, **cfg)
            m["fold"] = fold_i
            cv_results.append(m)
            with open(RES_DIR / "_cv_summary_so_far.json", "w") as f:
                json.dump(cv_results, f, indent=2)
    with open(RES_DIR / "cv_summary.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    print("\n=== B1 EUROSAT 5-FOLD CV SUMMARY ===")
    for m in cv_results:
        print(f"fold {m['fold']} | {m['arch']:>10s} | acc={m['test_accuracy']:.4f} macroF1={m['test_macro_f1']:.4f}")

    # ---------------- B2: final fit on full trainval -> held_out_test ----------------
    final_results = []
    for arch, cfg in ARCH_CFG.items():
        run_tag = f"final_helodtest_{arch}"
        out_path = RES_DIR / run_tag
        print(f"\n=== B2 {run_tag} === train={len(trainval)} test={len(held_out_test)}")
        m = run_experiment(trainval.reset_index(drop=True), held_out_test, "native_label", arch,
                            out_path, run_tag=run_tag, **cfg)
        final_results.append(m)
    with open(RES_DIR / "final_test_summary.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print("\n=== B2 EUROSAT FINAL HELD-OUT TEST SUMMARY ===")
    for m in final_results:
        print(f"{m['arch']:>10s} | acc={m['test_accuracy']:.4f} macroF1={m['test_macro_f1']:.4f}")

    print("\nStudy B complete.")


if __name__ == "__main__":
    main()

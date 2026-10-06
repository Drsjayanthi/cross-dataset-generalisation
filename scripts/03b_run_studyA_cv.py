"""
Stage 3b: Study A / A2 + A3 -- internal-validity CV references.

A2: pooled group-stratified 5-fold CV over the harmonized 12-class core (RESISC45+UCMerced+AID
    mixed together). Upper-bound reference: what's achievable when train and test come from the
    SAME mixture of sources (contrast against A1/LODO, where test is a fully unseen source).
A3: per-dataset native-taxonomy group-stratified 5-fold CV. Ceiling for each dataset alone,
    on its own full label set (45/21/30 classes) -- not just the 12-class harmonized subset.

Architecture: resnet18 only (the LODO stage already establishes the resnet18-vs-compactcnn
generalization comparison; repeating both archs here would not change the external-validity
conclusion and roughly doubles compute for a secondary/reference analysis).
"""
import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_train import run_experiment

MAN_DIR = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\manifests")
RES_DIR_A2 = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results\studyA_pooled_cv")
RES_DIR_A3 = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results\studyA_native_cv")
RES_DIR_A2.mkdir(parents=True, exist_ok=True)
RES_DIR_A3.mkdir(parents=True, exist_ok=True)

CFG = dict(arch="resnet18", img_size=224, max_epochs=25, patience=4, batch_size=64)


def main():
    # ---------------- A2: pooled 5-fold ----------------
    pooled = pd.read_csv(MAN_DIR / "splits" / "studyA_pooled_5fold.csv")
    a2_results = []
    for fold_i in sorted(pooled["pooled_fold"].unique()):
        train = pooled[pooled["pooled_fold"] != fold_i].reset_index(drop=True)
        test = pooled[pooled["pooled_fold"] == fold_i].reset_index(drop=True)
        run_tag = f"pooled_fold{fold_i}"
        out_path = RES_DIR_A2 / run_tag
        print(f"\n=== A2 {run_tag} === train={len(train)} test={len(test)}")
        m = run_experiment(train, test, "harmonized_label", CFG["arch"], out_path,
                            run_tag=run_tag, img_size=CFG["img_size"], max_epochs=CFG["max_epochs"],
                            patience=CFG["patience"], batch_size=CFG["batch_size"])
        m["fold"] = int(fold_i)
        a2_results.append(m)
        with open(RES_DIR_A2 / "_summary_so_far.json", "w") as f:
            json.dump(a2_results, f, indent=2)
    with open(RES_DIR_A2 / "summary.json", "w") as f:
        json.dump(a2_results, f, indent=2)
    print("\n=== A2 POOLED 5-FOLD SUMMARY ===")
    for m in a2_results:
        print(f"fold {m['fold']} | acc={m['test_accuracy']:.4f} macroF1={m['test_macro_f1']:.4f}")

    # ---------------- A3: per-dataset native 5-fold ----------------
    native = pd.read_csv(MAN_DIR / "splits" / "studyA_native_5fold.csv")
    a3_results = []
    for ds_name in ["RESISC45", "UCMerced", "AID"]:
        ds_df = native[native["dataset"] == ds_name]
        for fold_i in sorted(ds_df["native_fold"].unique()):
            train = ds_df[ds_df["native_fold"] != fold_i].reset_index(drop=True)
            test = ds_df[ds_df["native_fold"] == fold_i].reset_index(drop=True)
            run_tag = f"{ds_name}_fold{fold_i}"
            out_path = RES_DIR_A3 / run_tag
            print(f"\n=== A3 {run_tag} === train={len(train)} test={len(test)}")
            m = run_experiment(train, test, "native_label", CFG["arch"], out_path,
                                run_tag=run_tag, img_size=CFG["img_size"], max_epochs=CFG["max_epochs"],
                                patience=CFG["patience"], batch_size=CFG["batch_size"])
            m["dataset"] = ds_name
            m["fold"] = int(fold_i)
            a3_results.append(m)
            with open(RES_DIR_A3 / "_summary_so_far.json", "w") as f:
                json.dump(a3_results, f, indent=2)
    with open(RES_DIR_A3 / "summary.json", "w") as f:
        json.dump(a3_results, f, indent=2)
    print("\n=== A3 NATIVE 5-FOLD SUMMARY ===")
    for m in a3_results:
        print(f"{m['dataset']:>10s} fold {m['fold']} | acc={m['test_accuracy']:.4f} macroF1={m['test_macro_f1']:.4f}")

    print("\nAll Study A CV experiments complete.")


if __name__ == "__main__":
    main()

"""
Stage 7: Close every remaining coverage gap so all 5 architectures (resnet18, compactcnn,
convnext_tiny, vit_small_patch16_224, swin_tiny_patch4_window7_224) have run through every
experiment block in the study:

  (a) Pooled 5-fold CV for compactcnn                      -- 5 runs
  (b) EuroSAT 5-fold CV for convnext/vit/swin               -- 15 runs
  (c) Native per-dataset 5-fold CV for compactcnn + the 3
      extended architectures (RESISC45/UCMerced/AID)        -- 60 runs
Total: 80 runs. Run in that order (fastest wins first) so partial progress is maximally useful
if interrupted.
"""
import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_train import run_experiment

MAN_DIR = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\manifests")
RES_POOLED = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results\studyA_pooled_cv")
RES_NATIVE = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results\studyA_native_cv")
RES_EUROSAT = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results\studyB_eurosat")
for d in (RES_POOLED, RES_NATIVE, RES_EUROSAT):
    d.mkdir(parents=True, exist_ok=True)

PRETRAINED_CFG = dict(img_size=224, max_epochs=25, patience=4, batch_size=64)
SWIN_CFG = dict(img_size=224, max_epochs=25, patience=4, batch_size=48)
COMPACTCNN_CFG_A = dict(img_size=128, max_epochs=35, patience=6, batch_size=64)   # Study A scale
COMPACTCNN_CFG_B = dict(img_size=64, max_epochs=35, patience=6, batch_size=128)   # EuroSAT scale


def cfg_for(arch):
    if arch == "compactcnn":
        return COMPACTCNN_CFG_A
    if arch == "swin_tiny_patch4_window7_224":
        return SWIN_CFG
    return PRETRAINED_CFG


def main():
    # ---------------- (a) pooled 5-fold CV, compactcnn ----------------
    pooled = pd.read_csv(MAN_DIR / "splits" / "studyA_pooled_5fold.csv")
    for fold_i in sorted(pooled["pooled_fold"].unique()):
        train = pooled[pooled["pooled_fold"] != fold_i].reset_index(drop=True)
        test = pooled[pooled["pooled_fold"] == fold_i].reset_index(drop=True)
        run_tag = f"pooled_fold{fold_i}_compactcnn"
        out_path = RES_POOLED / run_tag
        print(f"\n=== (a) {run_tag} === train={len(train)} test={len(test)}")
        run_experiment(train, test, "harmonized_label", "compactcnn", out_path,
                        run_tag=run_tag, **COMPACTCNN_CFG_A)

    # ---------------- (b) EuroSAT 5-fold CV, convnext/vit/swin ----------------
    eu = pd.read_csv(MAN_DIR / "splits" / "studyB_splits.csv")
    trainval = eu[eu["studyB_split"] == "trainval"]
    for arch in ["convnext_tiny", "vit_small_patch16_224", "swin_tiny_patch4_window7_224"]:
        # reuse the exact same config already used for these archs' EuroSAT final-test run
        # (224px) -- otherwise the CV-fold numbers wouldn't be comparable to that number
        cfg = cfg_for(arch)
        for fold_i in sorted(trainval["cv_fold"].dropna().unique()):
            fold_i = int(fold_i)
            train = trainval[trainval["cv_fold"] != fold_i].reset_index(drop=True)
            test = trainval[trainval["cv_fold"] == fold_i].reset_index(drop=True)
            run_tag = f"cv_fold{fold_i}_{arch}"
            out_path = RES_EUROSAT / run_tag
            print(f"\n=== (b) {run_tag} === train={len(train)} test={len(test)}")
            run_experiment(train, test, "native_label", arch, out_path, run_tag=run_tag, **cfg)

    # ---------------- (c) native per-dataset 5-fold CV, compactcnn + 3 extended archs ----------------
    native = pd.read_csv(MAN_DIR / "splits" / "studyA_native_5fold.csv")
    for arch in ["compactcnn", "convnext_tiny", "vit_small_patch16_224",
                 "swin_tiny_patch4_window7_224"]:
        cfg = cfg_for(arch)
        for ds_name in ["RESISC45", "UCMerced", "AID"]:
            ds_df = native[native["dataset"] == ds_name]
            for fold_i in sorted(ds_df["native_fold"].unique()):
                train = ds_df[ds_df["native_fold"] != fold_i].reset_index(drop=True)
                test = ds_df[ds_df["native_fold"] == fold_i].reset_index(drop=True)
                run_tag = f"{ds_name}_fold{fold_i}_{arch}"
                out_path = RES_NATIVE / run_tag
                print(f"\n=== (c) {run_tag} === train={len(train)} test={len(test)}")
                run_experiment(train, test, "native_label", arch, out_path, run_tag=run_tag, **cfg)

    print("\n\nFull-parity stage complete.")


if __name__ == "__main__":
    main()

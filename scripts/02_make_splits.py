"""
Step 2: Build leakage-safe splits for Study A and Study B.

Leakage control: grouping key = MD5 hash (exact-duplicate images, even if they sit in
different rows, are forced into the same split). Perceptual-hash near-duplicates were
checked manually (01_build_manifests_and_qc.py output) and found to be false positives
at this resolution, so phash is NOT used as a grouping key.

Study A (RESISC45 + UCMerced + AID, harmonized 12-class core):
  A1. LODO (Leave-One-Dataset-Out): 3 folds. Train on the pooled union of two datasets,
      test on the third, held out in full. This is the true external-generalization test.
  A2. Pooled group-stratified 5-fold CV (stratify on dataset x harmonized_label) as the
      internal-validity reference (upper bound on what pooling can achieve when the test
      set is drawn from the same mixture as train).
  A3. Per-dataset (native full taxonomy) group-stratified 5-fold CV, as a baseline ceiling
      for each dataset in isolation.

Study B (EuroSAT-RGB, native 10-class taxonomy):
  B1. Group-stratified 5-fold CV + a final held-out test split (20%), held out before any
      model selection.

All split assignments are written as CSVs of (filepath, fold assignment) so the training
script never has to re-derive them (reproducibility).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

RNG_SEED = 42
MAN_DIR = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\manifests")
OUT_DIR = MAN_DIR / "splits"
OUT_DIR.mkdir(exist_ok=True)


def assign_dup_group(df):
    """Group id = md5 hash, so exact duplicates always land in the same split."""
    return df["md5"]


# ---------------------------------------------------------------------------
# Study A
# ---------------------------------------------------------------------------
study_a = pd.read_csv(MAN_DIR / "harmonized_core_manifest.csv")
print(f"Study A pool: {len(study_a)} images, datasets={study_a['dataset'].unique().tolist()}, "
      f"classes={sorted(study_a['harmonized_label'].unique())}")

# --- A1: LODO ---
lodo_assignments = []
for held_out in ["RESISC45", "UCMerced", "AID"]:
    fold_df = study_a.copy()
    fold_df["lodo_fold"] = held_out
    fold_df["lodo_split"] = np.where(fold_df["dataset"] == held_out, "test", "train")
    lodo_assignments.append(fold_df[["filepath", "dataset", "harmonized_label", "md5",
                                      "lodo_fold", "lodo_split"]])
lodo_df = pd.concat(lodo_assignments, ignore_index=True)
lodo_df.to_csv(OUT_DIR / "studyA_lodo_splits.csv", index=False)
for held_out in ["RESISC45", "UCMerced", "AID"]:
    sub = lodo_df[lodo_df["lodo_fold"] == held_out]
    n_train = (sub["lodo_split"] == "train").sum()
    n_test = (sub["lodo_split"] == "test").sum()
    print(f"  [A1-LODO] held-out={held_out}: train={n_train}, test={n_test}")

# --- A2: pooled group-stratified 5-fold (stratify key = dataset|class) ---
study_a = study_a.copy()
study_a["strat_key"] = study_a["dataset"] + "|" + study_a["harmonized_label"]
groups = assign_dup_group(study_a)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
study_a["pooled_fold"] = -1
for fold_i, (_, test_idx) in enumerate(sgkf.split(study_a, study_a["strat_key"], groups)):
    study_a.loc[study_a.index[test_idx], "pooled_fold"] = fold_i
study_a[["filepath", "dataset", "harmonized_label", "md5", "pooled_fold"]].to_csv(
    OUT_DIR / "studyA_pooled_5fold.csv", index=False)
print("  [A2-pooled] fold sizes:", study_a["pooled_fold"].value_counts().sort_index().to_dict())

# --- A3: per-dataset native-taxonomy group-stratified 5-fold ---
native_frames = []
for ds_name, root_manifest in [("RESISC45", "RESISC45_manifest.csv"),
                                ("UCMerced", "UCMerced_manifest.csv"),
                                ("AID", "AID_manifest.csv")]:
    df = pd.read_csv(MAN_DIR / root_manifest)
    df = df[~df["corrupt"]].copy()
    groups = assign_dup_group(df)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    df["native_fold"] = -1
    for fold_i, (_, test_idx) in enumerate(sgkf.split(df, df["native_label"], groups)):
        df.loc[df.index[test_idx], "native_fold"] = fold_i
    df["dataset"] = ds_name
    native_frames.append(df[["filepath", "dataset", "native_label", "md5", "native_fold"]])
    print(f"  [A3-native] {ds_name} fold sizes:",
          df["native_fold"].value_counts().sort_index().to_dict())
native_df = pd.concat(native_frames, ignore_index=True)
native_df.to_csv(OUT_DIR / "studyA_native_5fold.csv", index=False)

# ---------------------------------------------------------------------------
# Study B: EuroSAT
# ---------------------------------------------------------------------------
eurosat = pd.read_csv(MAN_DIR / "eurosat_manifest.csv")
groups = assign_dup_group(eurosat)

# held-out final test (20%), group-aware
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RNG_SEED)
trainval_idx, test_idx = next(gss.split(eurosat, eurosat["native_label"], groups))
eurosat["studyB_split"] = "trainval"
eurosat.loc[eurosat.index[test_idx], "studyB_split"] = "held_out_test"

# 5-fold CV within the trainval portion only
trainval = eurosat[eurosat["studyB_split"] == "trainval"].copy()
groups_tv = assign_dup_group(trainval)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
trainval["cv_fold"] = -1
for fold_i, (_, tidx) in enumerate(sgkf.split(trainval, trainval["native_label"], groups_tv)):
    trainval.loc[trainval.index[tidx], "cv_fold"] = fold_i
eurosat = eurosat.merge(trainval[["filepath", "cv_fold"]], on="filepath", how="left")
eurosat.to_csv(OUT_DIR / "studyB_splits.csv", index=False)
print(f"  [B1] held_out_test={sum(eurosat['studyB_split']=='held_out_test')}, "
      f"trainval={sum(eurosat['studyB_split']=='trainval')}")
print("  [B1] cv fold sizes (within trainval):",
      trainval["cv_fold"].value_counts().sort_index().to_dict())

print("\nAll split files written to", OUT_DIR)

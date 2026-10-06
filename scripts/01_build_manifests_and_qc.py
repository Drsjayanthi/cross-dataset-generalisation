"""
Step 1: Build per-image manifests + run genuine quality checks for all four datasets.

For every image we verify (not assume):
  - file actually opens as an image (catches truncated/corrupt files)
  - pixel dimensions and color mode
  - exact-duplicate detection via MD5 (within each dataset)
  - near-duplicate detection via perceptual hash (within each dataset) -- flags likely
    train/test leakage sources before any split is made
  - harmonized label assignment for the RESISC45/UCMerced/AID trio (strict 12-class core)

Outputs (in benchmark_study/manifests/):
  - <dataset>_manifest.csv   one row per image
  - qc_summary.json          aggregated QC report across all datasets
  - harmonized_core_manifest.csv  the pooled Study-A manifest (RESISC45+UCMerced+AID, 12-class core)
  - eurosat_manifest.csv     Study-B manifest (native 10 classes)
"""
import os
import sys
import json
import hashlib
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image
import imagehash
import pandas as pd

DATASETS = {
    "RESISC45": Path(r"D:\SJ\UVMDATA\NWPU-RESISC45"),
    "UCMerced": Path(r"D:\SJ\UVMDATA\UCMerced_LandUse\UCMerced_landuse"),
    "AID": Path(r"D:\SJ\UVMDATA\AID\AID"),
    "EuroSAT": Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\EuroSAT_RGB"),
}

OUT_DIR = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\manifests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Strict 12-class harmonized core across RESISC45 / UCMerced / AID ----
# native_label -> harmonized_label, keyed per dataset
HARMONIZED_MAP = {
    "RESISC45": {
        "beach": "beach",
        "forest": "forest",
        "river": "river",
        "dense_residential": "dense_residential",
        "medium_residential": "medium_residential",
        "sparse_residential": "sparse_residential",
        "parking_lot": "parking_lot",
        "storage_tank": "storage_tanks",
        "harbor": "harbor_port",
        "overpass": "overpass_viaduct",
        "baseball_diamond": "baseball_field",
        "circular_farmland": "agricultural_farmland",
        "rectangular_farmland": "agricultural_farmland",
    },
    "UCMerced": {
        "beach": "beach",
        "forest": "forest",
        "river": "river",
        "denseresidential": "dense_residential",
        "mediumresidential": "medium_residential",
        "sparseresidential": "sparse_residential",
        "parkinglot": "parking_lot",
        "storagetanks": "storage_tanks",
        "harbor": "harbor_port",
        "overpass": "overpass_viaduct",
        "baseballdiamond": "baseball_field",
        "agricultural": "agricultural_farmland",
    },
    "AID": {
        "Beach": "beach",
        "Forest": "forest",
        "River": "river",
        "DenseResidential": "dense_residential",
        "MediumResidential": "medium_residential",
        "SparseResidential": "sparse_residential",
        "Parking": "parking_lot",
        "StorageTanks": "storage_tanks",
        "Port": "harbor_port",
        "Viaduct": "overpass_viaduct",
        "BaseballField": "baseball_field",
        "Farmland": "agricultural_farmland",
    },
}


def scan_one(args):
    dataset_name, class_dir, fpath = args
    rec = {
        "dataset": dataset_name,
        "native_label": class_dir,
        "filepath": str(fpath),
        "filename": fpath.name,
        "bytes": None,
        "width": None,
        "height": None,
        "mode": None,
        "format": None,
        "md5": None,
        "phash": None,
        "corrupt": False,
        "error": None,
    }
    try:
        rec["bytes"] = fpath.stat().st_size
        with open(fpath, "rb") as f:
            data = f.read()
        rec["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(fpath) as im:
            im.verify()  # structural check
        with Image.open(fpath) as im:  # re-open: verify() invalidates the handle
            im = im.convert("RGB")
            rec["width"], rec["height"] = im.size
            rec["mode"] = im.mode
            rec["format"] = im.format
            rec["phash"] = str(imagehash.phash(im))
    except Exception as e:
        rec["corrupt"] = True
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def build_dataset_manifest(dataset_name, root):
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    tasks = []
    for cd in class_dirs:
        for fpath in cd.iterdir():
            if fpath.is_file():
                tasks.append((dataset_name, cd.name, fpath))

    print(f"[{dataset_name}] {len(tasks)} files to scan across {len(class_dirs)} classes ...")
    t0 = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futs = [ex.submit(scan_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            records.append(fut.result())
            if i % 5000 == 0:
                print(f"  [{dataset_name}] {i}/{len(tasks)} scanned ...")
    df = pd.DataFrame.from_records(records)
    print(f"[{dataset_name}] done in {time.time()-t0:.1f}s")

    # harmonized label (Study A datasets only)
    if dataset_name in HARMONIZED_MAP:
        mapping = HARMONIZED_MAP[dataset_name]
        df["harmonized_label"] = df["native_label"].map(mapping)
    else:
        df["harmonized_label"] = None

    out_csv = OUT_DIR / f"{dataset_name}_manifest.csv"
    df.to_csv(out_csv, index=False)
    print(f"[{dataset_name}] manifest -> {out_csv}")
    return df


def qc_report(df, dataset_name):
    n_total = len(df)
    n_corrupt = int(df["corrupt"].sum())
    ok = df[~df["corrupt"]]

    dup_exact = ok.groupby("md5").size()
    n_exact_dup_groups = int((dup_exact > 1).sum())
    n_exact_dup_files = int(dup_exact[dup_exact > 1].sum())

    dup_phash = ok.groupby("phash").size()
    n_near_dup_groups = int((dup_phash > 1).sum())
    n_near_dup_files = int(dup_phash[dup_phash > 1].sum())

    dims = ok.groupby(["width", "height"]).size().sort_values(ascending=False)
    modes = ok["mode"].value_counts().to_dict()

    class_counts = ok["native_label"].value_counts().to_dict()

    return {
        "dataset": dataset_name,
        "n_total_files": n_total,
        "n_corrupt": n_corrupt,
        "corrupt_examples": ok.loc[df["corrupt"], "filepath"].tolist()[:10] if n_corrupt else [],
        "n_exact_duplicate_groups": n_exact_dup_groups,
        "n_exact_duplicate_files_involved": n_exact_dup_files,
        "n_near_duplicate_groups(phash)": n_near_dup_groups,
        "n_near_duplicate_files_involved(phash)": n_near_dup_files,
        "unique_dimensions": {f"{w}x{h}": int(c) for (w, h), c in dims.items()},
        "color_modes": modes,
        "n_classes": ok["native_label"].nunique(),
        "class_counts_min_max": {
            "min": min(class_counts.values()) if class_counts else None,
            "max": max(class_counts.values()) if class_counts else None,
        },
    }


def main():
    all_qc = {}
    all_dfs = {}
    for name, root in DATASETS.items():
        df = build_dataset_manifest(name, root)
        all_dfs[name] = df
        all_qc[name] = qc_report(df, name)

    with open(OUT_DIR / "qc_summary.json", "w") as f:
        json.dump(all_qc, f, indent=2)
    print(f"\nQC summary -> {OUT_DIR / 'qc_summary.json'}")

    # ---- Study A pooled manifest (RESISC45 + UCMerced + AID, harmonized core) ----
    study_a_parts = []
    for name in ["RESISC45", "UCMerced", "AID"]:
        df = all_dfs[name]
        d = df[(~df["corrupt"]) & (df["harmonized_label"].notna())].copy()
        study_a_parts.append(d)
    study_a = pd.concat(study_a_parts, ignore_index=True)
    study_a.to_csv(OUT_DIR / "harmonized_core_manifest.csv", index=False)
    print(f"Study A (harmonized core) manifest -> {len(study_a)} images -> "
          f"{OUT_DIR / 'harmonized_core_manifest.csv'}")
    print(study_a.groupby(["dataset", "harmonized_label"]).size().unstack(fill_value=0))

    # ---- Study B manifest (EuroSAT native) ----
    eurosat = all_dfs["EuroSAT"]
    eurosat_ok = eurosat[~eurosat["corrupt"]].copy()
    eurosat_ok.to_csv(OUT_DIR / "eurosat_manifest.csv", index=False)
    print(f"\nStudy B (EuroSAT native) manifest -> {len(eurosat_ok)} images -> "
          f"{OUT_DIR / 'eurosat_manifest.csv'}")


if __name__ == "__main__":
    main()

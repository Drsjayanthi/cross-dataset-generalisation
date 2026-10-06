"""
Stage 11: Computational cost summary for Table 10 (Section 3.8) and the
"145 models... 18.9 cumulative GPU-hours" figure quoted in Section 2.7.

Aggregates train_time_sec and epochs_run from every run's metrics.json
across all four Study A/B result directories, grouped by architecture.
"""
import glob
import json
import os

RESULTS = r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results"
BASE_DIRS = [
    os.path.join(RESULTS, "studyA_lodo"),
    os.path.join(RESULTS, "studyA_pooled_cv"),
    os.path.join(RESULTS, "studyA_native_cv"),
    os.path.join(RESULTS, "studyB_eurosat"),
]

ARCHS = ["resnet18", "compactcnn", "convnext_tiny", "vit_small_patch16_224", "swin_tiny_patch4_window7_224"]
LABELS = {
    "resnet18": "ResNet-18", "compactcnn": "CompactCNN", "convnext_tiny": "ConvNeXt-Tiny",
    "vit_small_patch16_224": "ViT-Small/16", "swin_tiny_patch4_window7_224": "Swin-Tiny",
}


def main():
    per_arch = {a: {"n_runs": 0, "total_sec": 0.0, "total_epochs": 0} for a in ARCHS}
    grand_total_sec = 0.0
    grand_total_runs = 0

    for base_dir in BASE_DIRS:
        for run_dir in glob.glob(os.path.join(base_dir, "*")):
            mpath = os.path.join(run_dir, "metrics.json")
            if not os.path.isfile(mpath):
                continue
            with open(mpath) as f:
                m = json.load(f)
            arch = m.get("arch")
            if arch not in per_arch:
                continue
            per_arch[arch]["n_runs"] += 1
            per_arch[arch]["total_sec"] += m["train_time_sec"]
            per_arch[arch]["total_epochs"] += m["epochs_run"]
            grand_total_sec += m["train_time_sec"]
            grand_total_runs += 1

    print("=== Table 10: computational cost by architecture ===")
    for a in ARCHS:
        d = per_arch[a]
        mean_sec_per_epoch = d["total_sec"] / d["total_epochs"]
        total_hours = d["total_sec"] / 3600
        print(f"{LABELS[a]:16s} n_runs={d['n_runs']:3d}  mean_sec_per_epoch={mean_sec_per_epoch:5.1f}  "
              f"total_GPU_hours={total_hours:.2f}")

    print()
    print(f"=== Section 2.7: grand total across all runs ===")
    print(f"Total runs: {grand_total_runs} (expected 145)")
    print(f"Total GPU-hours: {grand_total_sec/3600:.1f} (expected ~18.9)")


if __name__ == "__main__":
    main()

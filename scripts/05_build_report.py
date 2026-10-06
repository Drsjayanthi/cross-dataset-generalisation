"""
Stage 5: Assemble a single results table (CSV + printed summary) from every completed
experiment stage, joined with the bootstrap CIs from 04_stats_analysis.py.
"""
import json
from pathlib import Path
import pandas as pd

RES_ROOT = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results")
REPORTS = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\reports")

with open(REPORTS / "statistical_analysis.json") as f:
    stats = json.load(f)

rows = []
for tag, ci in stats["bootstrap_ci"].items():
    rows.append({
        "run_tag": tag,
        "n_test": ci["n_test"],
        "accuracy": ci["accuracy"]["point"],
        "acc_ci95_lo": ci["accuracy"]["ci95_lo"],
        "acc_ci95_hi": ci["accuracy"]["ci95_hi"],
        "macro_f1": ci["macro_f1"]["point"],
        "f1_ci95_lo": ci["macro_f1"]["ci95_lo"],
        "f1_ci95_hi": ci["macro_f1"]["ci95_hi"],
    })
df = pd.DataFrame(rows).sort_values("run_tag")
df.to_csv(REPORTS / "all_results_with_ci.csv", index=False)
print(df.to_string(index=False))
print(f"\nSaved -> {REPORTS / 'all_results_with_ci.csv'}")

print("\n=== McNemar (resnet18 vs each challenger, per LODO fold) ===")
for k, v in stats.get("mcnemar", {}).items():
    print(f"{k}: statistic={v['statistic']:.3f} p={v['p_value']:.4g} "
          f"(resnet18-only-correct={v['resnet18_only_correct']}, {v['challenger']}-only-correct={v['challenger_only_correct']})")

print("\n=== TOST equivalence (LODO vs in-distribution pooled CV) ===")
for k, v in stats.get("tost_equivalence", {}).items():
    print(f"{k}: LODO_acc={v['lodo_accuracy']:.4f} vs pooled={v['in_distribution_pooled_mean_accuracy']:.4f}, "
          f"margin=+/-{v['equivalence_margin']:.4f}, 90%CI diff={v['diff_90ci']}, "
          f"verdict={v['verdict']}, holm_p={v.get('holm_adjusted_p', 'NA')}")

"""
Stage 3a: Study A / A1 -- Leave-One-Dataset-Out cross-dataset generalization.
This is the primary deliverable: does a model trained on 2 of {RESISC45, UCMerced, AID}
generalize to the 3rd, completely unseen dataset (different acquisition source/sensor
characteristics, same harmonized 12-class label space)?

Two architectures run per fold:
  - resnet18 (ImageNet-pretrained, fine-tuned)   img_size=224
  - compactcnn (trained from scratch, capacity-matched)  img_size=128
so the generalization result isn't an artifact of one architecture / one inductive bias.
"""
import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_train import run_experiment

MAN_DIR = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\manifests")
RES_DIR = Path(r"D:\SJ\UVMDATA\EuroSAT-RGB\benchmark_study\results\studyA_lodo")
RES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    lodo = pd.read_csv(MAN_DIR / "splits" / "studyA_lodo_splits.csv")

    ARCH_CFG = {
        "resnet18": dict(img_size=224, max_epochs=25, patience=4, batch_size=64),
        "compactcnn": dict(img_size=128, max_epochs=35, patience=6, batch_size=64),
    }

    all_results = []
    for held_out in ["RESISC45", "UCMerced", "AID"]:
        fold = lodo[lodo["lodo_fold"] == held_out]
        train = fold[fold["lodo_split"] == "train"].reset_index(drop=True)
        test = fold[fold["lodo_split"] == "test"].reset_index(drop=True)
        for arch, cfg in ARCH_CFG.items():
            run_tag = f"lodo_heldout-{held_out}_{arch}"
            out_path = RES_DIR / run_tag
            print(f"\n=== {run_tag} === train={len(train)} test={len(test)}")
            m = run_experiment(train, test, "harmonized_label", arch, out_path,
                                run_tag=run_tag, **cfg)
            m["held_out_dataset"] = held_out
            all_results.append(m)
            with open(RES_DIR / "_summary_so_far.json", "w") as f:
                json.dump(all_results, f, indent=2)

    print("\n\n=== A1 LODO SUMMARY ===")
    for m in all_results:
        print(f"{m['held_out_dataset']:>10s} | {m['arch']:>10s} | "
              f"acc={m['test_accuracy']:.4f} macroF1={m['test_macro_f1']:.4f} "
              f"n_test={m['n_test']}")

    with open(RES_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {RES_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()

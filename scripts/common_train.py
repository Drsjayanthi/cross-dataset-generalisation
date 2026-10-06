"""
Shared training/eval utilities used by all experiment drivers.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
from sklearn.model_selection import GroupShuffleSplit
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImgDataset(Dataset):
    def __init__(self, df, label_col, label2idx, img_size, train=True):
        self.paths = df["filepath"].tolist()
        self.labels = [label2idx[l] for l in df[label_col].tolist()]
        aug = [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),   # overhead imagery: no canonical "up"
            transforms.RandomRotation(15),
        ] if train else []
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        im = Image.open(self.paths[idx]).convert("RGB")
        im = self.tf(im)
        return im, self.labels[idx]


def build_model(arch, n_classes):
    if arch == "compactcnn":
        m = CompactCNN(n_classes)
    else:
        # any timm model name (resnet18, convnext_tiny, vit_small_patch16_224,
        # swin_tiny_patch4_window7_224, ...) -- pretrained, fine-tuned end-to-end
        m = timm.create_model(arch, pretrained=True, num_classes=n_classes)
    return m.to(DEVICE)


class CompactCNN(nn.Module):
    """Small from-scratch CNN: capacity-matched sanity-check baseline (no ImageNet prior)."""
    def __init__(self, n_classes):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def carve_val_from_train(train_df, group_col="md5", val_frac=0.1, label_col="label", seed=42):
    """Hold out a small group-aware validation slice from a training set (for early stopping)."""
    if train_df[group_col].nunique() < 2:
        return train_df, train_df.sample(frac=min(val_frac, 0.5), random_state=seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, val_idx = next(gss.split(train_df, train_df[label_col], train_df[group_col]))
    return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)


def run_experiment(train_df, test_df, label_col, arch, out_path, img_size=224,
                    max_epochs=25, patience=5, batch_size=64, lr=None, seed=42,
                    run_tag=""):
    """
    Trains `arch` on train_df (carving an internal val split for early stopping),
    evaluates once on test_df (never touched during training/model selection),
    and writes predictions + metrics to out_path.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    classes = sorted(train_df[label_col].unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}

    tr_df, val_df = carve_val_from_train(train_df, val_frac=0.1, label_col=label_col, seed=seed)

    train_ds = ImgDataset(tr_df, label_col, label2idx, img_size, train=True)
    val_ds = ImgDataset(val_df, label_col, label2idx, img_size, train=False)
    test_ds = ImgDataset(test_df, label_col, label2idx, img_size, train=False)

    nw = 6
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                         num_workers=2, pin_memory=True, persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                          num_workers=2, pin_memory=True)

    # inverse-frequency class weights (train split)
    counts = tr_df[label_col].value_counts()
    weights = torch.tensor([1.0 / counts.get(c, 1) for c in classes], dtype=torch.float32)
    weights = weights / weights.sum() * len(classes)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    model = build_model(arch, len(classes))
    # from-scratch models need a higher LR; every pretrained timm backbone (resnet18,
    # convnext_tiny, vit_small_patch16_224, swin_tiny_..., etc.) fine-tunes at 3e-4
    lr = lr or (1e-3 if arch == "compactcnn" else 3e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # short linear warmup then cosine decay -- cheap insurance against the optimization
    # instability LayerNorm/attention-based backbones (ViT/Swin/ConvNeXt) can hit in the
    # first steps at this LR, without changing the steady-state schedule for anyone else
    warmup_epochs = min(2, max(1, max_epochs // 10))
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_epochs - warmup_epochs))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine],
                                                   milestones=[warmup_epochs])
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_val_f1 = -1
    best_state = None
    epochs_no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        run_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * x.size(0)
        sched.step()
        train_loss = run_loss / len(train_ds)

        # validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    out = model(x)
                val_preds.extend(out.argmax(1).cpu().numpy().tolist())
                val_true.extend(y.numpy().tolist())
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_true, val_preds)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_f1, "val_acc": val_acc})
        print(f"  [{run_tag}] epoch {epoch:02d} loss={train_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}")

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [{run_tag}] early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    train_time = time.time() - t0

    # ---- final, single evaluation on the untouched test set ----
    model.eval()
    test_preds, test_true, test_probs = [], [], []
    test_paths = test_ds.paths
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                out = model(x)
            probs = torch.softmax(out.float(), dim=1)
            test_preds.extend(out.argmax(1).cpu().numpy().tolist())
            test_probs.extend(probs.max(1).values.cpu().numpy().tolist())
            test_true.extend(y.numpy().tolist())

    idx2label = {v: k for k, v in label2idx.items()}
    pred_df = pd.DataFrame({
        "filepath": test_paths,
        "true_label": [idx2label[i] for i in test_true],
        "pred_label": [idx2label[i] for i in test_preds],
        "confidence": test_probs,
    })

    metrics = {
        "run_tag": run_tag,
        "arch": arch,
        "n_train": len(tr_df), "n_val": len(val_df), "n_test": len(test_df),
        "n_classes": len(classes),
        "classes": classes,
        "best_val_macro_f1": best_val_f1,
        "epochs_run": len(history),
        "train_time_sec": train_time,
        "test_accuracy": accuracy_score(test_true, test_preds),
        "test_macro_f1": f1_score(test_true, test_preds, average="macro", zero_division=0),
        "test_weighted_f1": f1_score(test_true, test_preds, average="weighted", zero_division=0),
        "test_macro_precision": precision_score(test_true, test_preds, average="macro", zero_division=0),
        "test_macro_recall": recall_score(test_true, test_preds, average="macro", zero_division=0),
        "per_class_report": classification_report(test_true, test_preds, target_names=classes,
                                                    zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(test_true, test_preds).tolist(),
    }

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path / "predictions.csv", index=False)
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  [{run_tag}] DONE acc={metrics['test_accuracy']:.4f} "
          f"macroF1={metrics['test_macro_f1']:.4f} ({train_time:.0f}s, {len(history)} epochs)")
    return metrics

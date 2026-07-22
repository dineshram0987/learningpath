"""
train_mlp.py
-------------
MLP baseline for chart-type classification.
Flattens 128×128×3 images → 49,152-dim vector, trains a 3-layer MLP.

Run:
    python src/train_mlp.py
"""

import os
import sys
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
sys.path.insert(0, ".")

# ── Config ────────────────────────────────────────────────────────────────────
CHARTS_DIR  = os.path.join("data", "charts")
MODELS_DIR  = "models"
REPORTS_DIR = "reports"
IMG_SIZE    = 128
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-3
CLASSES     = ["bar", "line", "scatter", "pie", "histogram", "box"]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[mlp] Using device: {DEVICE}")


# ── Data Transforms ───────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_loaders():
    train_ds = datasets.ImageFolder(os.path.join(CHARTS_DIR, "train"), transform=transform)
    val_ds   = datasets.ImageFolder(os.path.join(CHARTS_DIR, "val"),   transform=transform)
    test_ds  = datasets.ImageFolder(os.path.join(CHARTS_DIR, "test"),  transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_ds.classes


# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """Simple 3-layer MLP that flattens image pixels as input."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        all_preds.extend(out.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, acc, f1, all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("DAY 3: MLP BASELINE TRAINING")
    print("=" * 60)

    train_loader, val_loader, test_loader, class_names = get_loaders()
    print(f"[mlp] Classes: {class_names}")
    print(f"[mlp] Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    input_dim = 3 * IMG_SIZE * IMG_SIZE  # channels × H × W
    model = MLP(input_dim=input_dim, num_classes=len(class_names)).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[mlp] Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  val_f1={vl_f1:.4f}")

    elapsed = time.time() - start

    # ── Test Set Evaluation ────────────────────────────────────────────────────
    _, test_acc, test_f1, preds, labels = evaluate(model, test_loader, criterion)
    print(f"\n[mlp] Test Accuracy: {test_acc:.4f} | Test F1 Macro: {test_f1:.4f}")
    print(f"[mlp] Training time: {elapsed:.1f}s\n")
    print(classification_report(labels, preds, target_names=class_names))

    # ── Save Metrics ───────────────────────────────────────────────────────────
    metrics = {
        "model": "MLP",
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1, 4),
        "training_time_sec": round(elapsed, 1),
        "total_params": total_params,
        "epochs": EPOCHS,
        "history": history,
        "class_names": class_names,
    }
    metrics_path = os.path.join(REPORTS_DIR, "day3_mlp_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[mlp] Saved metrics -> {metrics_path}")

    # ── Save model ─────────────────────────────────────────────────────────────
    mlp_path = os.path.join(MODELS_DIR, "mlp_chart.pt")
    torch.save(model.state_dict(), mlp_path)
    print(f"[mlp] Saved model -> {mlp_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

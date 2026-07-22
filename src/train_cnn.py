"""
train_cnn.py
-------------
Custom CNN-from-scratch for chart-type classification (6 classes).

Architecture:
  Conv(3→32) → BN → ReLU → MaxPool
  Conv(32→64) → BN → ReLU → MaxPool
  Conv(64→128) → BN → ReLU → MaxPool
  FC(2048→256) → Dropout → FC(256→6)

Data augmentation: RandomHorizontalFlip, RandomRotation(15), ColorJitter
Training: 20 epochs, Adam + CosineAnnealingLR, best val-acc model saved.

Run:
    python src/train_cnn.py
"""

import os
import sys
import time
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)
import seaborn as sns

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
sys.path.insert(0, ".")

# ── Config ────────────────────────────────────────────────────────────────────
CHARTS_DIR  = os.path.join("data", "charts")
MODELS_DIR  = "models"
REPORTS_DIR = "reports"
IMG_SIZE    = 128
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-3

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[cnn] Using device: {DEVICE}")


# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),       # mild — some charts are direction-sensitive
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def get_loaders():
    train_ds = datasets.ImageFolder(os.path.join(CHARTS_DIR, "train"),
                                    transform=train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(CHARTS_DIR, "val"),
                                    transform=eval_transform)
    test_ds  = datasets.ImageFolder(os.path.join(CHARTS_DIR, "test"),
                                    transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_ds.classes


# ── Model ─────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ChartCNN(nn.Module):
    """3-stage CNN classifier for 128×128 chart images."""

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),        # 128→64
            ConvBlock(32, 64),       # 64→32
            ConvBlock(64, 128),      # 32→16
            ConvBlock(128, 256),     # 16→8
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # → 256×4×4 = 4096
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ── Loops ─────────────────────────────────────────────────────────────────────
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
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        total_loss += criterion(out, labels).item() * imgs.size(0)
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, acc, f1, all_preds, all_labels


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("CNN-from-Scratch — Loss Curves")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("CNN-from-Scratch — Accuracy Curves")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(labels, preds, class_names, path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("CNN Confusion Matrix")
    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("DAY 3: CNN FROM SCRATCH TRAINING")
    print("=" * 60)

    train_loader, val_loader, test_loader, class_names = get_loaders()
    print(f"[cnn] Classes: {class_names}")

    model = ChartCNN(num_classes=len(class_names)).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cnn] Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "cnn_scratch.pt")
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        tag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_model_path)
            tag = "  [saved best]"

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.4f}  "
              f"vl_f1={vl_f1:.4f}{tag}")

    elapsed = time.time() - start

    # ── Load best, evaluate test ───────────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    _, test_acc, test_f1, preds, labels = evaluate(model, test_loader, criterion)

    print(f"\n[cnn] Best Val Acc: {best_val_acc:.4f}")
    print(f"[cnn] Test Accuracy: {test_acc:.4f} | Test F1 Macro: {test_f1:.4f}")
    print(f"[cnn] Training time: {elapsed:.1f}s\n")
    print(classification_report(labels, preds, target_names=class_names))

    # ── Plots ──────────────────────────────────────────────────────────────────
    curves_path = os.path.join(REPORTS_DIR, "day3_cnn_curves.png")
    plot_curves(history, curves_path)
    print(f"[cnn] Saved curves -> {curves_path}")

    cm_path = os.path.join(REPORTS_DIR, "day3_cnn_confusion.png")
    plot_confusion_matrix(labels, preds, class_names, cm_path)
    print(f"[cnn] Saved confusion matrix -> {cm_path}")

    # ── Save metrics ───────────────────────────────────────────────────────────
    metrics = {
        "model": "CNN-from-scratch",
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "training_time_sec": round(elapsed, 1),
        "total_params": total_params,
        "epochs": EPOCHS,
        "history": history,
        "class_names": class_names,
    }
    metrics_path = os.path.join(REPORTS_DIR, "day3_cnn_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[cnn] Saved metrics -> {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

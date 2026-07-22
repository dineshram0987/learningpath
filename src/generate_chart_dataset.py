"""
generate_chart_dataset.py
--------------------------
Programmatically generate ~3,000 synthetic chart images across 6 classes:
  bar, line, scatter, pie, histogram, box

Each class gets ~500 images with varied colors, data, labels, and sizes.
80/10/10 stratified split → data/charts/{train,val,test}/{class_name}/

Run:
    python src/generate_chart_dataset.py
"""

import os
import sys
import random
import math
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────
CHARTS_DIR  = os.path.join("data", "charts")
REPORTS_DIR = "reports"
IMG_SIZE    = (128, 128)      # pixels (width × height)
DPI         = 100             # at 128px / 100 DPI → 1.28 in figure
N_PER_CLASS = 500
CLASSES     = ["bar", "line", "scatter", "pie", "histogram", "box"]
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST = remaining 10 %

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Colour palettes to vary between images
PALETTES = [
    ["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
    ["#17becf", "#bcbd22", "#393b79", "#843c39"],
    ["steelblue", "tomato", "mediumseagreen", "goldenrod"],
    ["teal", "coral", "slateblue", "sienna"],
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _figsize_in():
    w = IMG_SIZE[0] / DPI
    h = IMG_SIZE[1] / DPI
    return (w, h)


def _rand_palette():
    return random.choice(PALETTES)


def _rand_data(n=None, low=5, high=100):
    n = n or random.randint(4, 10)
    return np.random.randint(low, high, size=n).astype(float)


def _rand_labels(n):
    bases = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
    return [f"{random.choice(bases)}{i}" for i in range(n)]


def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ── Chart Generators ──────────────────────────────────────────────────────────

def generate_bar(idx: int, out_dir: str):
    n = random.randint(4, 9)
    vals = _rand_data(n)
    labels = _rand_labels(n)
    palette = _rand_palette()
    colors = [palette[i % len(palette)] for i in range(n)]

    orient = random.choice(["v", "h"])
    fig, ax = plt.subplots(figsize=_figsize_in())
    ax.set_facecolor(random.choice(["white", "#f5f5f5", "#fafafa"]))

    if orient == "v":
        ax.bar(labels, vals, color=colors, width=random.uniform(0.5, 0.85),
               edgecolor="none")
        if random.random() > 0.5:
            ax.set_xlabel(random.choice(["Category", "Group", "Item"]),
                          fontsize=5)
        ax.set_ylabel(random.choice(["Count", "Value", "Score"]), fontsize=5)
    else:
        ax.barh(labels, vals, color=colors, edgecolor="none",
                height=random.uniform(0.5, 0.85))
        if random.random() > 0.5:
            ax.set_ylabel(random.choice(["Category", "Group"]), fontsize=5)
        ax.set_xlabel(random.choice(["Count", "Value"]), fontsize=5)

    if random.random() > 0.5:
        ax.set_title(random.choice(["Sales", "Scores", "Count by Group"]),
                     fontsize=6)
    ax.tick_params(labelsize=4)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"bar_{idx:04d}.png"))


def generate_line(idx: int, out_dir: str):
    n_lines = random.randint(1, 3)
    x_len = random.randint(10, 30)
    x = np.linspace(0, random.uniform(5, 20), x_len)
    palette = _rand_palette()

    fig, ax = plt.subplots(figsize=_figsize_in())
    ax.set_facecolor(random.choice(["white", "#f9f9f9"]))

    for k in range(n_lines):
        y = np.cumsum(np.random.randn(x_len)) + np.random.uniform(0, 50)
        style = random.choice(["-", "--", "-."])
        marker = random.choice(["", "o", "s", "^"])
        ax.plot(x, y, linestyle=style,
                marker=marker if marker else None,
                markersize=2,
                color=palette[k % len(palette)],
                linewidth=random.uniform(0.8, 2.0),
                label=f"Series {k+1}" if n_lines > 1 else None)

    if n_lines > 1 and random.random() > 0.5:
        ax.legend(fontsize=4, loc="best")

    ax.set_xlabel(random.choice(["Time", "Step", "Month"]), fontsize=5)
    ax.set_ylabel(random.choice(["Value", "Metric", "Score"]), fontsize=5)
    if random.random() > 0.5:
        ax.set_title(random.choice(["Trend Over Time", "Time Series"]), fontsize=6)
    ax.tick_params(labelsize=4)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"line_{idx:04d}.png"))


def generate_scatter(idx: int, out_dir: str):
    n = random.randint(30, 150)
    x = np.random.randn(n) * random.uniform(1, 5)
    y = x * random.uniform(-1, 1) + np.random.randn(n) * random.uniform(0.5, 3)
    palette = _rand_palette()

    n_groups = random.randint(1, 3)
    fig, ax = plt.subplots(figsize=_figsize_in())
    ax.set_facecolor(random.choice(["white", "#f8f8f8"]))

    chunk = len(x) // n_groups
    for k in range(n_groups):
        sl = slice(k * chunk, (k + 1) * chunk if k < n_groups - 1 else None)
        ax.scatter(x[sl], y[sl],
                   color=palette[k % len(palette)],
                   s=random.uniform(6, 25),
                   alpha=random.uniform(0.6, 1.0),
                   edgecolors="none",
                   label=f"Grp {k+1}" if n_groups > 1 else None)

    if n_groups > 1 and random.random() > 0.5:
        ax.legend(fontsize=4)

    ax.set_xlabel(random.choice(["X", "Feature A", "Input"]), fontsize=5)
    ax.set_ylabel(random.choice(["Y", "Feature B", "Output"]), fontsize=5)
    if random.random() > 0.5:
        ax.set_title(random.choice(["Scatter Plot", "Correlation"]), fontsize=6)
    ax.tick_params(labelsize=4)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"scatter_{idx:04d}.png"))


def generate_pie(idx: int, out_dir: str):
    n = random.randint(3, 7)
    sizes = np.abs(np.random.randn(n)) + 0.5
    sizes /= sizes.sum()
    palette = _rand_palette()
    colors = [palette[i % len(palette)] for i in range(n)]
    labels = _rand_labels(n) if random.random() > 0.4 else None

    explode_one = random.random() > 0.6
    explode = [0.0] * n
    if explode_one:
        explode[np.argmax(sizes)] = random.uniform(0.05, 0.15)

    fig, ax = plt.subplots(figsize=_figsize_in())
    ax.pie(sizes, labels=labels, colors=colors, explode=explode,
           autopct="%1.0f%%" if random.random() > 0.5 else None,
           textprops={"fontsize": 4},
           startangle=random.uniform(0, 360))
    if random.random() > 0.5:
        ax.set_title(random.choice(["Distribution", "Share", "Breakdown"]),
                     fontsize=6)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"pie_{idx:04d}.png"))


def generate_histogram(idx: int, out_dir: str):
    n = random.randint(100, 500)
    dist = random.choice(["normal", "uniform", "exponential", "bimodal"])
    if dist == "normal":
        data = np.random.normal(random.uniform(30, 70), random.uniform(5, 20), n)
    elif dist == "uniform":
        data = np.random.uniform(0, random.uniform(50, 100), n)
    elif dist == "exponential":
        data = np.random.exponential(random.uniform(10, 30), n)
    else:  # bimodal
        data = np.concatenate([
            np.random.normal(25, 5, n // 2),
            np.random.normal(75, 8, n // 2)
        ])

    bins = random.randint(8, 25)
    palette = _rand_palette()
    color = random.choice(palette)

    fig, ax = plt.subplots(figsize=_figsize_in())
    ax.set_facecolor(random.choice(["white", "#f7f7f7"]))
    ax.hist(data, bins=bins, color=color,
            edgecolor="white" if random.random() > 0.4 else "none",
            linewidth=0.3, density=random.random() > 0.5)
    ax.set_xlabel(random.choice(["Value", "Measurement", "Score"]), fontsize=5)
    ax.set_ylabel(random.choice(["Frequency", "Count", "Density"]), fontsize=5)
    if random.random() > 0.5:
        ax.set_title(random.choice(["Distribution", "Histogram"]), fontsize=6)
    ax.tick_params(labelsize=4)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"histogram_{idx:04d}.png"))


def generate_box(idx: int, out_dir: str):
    n_groups = random.randint(2, 6)
    groups = [np.random.normal(random.uniform(20, 80),
                               random.uniform(5, 20),
                               random.randint(30, 100))
              for _ in range(n_groups)]
    labels = _rand_labels(n_groups)
    palette = _rand_palette()
    colors = [palette[i % len(palette)] for i in range(n_groups)]

    fig, ax = plt.subplots(figsize=_figsize_in())
    ax.set_facecolor(random.choice(["white", "#f8f8f8"]))
    bp = ax.boxplot(groups, patch_artist=True, labels=labels,
                    medianprops={"color": "black", "linewidth": 1.0},
                    whiskerprops={"linewidth": 0.6},
                    capprops={"linewidth": 0.6},
                    flierprops={"markersize": 2, "alpha": 0.5})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(random.uniform(0.6, 1.0))

    ax.set_xlabel(random.choice(["Group", "Category"]), fontsize=5)
    ax.set_ylabel(random.choice(["Value", "Score"]), fontsize=5)
    if random.random() > 0.5:
        ax.set_title(random.choice(["Box Plot", "Distribution by Group"]),
                     fontsize=6)
    ax.tick_params(labelsize=4)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"box_{idx:04d}.png"))


# ── Generator dispatch ────────────────────────────────────────────────────────
GENERATORS = {
    "bar":       generate_bar,
    "line":      generate_line,
    "scatter":   generate_scatter,
    "pie":       generate_pie,
    "histogram": generate_histogram,
    "box":       generate_box,
}


# ── Split & Build directories ─────────────────────────────────────────────────

def build_split_dirs():
    """Create data/charts/{train,val,test}/{class}/ directories."""
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            os.makedirs(os.path.join(CHARTS_DIR, split, cls), exist_ok=True)


def split_indices(n, train_r=TRAIN_RATIO, val_r=VAL_RATIO):
    indices = list(range(n))
    random.shuffle(indices)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return (indices[:n_train],
            indices[n_train:n_train + n_val],
            indices[n_train + n_val:])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAY 3: SYNTHETIC CHART DATASET GENERATION")
    print("=" * 60)

    # Clean existing charts dir for a fresh run
    if os.path.exists(CHARTS_DIR):
        shutil.rmtree(CHARTS_DIR)
    build_split_dirs()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    class_counts = {}

    for cls_name in CLASSES:
        gen_fn = GENERATORS[cls_name]
        train_idx, val_idx, test_idx = split_indices(N_PER_CLASS)

        splits = {
            "train": train_idx,
            "val":   val_idx,
            "test":  test_idx,
        }

        for split, idxs in splits.items():
            out_dir = os.path.join(CHARTS_DIR, split, cls_name)
            for i, img_idx in enumerate(idxs):
                gen_fn(img_idx, out_dir)

        counts = {split: len(idxs) for split, idxs in splits.items()}
        class_counts[cls_name] = counts
        total = sum(counts.values())
        print(f"  {cls_name:12s}  train={counts['train']:4d}  "
              f"val={counts['val']:3d}  test={counts['test']:3d}  "
              f"total={total}")

    # ── Sample grid ───────────────────────────────────────────────────────────
    print("\n[generate] Saving sample grid -> reports/day3_dataset_sample.png")
    fig, axes = plt.subplots(6, 5, figsize=(10, 12))
    for row, cls_name in enumerate(CLASSES):
        train_dir = os.path.join(CHARTS_DIR, "train", cls_name)
        imgs = sorted(os.listdir(train_dir))[:5]
        for col, fname in enumerate(imgs):
            img = plt.imread(os.path.join(train_dir, fname))
            axes[row][col].imshow(img)
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_ylabel(cls_name, fontsize=9,
                                          rotation=0, labelpad=40,
                                          va="center")

    fig.suptitle("Day 3 — Synthetic Chart Dataset (5 samples per class)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    sample_path = os.path.join(REPORTS_DIR, "day3_dataset_sample.png")
    fig.savefig(sample_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[generate] Saved -> {sample_path}")

    total_images = sum(
        sum(v.values()) for v in class_counts.values()
    )
    print(f"\n[generate] DONE — {total_images} images across {len(CLASSES)} classes")
    print("=" * 60)


if __name__ == "__main__":
    main()

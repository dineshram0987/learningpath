"""
evaluate.py
-----------
Model evaluation for the Personalized Learning Platform.

Produces:
  - Classification reports (Accuracy, Precision, Recall, F1) for both models
  - ROC-AUC + Confusion Matrix for dropout_risk model
  - Feature importance bar charts (Random Forest) for both tasks
  - All plots saved to reports/
  - Metrics JSON saved to reports/day1_metrics.json

Usage:
    python src/evaluate.py
    # Or import after training:
    from src.evaluate import run_evaluation
    run_evaluation(training_artifacts)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

REPORTS_DIR = "reports"
MODELS_DIR  = "models"

# ── Consistent plot style ──────────────────────────────────────────
PALETTE = {
    "bg":       "#0f1117",
    "surface":  "#1a1d27",
    "accent1":  "#6c63ff",
    "accent2":  "#00d4aa",
    "accent3":  "#ff6b6b",
    "text":     "#e8e8f0",
    "grid":     "#2a2d3e",
}

def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["surface"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["text"],
        "ytick.color":       PALETTE["text"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["grid"],
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "axes.grid":         True,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "figure.dpi":        120,
    })


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------

def compute_classification_metrics(y_true, y_pred, y_prob=None, average="macro"):
    """Return a dict of evaluation metrics."""
    metrics = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, average=average, zero_division=0)), 4),
    }
    if y_prob is not None and average == "binary":
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    return metrics


# ------------------------------------------------------------------
# Plot: Confusion Matrix
# ------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Save a styled confusion matrix heatmap."""
    set_plot_style()
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor=PALETTE["grid"],
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, pad=15, color=PALETTE["text"])
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label", labelpad=10)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[evaluate] Saved confusion matrix -> {save_path}")


# ------------------------------------------------------------------
# Plot: ROC Curve
# ------------------------------------------------------------------

def plot_roc_curve(y_true, y_prob, auc_score, save_path):
    """Save ROC curve plot for binary classifier."""
    set_plot_style()
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    ax.plot(fpr, tpr, color=PALETTE["accent1"], lw=2.5, label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["accent3"], lw=1.5, linestyle="--", alpha=0.7, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE["accent1"])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Dropout Risk Model")
    ax.legend(loc="lower right", facecolor=PALETTE["surface"], edgecolor=PALETTE["grid"])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[evaluate] Saved ROC curve -> {save_path}")


# ------------------------------------------------------------------
# Plot: Feature Importance
# ------------------------------------------------------------------

def plot_feature_importance(rf_model, feature_cols, title, save_path, top_n=20):
    """Save a horizontal bar chart of top-N feature importances."""
    set_plot_style()
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    sorted_features    = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]

    # Clean up feature names for display
    display_names = [f.replace("score_", "").replace("_", " ").replace("career goal ", "goal: ").title()
                     for f in sorted_features]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["surface"])

    colors = [PALETTE["accent1"] if i % 2 == 0 else PALETTE["accent2"] for i in range(len(sorted_importances))]
    bars = ax.barh(range(top_n), sorted_importances[::-1], color=colors[::-1], edgecolor="none", height=0.7)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(display_names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)", labelpad=10)
    ax.set_title(title, pad=12)

    # Annotate bars
    for bar, val in zip(bars, sorted_importances[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color=PALETTE["text"])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[evaluate] Saved feature importance -> {save_path}")


# ------------------------------------------------------------------
# Plot: Model Comparison Bar Chart
# ------------------------------------------------------------------

def plot_model_comparison(module_results, dropout_results, save_path):
    """Side-by-side bar chart comparing LR vs RF on both tasks."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Baseline Model Comparison — LR vs Random Forest", fontsize=14,
                 fontweight="bold", color=PALETTE["text"], y=1.02)

    datasets = [
        (module_results,  "Next Best Module (Multi-class)", axes[0]),
        (dropout_results, "Dropout Risk (Binary)",          axes[1]),
    ]

    metrics   = ["accuracy", "f1_macro"]
    metric_labels = ["Accuracy", "F1 Macro"]
    model_colors  = [PALETTE["accent1"], PALETTE["accent2"]]
    x = np.arange(len(metrics))
    width = 0.3

    for results, title, ax in datasets:
        ax.set_facecolor(PALETTE["surface"])
        for i, (result, color) in enumerate(zip(results, model_colors)):
            vals = [result.get("accuracy", 0), result.get("f1_macro", 0)]
            bars = ax.bar(x + i * width, vals, width=width, label=result["model"],
                          color=color, alpha=0.9, edgecolor="none")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=PALETTE["text"])

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, pad=10)
        ax.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["grid"], fontsize=9)
        ax.set_ylabel("Score")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[evaluate] Saved model comparison -> {save_path}")


# ------------------------------------------------------------------
# Main Evaluation Pipeline
# ------------------------------------------------------------------

def run_evaluation(training_artifacts: dict) -> dict:
    """
    Full evaluation pipeline consuming training artifacts dict.
    Produces all reports and plots. Returns metrics dict.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    best_module_model  = training_artifacts["best_module_model"]
    best_dropout_model = training_artifacts["best_dropout_model"]
    X_test             = training_artifacts["X_test"]
    ym_test            = training_artifacts["ym_test"]
    yd_test            = training_artifacts["yd_test"]
    le                 = training_artifacts["le"]
    feature_cols       = training_artifacts["feature_cols"]
    rf_module_model    = training_artifacts["rf_module_model"]
    rf_dropout_model   = training_artifacts["rf_dropout_model"]
    module_results     = training_artifacts["module_results"]
    dropout_results    = training_artifacts["dropout_results"]
    best_module_name   = training_artifacts["best_module_result"]["model"]
    best_dropout_name  = training_artifacts["best_dropout_result"]["model"]

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # ── 1. Next Best Module Evaluation ────────────────────────────
    print("\n── next_best_module (Multi-class) ──")
    ym_pred = best_module_model.predict(X_test)
    module_metrics = compute_classification_metrics(ym_test, ym_pred, average="macro")

    print(f"  Accuracy : {module_metrics['accuracy']:.4f}")
    print(f"  Precision: {module_metrics['precision']:.4f}")
    print(f"  Recall   : {module_metrics['recall']:.4f}")
    print(f"  F1 Macro : {module_metrics['f1']:.4f}")
    print("\n  Full Classification Report:")
    print(classification_report(ym_test, ym_pred, target_names=le.classes_, zero_division=0))

    # Confusion matrix
    plot_confusion_matrix(
        ym_test, ym_pred,
        class_names=le.classes_,
        title="Confusion Matrix — Next Best Module",
        save_path=os.path.join(REPORTS_DIR, "module_confusion_matrix.png"),
    )

    # ── 2. Dropout Risk Evaluation ─────────────────────────────────
    print("\n── dropout_risk (Binary) ──")
    yd_pred = best_dropout_model.predict(X_test)

    if hasattr(best_dropout_model, "predict_proba"):
        yd_prob = best_dropout_model.predict_proba(X_test)[:, 1]
    else:
        yd_prob = best_dropout_model.decision_function(X_test)

    dropout_metrics = compute_classification_metrics(
        yd_test, yd_pred, y_prob=yd_prob, average="binary"
    )
    print(f"  Accuracy : {dropout_metrics['accuracy']:.4f}")
    print(f"  Precision: {dropout_metrics['precision']:.4f}")
    print(f"  Recall   : {dropout_metrics['recall']:.4f}")
    print(f"  F1       : {dropout_metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {dropout_metrics['roc_auc']:.4f}")
    print("\n  Full Classification Report:")
    print(classification_report(yd_test, yd_pred, target_names=["No Risk", "At Risk"], zero_division=0))

    # Confusion matrix
    plot_confusion_matrix(
        yd_test, yd_pred,
        class_names=["No Risk", "At Risk"],
        title="Confusion Matrix — Dropout Risk",
        save_path=os.path.join(REPORTS_DIR, "dropout_confusion_matrix.png"),
    )

    # ROC Curve
    plot_roc_curve(
        yd_test, yd_prob,
        auc_score=dropout_metrics["roc_auc"],
        save_path=os.path.join(REPORTS_DIR, "dropout_roc_curve.png"),
    )

    # ── 3. Feature Importance Plots ────────────────────────────────
    plot_feature_importance(
        rf_module_model, list(feature_cols),
        title="Feature Importance — Next Best Module (Random Forest)",
        save_path=os.path.join(REPORTS_DIR, "module_feature_importance.png"),
    )

    plot_feature_importance(
        rf_dropout_model, list(feature_cols),
        title="Feature Importance — Dropout Risk (Random Forest)",
        save_path=os.path.join(REPORTS_DIR, "dropout_feature_importance.png"),
    )

    # ── 4. Model Comparison Plot ───────────────────────────────────
    plot_model_comparison(
        module_results, dropout_results,
        save_path=os.path.join(REPORTS_DIR, "model_comparison.png"),
    )

    # ── 5. Save metrics JSON ───────────────────────────────────────
    all_metrics = {
        "next_best_module": {
            "best_model": best_module_name,
            "metrics":    module_metrics,
            "all_models": [
                {k: v for k, v in r.items() if k != "model_obj"}
                for r in module_results
            ],
        },
        "dropout_risk": {
            "best_model": best_dropout_name,
            "metrics":    dropout_metrics,
            "all_models": [
                {k: v for k, v in r.items() if k != "model_obj"}
                for r in dropout_results
            ],
        },
    }

    metrics_path = os.path.join(REPORTS_DIR, "day1_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[evaluate] Metrics saved -> {metrics_path}")

    print("\n[evaluate] All evaluation artefacts saved to reports/")
    return all_metrics


if __name__ == "__main__":
    # Stand-alone evaluation: load saved models and run on test split
    from src.data_prep import run_data_prep
    from src.features import build_features, get_train_test_split
    import pandas as pd

    processed_path = os.path.join("data", "processed", "learners_clean.csv")
    if not os.path.exists(processed_path):
        df_clean = run_data_prep()
    else:
        df_clean = pd.read_csv(processed_path)

    df_feat = build_features(df_clean)
    X_train, X_test, ym_train, ym_test, yd_train, yd_test, le, feature_cols = get_train_test_split(df_feat)

    best_module_model  = joblib.load(os.path.join(MODELS_DIR, "next_best_module_model.pkl"))
    best_dropout_model = joblib.load(os.path.join(MODELS_DIR, "dropout_risk_model.pkl"))
    rf_module_model    = best_module_model
    rf_dropout_model   = best_dropout_model

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    # Minimal artifacts for standalone evaluation
    artifacts = {
        "best_module_model":   best_module_model,
        "best_dropout_model":  best_dropout_model,
        "rf_module_model":     rf_module_model,
        "rf_dropout_model":    rf_dropout_model,
        "X_test":              X_test,
        "ym_test":             ym_test,
        "yd_test":             yd_test,
        "le":                  le,
        "feature_cols":        feature_cols,
        "module_results":      [{"model": "Saved", "task": "multiclass", "accuracy": 0, "f1_macro": 0}],
        "dropout_results":     [{"model": "Saved", "task": "binary",     "accuracy": 0, "f1_macro": 0}],
        "best_module_result":  {"model": type(best_module_model).__name__},
        "best_dropout_result": {"model": type(best_dropout_model).__name__},
    }
    run_evaluation(artifacts)

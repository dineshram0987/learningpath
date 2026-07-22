"""
train.py
--------
Baseline model training for the Personalized Learning Platform.

Trains TWO sets of models:
  1. next_best_module  -> Multi-class classification (7 classes)
     Models tried: Logistic Regression, Random Forest
  2. dropout_risk      -> Binary classification
     Models tried: Logistic Regression, Random Forest

Selection criterion: macro-averaged F1 on test set (handles class imbalance).
Best models are saved to models/ as .pkl files.

Usage:
    python src/train.py
    # Or import:
    from src.train import run_training
    results = run_training()
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.data_prep import run_data_prep, load_raw_data, clean_data
from src.features import build_features, get_train_test_split

MODELS_DIR = "models"
REPORTS_DIR = "reports"


def train_logistic_regression(X_train, y_train, task: str = "multiclass"):
    """
    Train Logistic Regression with L2 regularisation.
    - multi_class='multinomial' and solver='lbfgs' for multi-class.
    - class_weight='balanced' to handle class imbalance in dropout task.
    """
    if task == "binary":
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs",
        )
    else:
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
        )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, task: str = "multiclass"):
    """
    Train Random Forest Classifier.
    - n_estimators=200 for stable estimates without excessive compute.
    - class_weight='balanced_subsample' to handle imbalance at bootstrap level.
    - max_depth=None (fully grown) for Day 1 baseline — will be tuned on Day 2.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
        max_depth=15,           # slight constraint to reduce overfitting
        min_samples_leaf=3,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str, task: str):
    """Compute and return evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    avg = "binary" if task == "binary" else "macro"

    metrics = {
        "model": model_name,
        "task": task,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
    }
    print(f"\n  [{model_name}] Accuracy={metrics['accuracy']:.4f}  F1-macro={metrics['f1_macro']:.4f}")
    return metrics, y_pred


def select_best_model(results: list) -> dict:
    """Select model with highest macro F1 score."""
    return max(results, key=lambda r: r["f1_macro"])


def run_training() -> dict:
    """
    Full training pipeline:
    1. Load and clean raw data
    2. Build features
    3. Train and compare LR vs RF for both tasks
    4. Save best models
    5. Return results dict for evaluate.py
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ---- Load + Clean + Feature Engineering ----
    print("=" * 60)
    print("STEP 1: Data loading and cleaning")
    print("=" * 60)

    # Try to load processed data; if not found, run full pipeline
    processed_path = os.path.join("data", "processed", "learners_clean.csv")
    if not os.path.exists(processed_path):
        df_clean = run_data_prep()
    else:
        df_clean = pd.read_csv(processed_path)
        print(f"[train] Loaded existing cleaned data: {df_clean.shape}")

    print("\n" + "=" * 60)
    print("STEP 2: Feature engineering")
    print("=" * 60)
    df_feat = build_features(df_clean)

    print("\n" + "=" * 60)
    print("STEP 3: Train/test split")
    print("=" * 60)
    X_train, X_test, ym_train, ym_test, yd_train, yd_test, le, feature_cols = get_train_test_split(df_feat)

    # ---- MODULE CLASSIFICATION (Multi-class) ----
    print("\n" + "=" * 60)
    print("STEP 4a: next_best_module — Multi-class Classification")
    print("=" * 60)
    print(f"  Classes: {list(le.classes_)}")

    module_results = []

    # Logistic Regression
    print("\n  Training Logistic Regression...")
    lr_module = train_logistic_regression(X_train, ym_train, task="multiclass")
    m, _ = evaluate_model(lr_module, X_test, ym_test, "LogisticRegression", "multiclass")
    module_results.append({**m, "model_obj": lr_module})

    # Random Forest
    print("\n  Training Random Forest...")
    rf_module = train_random_forest(X_train, ym_train, task="multiclass")
    m, _ = evaluate_model(rf_module, X_test, ym_test, "RandomForest", "multiclass")
    module_results.append({**m, "model_obj": rf_module})

    best_module_result = select_best_model(module_results)
    best_module_model  = best_module_result.pop("model_obj")
    print(f"\n  [OK] Best model for next_best_module: {best_module_result['model']} (F1={best_module_result['f1_macro']:.4f})")

    # Save best module model
    module_model_path = os.path.join(MODELS_DIR, "next_best_module_model.pkl")
    joblib.dump(best_module_model, module_model_path)
    print(f"  [OK] Saved -> {module_model_path}")

    # ---- DROPOUT RISK CLASSIFICATION (Binary) ----
    print("\n" + "=" * 60)
    print("STEP 4b: dropout_risk — Binary Classification")
    print("=" * 60)

    dropout_results = []

    # Logistic Regression
    print("\n  Training Logistic Regression...")
    lr_dropout = train_logistic_regression(X_train, yd_train, task="binary")
    m, _ = evaluate_model(lr_dropout, X_test, yd_test, "LogisticRegression", "binary")
    dropout_results.append({**m, "model_obj": lr_dropout})

    # Random Forest
    print("\n  Training Random Forest...")
    rf_dropout = train_random_forest(X_train, yd_train, task="binary")
    m, _ = evaluate_model(rf_dropout, X_test, yd_test, "RandomForest", "binary")
    dropout_results.append({**m, "model_obj": rf_dropout})

    best_dropout_result = select_best_model(dropout_results)
    best_dropout_model  = best_dropout_result.pop("model_obj")
    print(f"\n  [OK] Best model for dropout_risk: {best_dropout_result['model']} (F1={best_dropout_result['f1_macro']:.4f})")

    # Save best dropout model
    dropout_model_path = os.path.join(MODELS_DIR, "dropout_risk_model.pkl")
    joblib.dump(best_dropout_model, dropout_model_path)
    print(f"  [OK] Saved -> {dropout_model_path}")

    # ---- Save all competitor results (for evaluate.py report) ----
    all_lr_module_result  = [r for r in module_results  if r["model"] == "LogisticRegression"]
    all_rf_module_result  = [r for r in module_results  if r["model"] == "RandomForest"]
    all_lr_dropout_result = [r for r in dropout_results if r["model"] == "LogisticRegression"]
    all_rf_dropout_result = [r for r in dropout_results if r["model"] == "RandomForest"]

    # Store model references for evaluate.py
    training_artifacts = {
        "best_module_model":       best_module_model,
        "best_module_result":      best_module_result,
        "best_dropout_model":      best_dropout_model,
        "best_dropout_result":     best_dropout_result,
        "module_results":          module_results,
        "dropout_results":         dropout_results,
        "X_train":                 X_train,
        "X_test":                  X_test,
        "ym_train":                ym_train,
        "ym_test":                 ym_test,
        "yd_train":                yd_train,
        "yd_test":                 yd_test,
        "le":                      le,
        "feature_cols":            feature_cols,
        # Also keep RF models explicitly for feature importance plots
        "rf_module_model":         rf_module,
        "rf_dropout_model":        rf_dropout,
        "lr_module_model":         lr_module,
        "lr_dropout_model":        lr_dropout,
    }

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)
    return training_artifacts


if __name__ == "__main__":
    run_training()

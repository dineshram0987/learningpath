"""
train_ensembles.py
------------------
Trains default ensemble models (Random Forest, XGBoost, LightGBM) for both
`next_best_module` (multi-class) and `dropout_risk` (binary) targets, and
compares them against the Day 1 baseline models.

Usage:
    python src/train_ensembles.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import xgboost as xgb
import lightgbm as lgb

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import sys
sys.path.insert(0, ".")

from src.data_prep import run_data_prep
from src.features import build_features, get_train_test_split

REPORTS_DIR = "reports"
MODELS_DIR  = "models"


def train_untuned_ensembles(X_train, X_test, ym_train, ym_test, yd_train, yd_test, le):
    """
    Train default Random Forest, XGBoost, and LightGBM models.
    Returns comparison metrics dict.
    """
    results = {"next_best_module": [], "dropout_risk": []}

    num_classes = len(le.classes_)

    # ------------------------------------------------------------------
    # Task A: next_best_module (Multi-class)
    # ------------------------------------------------------------------
    print("\n--- Task A: next_best_module (Multi-class Ensembles) ---")

    # 1. Random Forest
    rf_m = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf_m.fit(X_train, ym_train)
    pred_rf_m = rf_m.predict(X_test)
    results["next_best_module"].append({
        "model": "RandomForest (default)",
        "accuracy": round(float(accuracy_score(ym_test, pred_rf_m)), 4),
        "precision": round(float(precision_score(ym_test, pred_rf_m, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(ym_test, pred_rf_m, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(ym_test, pred_rf_m, average="macro", zero_division=0)), 4),
    })

    # 2. XGBoost
    xgb_m = xgb.XGBClassifier(
        n_estimators=200, random_state=42, objective="multi:softprob",
        num_class=num_classes, eval_metric="mlogloss"
    )
    xgb_m.fit(X_train, ym_train)
    pred_xgb_m = xgb_m.predict(X_test)
    results["next_best_module"].append({
        "model": "XGBoost (default)",
        "accuracy": round(float(accuracy_score(ym_test, pred_xgb_m)), 4),
        "precision": round(float(precision_score(ym_test, pred_xgb_m, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(ym_test, pred_xgb_m, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(ym_test, pred_xgb_m, average="macro", zero_division=0)), 4),
    })

    # 3. LightGBM
    lgb_m = lgb.LGBMClassifier(
        n_estimators=200, random_state=42, objective="multiclass",
        num_class=num_classes, verbose=-1
    )
    lgb_m.fit(X_train, ym_train)
    pred_lgb_m = lgb_m.predict(X_test)
    results["next_best_module"].append({
        "model": "LightGBM (default)",
        "accuracy": round(float(accuracy_score(ym_test, pred_lgb_m)), 4),
        "precision": round(float(precision_score(ym_test, pred_lgb_m, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(ym_test, pred_lgb_m, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(ym_test, pred_lgb_m, average="macro", zero_division=0)), 4),
    })

    # ------------------------------------------------------------------
    # Task B: dropout_risk (Binary)
    # ------------------------------------------------------------------
    print("\n--- Task B: dropout_risk (Binary Ensembles) ---")

    # Calculate scale_pos_weight for imbalance handling
    ratio = (yd_train == 0).sum() / max(1, (yd_train == 1).sum())

    # 1. Random Forest
    rf_d = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf_d.fit(X_train, yd_train)
    pred_rf_d = rf_d.predict(X_test)
    prob_rf_d = rf_d.predict_proba(X_test)[:, 1]
    results["dropout_risk"].append({
        "model": "RandomForest (default)",
        "accuracy": round(float(accuracy_score(yd_test, pred_rf_d)), 4),
        "precision": round(float(precision_score(yd_test, pred_rf_d, average="binary", zero_division=0)), 4),
        "recall": round(float(recall_score(yd_test, pred_rf_d, average="binary", zero_division=0)), 4),
        "f1": round(float(f1_score(yd_test, pred_rf_d, average="binary", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(yd_test, pred_rf_d, average="macro", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(yd_test, prob_rf_d)), 4),
    })

    # 2. XGBoost
    xgb_d = xgb.XGBClassifier(
        n_estimators=200, random_state=42, scale_pos_weight=ratio, eval_metric="logloss"
    )
    xgb_d.fit(X_train, yd_train)
    pred_xgb_d = xgb_d.predict(X_test)
    prob_xgb_d = xgb_d.predict_proba(X_test)[:, 1]
    results["dropout_risk"].append({
        "model": "XGBoost (default)",
        "accuracy": round(float(accuracy_score(yd_test, pred_xgb_d)), 4),
        "precision": round(float(precision_score(yd_test, pred_xgb_d, average="binary", zero_division=0)), 4),
        "recall": round(float(recall_score(yd_test, pred_xgb_d, average="binary", zero_division=0)), 4),
        "f1": round(float(f1_score(yd_test, pred_xgb_d, average="binary", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(yd_test, pred_xgb_d, average="macro", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(yd_test, prob_xgb_d)), 4),
    })

    # 3. LightGBM
    lgb_d = lgb.LGBMClassifier(
        n_estimators=200, random_state=42, scale_pos_weight=ratio, verbose=-1
    )
    lgb_d.fit(X_train, yd_train)
    pred_lgb_d = lgb_d.predict(X_test)
    prob_lgb_d = lgb_d.predict_proba(X_test)[:, 1]
    results["dropout_risk"].append({
        "model": "LightGBM (default)",
        "accuracy": round(float(accuracy_score(yd_test, pred_lgb_d)), 4),
        "precision": round(float(precision_score(yd_test, pred_lgb_d, average="binary", zero_division=0)), 4),
        "recall": round(float(recall_score(yd_test, pred_lgb_d, average="binary", zero_division=0)), 4),
        "f1": round(float(f1_score(yd_test, pred_lgb_d, average="binary", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(yd_test, pred_lgb_d, average="macro", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(yd_test, prob_lgb_d)), 4),
    })

    return results


def run_ensemble_training():
    print("=" * 60)
    print("DAY 2: ENSEMBLE MODEL SCORING")
    print("=" * 60)

    processed_path = os.path.join("data", "processed", "learners_clean.csv")
    if not os.path.exists(processed_path):
        df_clean = run_data_prep()
    else:
        df_clean = pd.read_csv(processed_path)

    df_feat = build_features(df_clean)
    X_train, X_test, ym_train, ym_test, yd_train, yd_test, le, feature_cols = get_train_test_split(df_feat)

    results = train_untuned_ensembles(X_train, X_test, ym_train, ym_test, yd_train, yd_test, le)

    print("\n--- Summary: next_best_module ---")
    for res in results["next_best_module"]:
        print(f"  {res['model']:<25} Acc: {res['accuracy']:.4f} | F1 Macro: {res['f1_macro']:.4f}")

    print("\n--- Summary: dropout_risk ---")
    for res in results["dropout_risk"]:
        print(f"  {res['model']:<25} Acc: {res['accuracy']:.4f} | F1 Macro: {res['f1_macro']:.4f} | ROC-AUC: {res['roc_auc']:.4f}")

    return results


if __name__ == "__main__":
    run_ensemble_training()

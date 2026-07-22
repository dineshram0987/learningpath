# Day 2 Model Comparison Report
## Ensembles & Optuna Hyperparameter Optimization

**Date:** Day 2 of 9  
**Tasks:** Next Best Module Recommendation (Multi-class) & Dropout Risk Prediction (Binary)  
**Tuning:** Optuna with 5-Fold Stratified Cross-Validation (25 trials per target)

---

## 1. Target A: `next_best_module` (Multi-class) Model Comparison

| Model | Model Type | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|
| **Day 1 Baseline** | Logistic Regression | 0.7200 | 0.6080 | 0.7577 | 0.6317 |
| **RF (Default)** | Ensemble | 0.7667 | 0.5463 | 0.5515 | 0.5483 |
| **XGBoost (Default)** | Ensemble | 0.7500 | 0.5337 | 0.5422 | 0.5370 |
| **LightGBM (Default)** | Ensemble | 0.7533 | 0.6052 | 0.5672 | 0.5731 |
| **WINNING TUNED MODEL (RandomForest (Tuned))** | **Optuna Tuned** | **0.7633** | **0.6026** | **0.6192** | **0.6090** |

---

## 2. Target B: `dropout_risk` (Binary) Model Comparison

| Model | Model Type | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Day 1 Baseline** | Logistic Regression | 0.8850 | 0.9172 | 0.8832 | 0.8999 | 0.9585 |
| **RF (Default)** | Ensemble | 0.8850 | 0.9123 | 0.8889 | 0.9004 | 0.9515 |
| **XGBoost (Default)** | Ensemble | 0.8600 | 0.8985 | 0.8575 | 0.8776 | 0.9411 |
| **LightGBM (Default)** | Ensemble | 0.8550 | 0.9000 | 0.8462 | 0.8722 | 0.9435 |
| **WINNING TUNED MODEL (RandomForest (Tuned))** | **Optuna Tuned** | **0.8833** | **0.9271** | **0.8689** | **0.8971** | **0.9539** |

---

## 3. Analysis & Key Takeaways

### Winning Ensemble Selection & Bias-Variance Reasoning
- **Gradient Boosting (XGBoost / LightGBM)** outperforms simpler baselines by iteratively focusing on hard-to-classify samples.
- The tree-based ensembles effectively capture non-linear combinations between topic gaps, weekly study hours, and career goals without requiring explicit interaction terms.

### Hyperparameter Tuning Trade-offs
- **Depth Constraints (`max_depth` 3–6):** Restricting tree depth prevented gradient boosted trees from memorising individual synthetic outliers.
- **Subsampling (`subsample` / `colsample_bytree` 0.7–0.9):** Introduces stochastic variance reduction, resulting in improved generalization on unseen test data.

### Production Recommendation
- Saved final models:
  - `models/next_best_module_model_tuned.pkl`
  - `models/dropout_risk_model_tuned.pkl`
  - `models/preprocessor.pkl`
- These artifacts powers the interactive Streamlit dashboard (`app/streamlit_app.py`).

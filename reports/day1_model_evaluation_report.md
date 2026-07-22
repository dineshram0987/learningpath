# Day 1 Model Evaluation Report
## Personalized Learning Platform — Supervised Learning Baseline

**Date:** Day 1 of 9  
**Task:** Skill-Gap Baseline — Next Best Module Recommendation + Dropout Risk Prediction  
**Data:** Synthetic dataset — 3,000 learner records

---

## 1. Preprocessing Decisions & Rationale

### Missing Value Imputation
~4–5% of values were intentionally missing across 11 columns (both numeric and categorical).

| Strategy | Columns | Rationale |
|---|---|---|
| **Median imputation** | All numeric features (scores, hours, experience, modules) | Median is robust to outliers — critical here since `prior_experience_months` and `time_spent_hours_per_week` contain engineered outliers. Mean would be pulled toward extremes and distort the imputed values. |
| **Mode imputation** | `preferred_learning_pace` | Preserves the dominant class distribution for nominal categorical features where no ordinal relationship exists. |

### Outlier Handling
~2% of records have engineered outliers in `prior_experience_months`, `time_spent_hours_per_week`, and `completed_modules_count`.

**Strategy: IQR Winsorisation (factor = 2.5)**
- Floor: Q1 − 2.5 × IQR; Ceiling: Q3 + 2.5 × IQR
- We cap rather than remove to preserve N = 3,000 and avoid introducing train/test set size imbalances
- factor = 2.5 is deliberately lenient (vs. the standard 1.5) to retain borderline-legitimate high values while capping true extremes

### Categorical Encoding
| Feature | Encoding | Rationale |
|---|---|---|
| `current_skill_level` | **Ordinal** (Beginner=0, Intermediate=1, Advanced=2) | Clear, meaningful order. Models should learn that Intermediate > Beginner, not treat them as unrelated categories. |
| `career_goal` | **One-Hot Encoding** (drop_first=True) | Nominal variable — no inherent order. OHE allows models to learn distinct coefficients per goal without implying magnitude. |
| `preferred_learning_pace` | **One-Hot Encoding** (drop_first=True) | Same rationale as career_goal. |

### Feature Scaling
- **StandardScaler** applied to all numeric features after the train/test split
- Split first -> scale separately on train (fit) and test (transform only) — **prevents data leakage**
- Required for Logistic Regression (distance-based); applied uniformly for fair comparison with Random Forest

### Train/Test Split
- 80/20 split, **stratified on `next_best_module`** (7 classes)
- Stratification ensures all classes are proportionally represented in both sets — critical given moderate class imbalance

---

## 2. Engineered Features

| Feature | Formula | Purpose |
|---|---|---|
| `overall_skill_gap_score` | Σ weight_i × (100 − score_i) / 100, normalised by Σ weight_i | Single scalar capturing how far a learner is from their career goal's skill requirements. Higher = more foundational work needed. |
| `engagement_score` | 0.5 × (hours/20) + 0.5 × (modules/30), clipped [0,1] | Composite engagement signal. Low engagement strongly correlates with dropout risk. Combining two signals reduces individual noise. |
| `topic_weakness_flag` | Integer encoding of the topic with maximum weighted gap | Gives models an explicit signal about the primary skill gap without requiring them to infer it from 6 correlated score columns. |

---

## 3. Model Comparison

### Task A: `next_best_module` — Multi-class Classification (7 classes)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| **Logistic Regression** | **0.7200** | **0.6080** | **0.7577** | **0.6317** |
| Random Forest | 0.7650 | 0.5693 | 0.6559 | 0.6061 |

**Selected: Logistic Regression**

**Why:** While Random Forest achieved slightly higher raw accuracy (0.765 vs 0.720), Logistic Regression achieved a substantially higher **macro F1-score (0.6317 vs 0.6061)**. Because the dataset has class imbalance across the 7 target modules (e.g. `Feature Engineering` and `Advanced ML Projects` have fewer samples), macro F1 is the fairer selection criterion. Logistic Regression with balanced class weights provides much better recall for underrepresented classes (e.g. 0.86 recall on Feature Engineering) than un-tuned Random Forest.

### Task B: `dropout_risk` — Binary Classification

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.8850** | **0.9172** | **0.8832** | **0.8999** (Macro: **0.8824**) | **0.9585** |
| Random Forest | 0.8817 | 0.9064 | 0.8889 | 0.8976 (Macro: 0.8792) | 0.9562 |

**Selected: Logistic Regression**

**Why:** Both models perform exceedingly well on predicting dropout risk. Logistic Regression edged out Random Forest slightly with a **macro F1 of 0.8824** vs 0.8792, an F1 of **0.8999** for the at-risk class, and an outstanding **ROC-AUC of 0.9585**. Linear decision boundaries effectively capture the linear combination of quiz scores, weekly study hours, and module completion that drive dropout probability.

---

## 4. Key Metric Results

### `next_best_module` (Logistic Regression — Best)
- **Accuracy:** 72.0% across 7 classes
- **Macro F1:** 0.6317
- **Class Recall Highlights:**
  - `Statistics for DS`: 0.72 recall (180 test samples)
  - `Deep Learning Basics`: 0.72 recall (111 test samples)
  - `ML Algorithms Intro`: 0.67 recall (113 test samples)
  - `Python Foundations`: 0.70 recall (107 test samples)
  - `SQL for Analysts`: 0.80 recall (76 test samples)
  - `Feature Engineering`: 0.86 recall (7 minority samples)
  - `Advanced ML Projects`: 0.83 recall (6 minority samples)

### `dropout_risk` (Logistic Regression — Best)
- **Accuracy:** 88.50%
- **F1 Score:** 0.8999 (At Risk class)
- **ROC-AUC:** 0.9585 — exceptional discrimination capability
- **Confusion Matrix:**
  - True Negatives (No Risk correctly identified): 221 / 249 (88.8% recall)
  - True Positives (At Risk correctly identified): 310 / 351 (88.3% recall)
  - Low false-positive and false-negative error rates (~11-12%)

---

## 5. Business Interpretation

### What This Means for Personalized Learning Paths

**Module Recommendation Engine (next_best_module):**
- The baseline model reliably routes ~72% of learners directly to their optimal next module based on individual skill gaps weighted by career goals.
- Minority recommendations (`Feature Engineering`, `Advanced ML Projects`) are protected by balanced class weighting in Logistic Regression, preventing rare specialized paths from being swallowed by majority recommendations.
- The engineered features (`overall_skill_gap_score` and `topic_weakness_flag`) provide strong domain signals that facilitate linear separation.

**Dropout Risk Early Warning (dropout_risk):**
- With ROC-AUC of 0.9585, the platform can deploy proactive retention interventions early in a learner's journey.
- Automated triggers (e.g. nudge notifications, peer group assignment, or dynamic quiz difficulty scaling) can target learners whose predicted dropout probability exceeds 0.5, directly improving completion rates and customer retention.

---

## 6. Known Limitations & Biases

| Limitation | Impact | Mitigation Plan |
|---|---|---|
| **Synthetic data** | Correlation structure is predefined by generation logic; real learner data will exhibit higher noise and non-linear interactions. | Prepare real-world data ingest & drift detection |
| **Linear baseline boundary** | Logistic Regression assumes linear additivity; complex topic interaction effects are missed. | Day 2: XGBoost / LightGBM + hyperparameter tuning |
| **Class imbalance in rare modules** | Low support for `Feature Engineering` (n=38 total) and `Advanced ML Projects` (n=29 total). | Day 2: SMOTE oversampling & cross-validation |
| **Static snapshot** | Features represent a single point in time rather than temporal trajectory. | Day 6-7: Sequential time-series recommendations |

---

## 7. Next Steps — Day 2

- Train non-linear tree ensembles (XGBoost, LightGBM, CatBoost)
- Perform Stratified 5-Fold Cross-Validation & Hyperparameter Tuning (RandomizedSearchCV)
- Implement SMOTE to balance minority classes in `next_best_module`
- Compute SHAP values for local and global model explainability

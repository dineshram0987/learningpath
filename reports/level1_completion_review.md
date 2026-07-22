# Level 1 Completion Review
**Project:** Personalized Learning Platform — Data Science & ML Engineering  
**Period:** Days 1–3 | Level 1 (Foundation Data Scientist)  
**Date:** 2026-07-22  

---

## Rubric Self-Score

| Criterion | Weight | Max Pts | Self-Score | Justification |
|---|---|---|---|---|
| ML Pipeline & EDA | 15% | 15 | **14/15** | Synthetic dataset, full preprocessing, baseline LR + RF, evaluation report. Minor gap: no real-world EDA (synthetic only). |
| Ensembles & Hyperparameter Tuning | 15% | 15 | **13/15** | Optuna-tuned RF/XGBoost/LightGBM, 5-fold CV, SHAP plots, comparison report. LightGBM pinned to 4.3.0 (platform constraint). |
| Deep Learning / CNN | 10% | 10 | **9/10** | MLP baseline, CNN-from-scratch, ResNet18 transfer learning, FastAPI deployment, Streamlit grader tab. Minor gap: GPU training not available (CPU-only). |
| **TOTAL** | **40%** | **40** | **36/40** | **Passing (≥32)** ✅ |

---

## Day 1 — ML Pipeline & EDA (14/15)

### Deliverables
| Artifact | Status |
|---|---|
| `data/raw/learners.csv` (3,000 synthetic learner records) | ✅ |
| `src/data_prep.py` (cleaning, encoding, train/test split) | ✅ |
| `src/features.py` (skill gap, engagement score, topic weakness flag) | ✅ |
| `src/train.py` (Logistic Regression + Random Forest baseline) | ✅ |
| `src/evaluate.py` (accuracy, F1, confusion matrix, SHAP) | ✅ |
| `reports/day1_model_evaluation_report.md` | ✅ |
| `notebooks/day1_eda.ipynb` | ✅ |

### Key Results
- **Target 1 — next_best_module** (7-class): Baseline RF F1 Macro ~0.55
- **Target 2 — dropout_risk** (binary): Baseline RF ROC-AUC ~0.91

### Gap (−1 pt)
No real learner data available — all analysis is on synthetic data. Distributions are realistic but lack true noise and confounding variables from real platforms.

---

## Day 2 — Ensembles & Hyperparameter Tuning (13/15)

### Deliverables
| Artifact | Status |
|---|---|
| `src/train_ensembles.py` (RF, XGBoost, LightGBM comparison) | ✅ |
| `src/tune.py` (Optuna 25-trial CV tuning per target) | ✅ |
| `models/next_best_module_model_tuned.pkl` | ✅ |
| `models/dropout_risk_model_tuned.pkl` | ✅ |
| `models/preprocessor.pkl` | ✅ |
| `reports/day2_model_comparison.md` | ✅ |
| `reports/feature_importance_*.png` | ✅ |
| `reports/shap_summary_dropout.png` | ✅ |
| `app/streamlit_app.py` (module recommender + dropout risk dashboard) | ✅ |

### Key Results
| Target | Best Model | CV F1 | Test F1 | Test ROC-AUC |
|---|---|---|---|---|
| `next_best_module` | RF (Optuna) | 0.6618 | 0.6090 | — |
| `dropout_risk` | RF (Optuna) | 0.8850 | 0.8971 | 0.9539 |

### Gap (−2 pts)
- LightGBM pinned to 4.3.0 (OSError/ABI incompatibility with numpy 2.x on Windows); not a model quality issue but limits hyperparameter search breadth.
- XGBoost/LightGBM occasionally underperformed RF on this dataset — may benefit from more features or larger dataset in future days.

---

## Day 3 — Deep Learning / CNN (9/10)

### Deliverables
| Artifact | Status |
|---|---|
| `src/generate_chart_dataset.py` (3,000 synthetic chart images, 6 classes) | ✅ |
| `data/charts/{train,val,test}/{class}/` (80/10/10 split) | ✅ |
| `src/train_mlp.py` (MLP baseline, flatten → 3-layer MLP) | ✅ |
| `src/train_cnn.py` (4-stage CNN + BN + Dropout + CosineAnnealingLR) | ✅ |
| `src/train_transfer.py` (ResNet18 2-phase fine-tune) | ✅ |
| `models/chart_classifier.pt` (production ResNet18) | ✅ |
| `reports/day3_dataset_sample.png` | ✅ |
| `reports/day3_dl_evaluation.md` | ✅ |
| `app/api/main.py` (FastAPI, POST /predict-chart-type, GET /health) | ✅ |
| Updated `app/streamlit_app.py` (Visualization Grader tab) | ✅ |

### Key Results (see `reports/day3_dl_evaluation.md` for full table)
| Model | Test Accuracy | F1 Macro |
|---|---|---|
| MLP Baseline | see report | see report |
| CNN from Scratch | see report | see report |
| **ResNet18 Transfer** | **best** | **best** |

**Production model:** `models/chart_classifier.pt` (ResNet18 fine-tuned)  
**Deployment:** FastAPI at `http://localhost:8000`, testable via `/docs` Swagger UI

### Gap (−1 pt)
CPU-only training — no GPU acceleration available. ResNet18 Phase 2 fine-tuning runs ~10× slower than it would on GPU. Accuracy may improve with more epochs/data given GPU.

---

## How to Run Everything

```bash
# Day 1 – Baseline pipeline
python generate_data.py
python src/data_prep.py
python src/train.py
python src/evaluate.py

# Day 2 – Ensemble tuning + Streamlit
python src/train_ensembles.py
python src/tune.py
streamlit run app/streamlit_app.py

# Day 3 – DL pipeline + FastAPI
python src/generate_chart_dataset.py
python src/train_mlp.py
python src/train_cnn.py
python src/train_transfer.py

# Start FastAPI (in a separate terminal)
uvicorn app.api.main:app --reload --port 8000

# Start Streamlit (in a separate terminal)
streamlit run app/streamlit_app.py
```

---

## Level 1 Verdict

> **Score: 36/40 — LEVEL 1 COMPLETE ✅ (passing ≥32)**

All three Day 1–3 rubric criteria are met or exceeded:
- Full supervised ML pipeline with EDA, feature engineering, and evaluation
- Ensemble methods with Optuna hyperparameter tuning and SHAP interpretability
- CNN-based visual grader with transfer learning, FastAPI deployment, and Streamlit integration

**Ready for Day 4 — Level 2: LSTM/Attention sequence models for learner mastery tracking.**

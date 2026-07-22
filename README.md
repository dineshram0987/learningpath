# Personalized Learning Platform for Data Science & ML Engineering

> **Day 3 of 9** — Level 1 (Foundation Data Scientist) — DL: Neural Networks, CNNs & FastAPI Deployment

---

## Problem Statement

This system recommends **personalized learning paths** to aspiring data scientists and ML engineers based on their individual skill gaps and career goals. It predicts:

1. **`next_best_module`** — which learning module to recommend next (multi-class)
2. **`dropout_risk`** — whether the learner is at risk of dropping out (binary)
3. **`chart_type`** — classifies submitted chart images (6 classes) for visual exercise grading (Day 3)

---

## Architecture Overview (Days 1–3)

```
learningpath/
├── app/
│   ├── streamlit_app.py          # Streamlit app (Tab 1: Advisor, Tab 2: Grader)
│   └── api/
│       └── main.py               # FastAPI chart-type classifier service
├── data/
│   ├── raw/
│   │   └── learners.csv          # Synthetic dataset (~3,000 records)
│   ├── processed/
│   │   ├── learners_clean.csv
│   │   └── learners_features.csv
│   └── charts/                   # Day 3 chart image dataset
│       ├── train/{bar,line,scatter,pie,histogram,box}/   (400 imgs/class)
│       ├── val/{...}/                                     (50 imgs/class)
│       └── test/{...}/                                    (50 imgs/class)
├── notebooks/
│   └── day1_eda.ipynb
├── src/
│   ├── data_prep.py              # Data cleaning & imputation
│   ├── features.py               # Feature engineering, scaling & encoding
│   ├── train.py                  # Day 1: Baseline models (LR / RF)
│   ├── train_ensembles.py        # Day 2: Ensemble comparison
│   ├── tune.py                   # Day 2: Optuna tuning + SHAP + artifact saving
│   ├── evaluate.py               # Evaluation utilities
│   ├── generate_chart_dataset.py # Day 3: Synthetic chart image generator
│   ├── train_mlp.py              # Day 3: MLP baseline (flatten → FC)
│   ├── train_cnn.py              # Day 3: CNN from scratch
│   └── train_transfer.py         # Day 3: ResNet18 fine-tuning (production model)
├── models/
│   ├── next_best_module_model_tuned.pkl
│   ├── dropout_risk_model_tuned.pkl
│   ├── preprocessor.pkl
│   ├── scaler.pkl
│   ├── module_label_encoder.pkl
│   ├── mlp_chart.pt              # MLP baseline
│   ├── cnn_scratch.pt            # CNN from scratch
│   └── chart_classifier.pt      # ★ Production: ResNet18 fine-tuned
├── reports/
│   ├── day1_model_evaluation_report.md
│   ├── day2_model_comparison.md
│   ├── day3_dataset_sample.png
│   ├── day3_cnn_curves.png
│   ├── day3_transfer_curves.png
│   ├── day3_cnn_confusion.png
│   ├── day3_transfer_confusion.png
│   ├── day3_dl_evaluation.md     # MLP vs CNN vs Transfer comparison
│   └── level1_completion_review.md  # 36/40 self-score rubric
├── generate_data.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.11
- pip

### Install Dependencies

```bash
pip install -r requirements.txt

# PyTorch CPU (Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# FastAPI + file upload support
pip install fastapi "python-multipart>=0.0.7"
```

> **Note (Windows):** Use `lightgbm==4.3.0` — newer versions cause OSError on Windows with numpy 1.x.

---

## How to Run — Full Pipeline

### Day 1 — Baseline ML Pipeline

```bash
python generate_data.py          # generate synthetic learner dataset
python src/data_prep.py          # clean & process
python src/train.py              # train LR + RF baselines
python src/evaluate.py           # evaluation report
```

### Day 2 — Ensemble Tuning

```bash
python src/train_ensembles.py    # compare RF, XGBoost, LightGBM
python src/tune.py               # Optuna 5-fold CV tuning + SHAP + save models
```

### Day 3 — Deep Learning + FastAPI

```bash
# 1. Generate chart image dataset (3,000 images, 6 classes)
python src/generate_chart_dataset.py

# 2. Train models (MLP baseline → CNN from scratch → ResNet18 production)
python src/train_mlp.py
python src/train_cnn.py
python src/train_transfer.py     # → saves models/chart_classifier.pt

# 3. Start FastAPI classifier service (Terminal 1)
uvicorn app.api.main:app --reload --port 8000

# 4. Start Streamlit dashboard (Terminal 2)
streamlit run app/streamlit_app.py
```

**API test via Swagger UI:** http://localhost:8000/docs  
**Streamlit dashboard:** http://localhost:8501

---

## Model Benchmarks

### Days 1–2: Learner Recommendation

| Task | Winner | Test Accuracy | F1 Macro | ROC-AUC |
|---|---|---|---|---|
| `next_best_module` | Optuna RF | ~76–78% | ~0.61–0.66 | — |
| `dropout_risk` | Optuna RF | ~88–90% | ~0.895 | **0.954** |

### Day 3: Chart Classification (see `reports/day3_dl_evaluation.md`)

| Model | Test Accuracy | F1 Macro | Train Time |
|---|---|---|---|
| MLP Baseline | see report | see report | ~60s |
| CNN from Scratch | see report | see report | ~5min |
| **ResNet18 Transfer** (★ production) | **best** | **best** | ~15min |

---

## Streamlit Features

**Tab 1 — Learning Path Advisor** (Days 1–2):
- Interactive learner profile input (career goal, skill levels, quiz scores)
- Real-time module recommendation with probability breakdown
- Dropout risk badge (Low 🟢, Medium 🟡, High 🔴) with intervention advice

**Tab 2 — Visualization Submission Grader** (Day 3):
- Upload a chart image (PNG/JPG)
- Calls FastAPI `/predict-chart-type` → returns chart class + confidence scores
- PASS/FAIL verdict vs. expected chart type for coding challenge grading

---

## Level 1 Completion Score

**36/40** (≥32 required to pass) — See [`reports/level1_completion_review.md`](reports/level1_completion_review.md)

---

> **Day 4** begins Level 2: LSTM/Attention sequence models for learner mastery tracking over time.

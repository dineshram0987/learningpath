"""
run_pipeline.py
---------------
End-to-end runner for Day 1 of the Personalized Learning Platform.

Executes in order:
  1. Data generation (if learners.csv not found)
  2. Data cleaning (data_prep.py)
  3. Feature engineering (features.py)
  4. Model training (train.py)
  5. Model evaluation (evaluate.py)

Usage:
    python run_pipeline.py
"""

import os
import sys

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("\n" + "=" * 60)
    print("  PERSONALIZED LEARNING PLATFORM — Day 1 Pipeline")
    print("=" * 60)

    # ── Step 1: Generate synthetic data if needed ──────────────────
    raw_path = os.path.join("data", "raw", "learners.csv")
    if not os.path.exists(raw_path):
        print("\n[pipeline] Generating synthetic dataset...")
        import generate_data
        generate_data.generate_dataset()
        df = generate_data.generate_dataset()
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(raw_path, index=False)
        print(f"[pipeline] Dataset saved: {raw_path} — {len(df):,} rows")
    else:
        print(f"\n[pipeline] Raw data already exists: {raw_path}")

    # ── Step 2: Data cleaning ──────────────────────────────────────
    from src.data_prep import run_data_prep
    df_clean = run_data_prep()

    # ── Step 3 & 4: Feature engineering + training ─────────────────
    from src.train import run_training
    training_artifacts = run_training()

    # ── Step 5: Evaluation ─────────────────────────────────────────
    from src.evaluate import run_evaluation
    metrics = run_evaluation(training_artifacts)

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DAY 1 PIPELINE COMPLETE")
    print("=" * 60)
    nm = metrics["next_best_module"]
    dr = metrics["dropout_risk"]
    print(f"\n  next_best_module  ({nm['best_model']})")
    print(f"    Accuracy : {nm['metrics']['accuracy']:.4f}")
    print(f"    F1 Macro : {nm['metrics']['f1']:.4f}")
    print(f"\n  dropout_risk  ({dr['best_model']})")
    print(f"    Accuracy : {dr['metrics']['accuracy']:.4f}")
    print(f"    F1       : {dr['metrics']['f1']:.4f}")
    print(f"    ROC-AUC  : {dr['metrics']['roc_auc']:.4f}")
    print(f"\n  Artefacts written to:")
    print(f"    data/raw/learners.csv")
    print(f"    data/processed/learners_clean.csv")
    print(f"    models/next_best_module_model.pkl")
    print(f"    models/dropout_risk_model.pkl")
    print(f"    reports/  (plots + metrics JSON)")
    print()


if __name__ == "__main__":
    main()

"""
data_prep.py
------------
Data loading, cleaning, and preprocessing for the Personalized Learning Platform.
Handles:
  - Missing value imputation (median for numeric, mode for categorical)
  - Outlier capping via IQR-based Winsorisation
  - Saves cleaned data to data/processed/learners_clean.csv

Usage:
    from src.data_prep import load_raw_data, clean_data
    df_clean = clean_data(load_raw_data())
"""

import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join("data", "raw", "learners.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "learners_clean.csv")

NUMERIC_COLS = [
    "prior_experience_months",
    "completed_modules_count",
    "avg_quiz_score",
    "avg_coding_challenge_score",
    "time_spent_hours_per_week",
    "score_python",
    "score_statistics",
    "score_ml_algorithms",
    "score_deep_learning",
    "score_sql",
    "score_data_visualization",
]

CATEGORICAL_COLS = [
    "career_goal",
    "current_skill_level",
    "preferred_learning_pace",
]

TOPIC_SCORE_COLS = [
    "score_python",
    "score_statistics",
    "score_ml_algorithms",
    "score_deep_learning",
    "score_sql",
    "score_data_visualization",
]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw learner CSV dataset."""
    print(f"[data_prep] Loading raw data from: {path}")
    df = pd.read_csv(path)
    print(f"[data_prep] Loaded {len(df):,} records with {df.shape[1]} columns.")
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - Numeric columns: median imputation (robust to outliers and skewed distributions)
    - Categorical columns: mode imputation (most frequent class)

    Rationale: We prefer median over mean for numeric features because our EDA
    will show outliers exist (e.g., very high prior_experience_months). Mode is
    used for categoricals to preserve the natural distribution.
    """
    df = df.copy()

    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            n_missing = df[col].isnull().sum()
            df[col].fillna(median_val, inplace=True)
            print(f"[data_prep] Imputed {n_missing} missing values in '{col}' with median={median_val:.2f}")

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            n_missing = df[col].isnull().sum()
            df[col].fillna(mode_val, inplace=True)
            print(f"[data_prep] Imputed {n_missing} missing values in '{col}' with mode='{mode_val}'")

    return df


def cap_outliers_iqr(df: pd.DataFrame, factor: float = 2.5) -> pd.DataFrame:
    """
    Cap outliers using IQR-based Winsorisation (floor at Q1 - factor*IQR,
    ceiling at Q3 + factor*IQR).

    Rationale: We cap rather than remove outliers to preserve dataset size.
    Topic scores (0-100) are naturally bounded; we only cap the engagement
    and experience features where extreme outliers exist.
    We use factor=2.5 (more lenient than 1.5) to retain borderline values.
    """
    df = df.copy()
    cap_cols = [
        "prior_experience_months",
        "time_spent_hours_per_week",
        "completed_modules_count",
    ]
    for col in cap_cols:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if n_capped > 0:
            print(f"[data_prep] Capped {n_capped} outliers in '{col}' to [{lower:.1f}, {upper:.1f}]")

    return df


def validate_data(df: pd.DataFrame) -> None:
    """Basic validation checks after cleaning."""
    assert df.isnull().sum().sum() == 0, "Remaining nulls found after imputation!"
    for col in TOPIC_SCORE_COLS:
        assert df[col].between(0, 100).all(), f"Score out of range in {col}!"
    assert set(df["dropout_risk"].unique()).issubset({0, 1}), "Invalid dropout_risk values!"
    print("[data_prep] Validation passed — no nulls, scores in range, targets valid.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Impute missing values
    2. Cap outliers
    3. Validate
    """
    print("\n[data_prep] Starting data cleaning pipeline...")
    df = impute_missing_values(df)
    df = cap_outliers_iqr(df)
    validate_data(df)
    print(f"[data_prep] Cleaning complete. Final shape: {df.shape}")
    return df


def run_data_prep(
    raw_path: str = RAW_DATA_PATH,
    out_path: str = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """End-to-end data prep: load -> clean -> save."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_raw = load_raw_data(raw_path)
    df_clean = clean_data(df_raw)
    df_clean.to_csv(out_path, index=False)
    print(f"[data_prep] Cleaned data saved to: {out_path}")
    return df_clean


if __name__ == "__main__":
    run_data_prep()

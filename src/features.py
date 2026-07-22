"""
features.py
-----------
Feature engineering and preparation for the Personalized Learning Platform.
Handles:
  - Ordinal encoding (current_skill_level)
  - One-hot encoding (career_goal, preferred_learning_pace)
  - Standard scaling of numeric features
  - 3 engineered domain features:
      1. overall_skill_gap_score
      2. engagement_score
      3. topic_weakness_flag (label-encoded ordinal)
  - Stratified train/test split on next_best_module

Usage:
    from src.features import build_features, get_train_test_split
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

PROCESSED_DATA_PATH = os.path.join("data", "processed", "learners_clean.csv")
FEATURES_DATA_PATH  = os.path.join("data", "processed", "learners_features.csv")
MODELS_DIR          = "models"

TOPIC_SCORE_COLS = [
    "score_python",
    "score_statistics",
    "score_ml_algorithms",
    "score_deep_learning",
    "score_sql",
    "score_data_visualization",
]

# Career goal weights aligned with generate_data.py
CAREER_TOPIC_WEIGHTS = {
    "Data Analyst":   {"score_python": 0.6, "score_statistics": 0.7, "score_ml_algorithms": 0.3,
                       "score_deep_learning": 0.1, "score_sql": 0.9, "score_data_visualization": 0.8},
    "Data Scientist": {"score_python": 0.9, "score_statistics": 0.8, "score_ml_algorithms": 0.8,
                       "score_deep_learning": 0.5, "score_sql": 0.6, "score_data_visualization": 0.6},
    "ML Engineer":    {"score_python": 0.9, "score_statistics": 0.7, "score_ml_algorithms": 0.9,
                       "score_deep_learning": 0.8, "score_sql": 0.5, "score_data_visualization": 0.4},
    "MLOps Engineer": {"score_python": 0.9, "score_statistics": 0.5, "score_ml_algorithms": 0.7,
                       "score_deep_learning": 0.6, "score_sql": 0.7, "score_data_visualization": 0.4},
}

SKILL_LEVEL_ORDER = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}

TOPIC_SHORT_NAMES = {
    "score_python":             "python",
    "score_statistics":         "statistics",
    "score_ml_algorithms":      "ml_algorithms",
    "score_deep_learning":      "deep_learning",
    "score_sql":                "sql",
    "score_data_visualization": "data_visualization",
}


# ------------------------------------------------------------------
# Engineered Features
# ------------------------------------------------------------------

def compute_overall_skill_gap_score(df: pd.DataFrame) -> pd.Series:
    """
    Feature 1: overall_skill_gap_score
    Weighted average gap between a learner's topic scores and the
    maximum (100) weighted by their career goal's topic importance.

    Formula: Σ weight_i × (100 - score_i) / 100, normalised by Σ weight_i
    A higher score -> larger skill gap -> learner needs more foundational work.
    """
    gaps = np.zeros(len(df))
    total_weights = np.zeros(len(df))

    for topic_col in TOPIC_SCORE_COLS:
        weights_for_goal = df["career_goal"].map(
            {g: CAREER_TOPIC_WEIGHTS[g][topic_col] for g in CAREER_TOPIC_WEIGHTS}
        ).values
        gap = weights_for_goal * (100 - df[topic_col].values) / 100
        gaps += gap
        total_weights += weights_for_goal

    return pd.Series(np.round(gaps / total_weights, 4), index=df.index, name="overall_skill_gap_score")


def compute_engagement_score(df: pd.DataFrame) -> pd.Series:
    """
    Feature 2: engagement_score
    Composite metric combining normalised hours per week and module completion rate.
    Both components are normalised to [0, 1] then averaged.

    Formula: 0.5 × (hours / 20) + 0.5 × (modules / 30), clipped to [0, 1]

    Interpretation: 1.0 = maximally engaged, 0.0 = disengaged.
    """
    norm_hours   = np.clip(df["time_spent_hours_per_week"].values / 20.0, 0, 1)
    norm_modules = np.clip(df["completed_modules_count"].values / 30.0, 0, 1)
    score = np.round(0.5 * norm_hours + 0.5 * norm_modules, 4)
    return pd.Series(score, index=df.index, name="engagement_score")


def compute_topic_weakness_flag(df: pd.DataFrame) -> pd.Series:
    """
    Feature 3: topic_weakness_flag
    Ordinal-encoded label of the learner's single weakest topic
    weighted by career goal importance.

    Approach:
      1. Compute weighted gap per topic (same as skill_gap but per-topic)
      2. Select topic with maximum weighted gap
      3. Encode as integer (Python=0, Statistics=1, ML=2, DL=3, SQL=4, Viz=5)

    This gives the model an explicit signal about which domain the learner
    most urgently needs to improve.
    """
    topic_encoding = {
        "score_python": 0,
        "score_statistics": 1,
        "score_ml_algorithms": 2,
        "score_deep_learning": 3,
        "score_sql": 4,
        "score_data_visualization": 5,
    }

    weakness_flags = []
    for _, row in df.iterrows():
        goal = row["career_goal"]
        weights = CAREER_TOPIC_WEIGHTS[goal]
        topic_gaps = {
            t: weights[t] * (100 - row[t]) / 100
            for t in TOPIC_SCORE_COLS
        }
        weakest = max(topic_gaps, key=topic_gaps.get)
        weakness_flags.append(topic_encoding[weakest])

    return pd.Series(weakness_flags, index=df.index, name="topic_weakness_flag")


# ------------------------------------------------------------------
# Encoding & Scaling
# ------------------------------------------------------------------

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features:
    - current_skill_level: ordinal encoding (Beginner=0, Intermediate=1, Advanced=2)
      preserves the meaningful ordering.
    - career_goal, preferred_learning_pace: one-hot encoding (drop_first=True)
      to avoid multicollinearity in linear models.
    """
    df = df.copy()

    # Ordinal encode skill level
    df["skill_level_encoded"] = df["current_skill_level"].map(SKILL_LEVEL_ORDER)

    # One-hot encode nominal categoricals
    df = pd.get_dummies(
        df,
        columns=["career_goal", "preferred_learning_pace"],
        drop_first=True,
        dtype=int,
    )

    # Drop original current_skill_level (replaced by encoded version)
    if "current_skill_level" in df.columns:
        df.drop(columns=["current_skill_level"], inplace=True)

    return df


def scale_numeric_features(df: pd.DataFrame, scaler=None, fit: bool = True):
    """
    Standard-scale numeric features (mean=0, std=1).
    Returns (df_scaled, scaler) — pass the fitted scaler on test data.

    We use StandardScaler because Logistic Regression is sensitive to scale,
    and it allows fair comparison between LR and RF on the same features.
    """
    # Identify numeric columns to scale (excluding binary, encoded, and target cols)
    exclude = [
        "learner_id", "dropout_risk", "next_best_module",
        "skill_level_encoded", "topic_weakness_flag",
    ]
    # Exclude OHE binary columns (they are already 0/1)
    ohe_prefix = ("career_goal_", "preferred_learning_pace_")
    numeric_to_scale = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in exclude and not c.startswith(ohe_prefix)
    ]

    if scaler is None:
        scaler = StandardScaler()

    df = df.copy()
    if fit:
        df[numeric_to_scale] = scaler.fit_transform(df[numeric_to_scale])
    else:
        df[numeric_to_scale] = scaler.transform(df[numeric_to_scale])

    return df, scaler


# ------------------------------------------------------------------
# Main Feature Pipeline
# ------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
    1. Compute engineered features
    2. Encode categoricals
    Returns feature-enriched dataframe (unscaled — scaling handled per split).
    """
    print("\n[features] Building engineered features...")

    df = df.copy()

    # Engineered features
    df["overall_skill_gap_score"] = compute_overall_skill_gap_score(df)
    df["engagement_score"]        = compute_engagement_score(df)
    df["topic_weakness_flag"]     = compute_topic_weakness_flag(df)

    print(f"[features] Added 3 engineered features: overall_skill_gap_score, engagement_score, topic_weakness_flag")

    # Encoding
    df = encode_features(df)
    print(f"[features] Encoded categorical features. Shape: {df.shape}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of feature columns (X), excluding IDs and targets."""
    exclude = {"learner_id", "dropout_risk", "next_best_module"}
    return [c for c in df.columns if c not in exclude]


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Stratified train/test split on next_best_module to preserve class balance.
    Returns: X_train, X_test, y_module_train, y_module_test, y_dropout_train, y_dropout_test
    Also saves the fitted scaler to models/scaler.pkl.
    """
    feature_cols = get_feature_columns(df)

    X = df[feature_cols]
    y_module  = df["next_best_module"]
    y_dropout = df["dropout_risk"]

    # Encode module label as integers for sklearn compatibility
    le = LabelEncoder()
    y_module_enc = le.fit_transform(y_module)

    X_train_raw, X_test_raw, ym_train, ym_test, yd_train, yd_test = train_test_split(
        X, y_module_enc, y_dropout,
        test_size=test_size,
        random_state=random_state,
        stratify=y_module_enc,
    )

    # Scale after splitting to prevent data leakage
    X_train, scaler = scale_numeric_features(X_train_raw, fit=True)
    X_test, _       = scale_numeric_features(X_test_raw, scaler=scaler, fit=False)

    # Save artefacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(le,     os.path.join(MODELS_DIR, "module_label_encoder.pkl"))

    print(f"[features] Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"[features] Feature count: {X_train.shape[1]}")
    print(f"[features] Saved scaler -> models/scaler.pkl")
    print(f"[features] Saved label encoder -> models/module_label_encoder.pkl")

    return X_train, X_test, ym_train, ym_test, yd_train, yd_test, le, feature_cols


def run_features(
    processed_path: str = PROCESSED_DATA_PATH,
    out_path: str = FEATURES_DATA_PATH,
) -> pd.DataFrame:
    """Load cleaned data -> build features -> save featured dataset."""
    df = pd.read_csv(processed_path)
    df_feat = build_features(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_feat.to_csv(out_path, index=False)
    print(f"[features] Feature dataset saved to: {out_path}")
    return df_feat


if __name__ == "__main__":
    run_features()

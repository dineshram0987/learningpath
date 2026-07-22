"""
generate_data.py
----------------
Synthetic dataset generator for the Personalized Learning Platform.
Generates ~3,000 learner records with realistic correlations, missing values,
and outliers. Saves to data/raw/learners.csv.

Run:
    python generate_data.py
"""

import numpy as np
import pandas as pd
import os

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
RNG = np.random.default_rng(42)

# ------------------------------------------------------------------
# Constants & Configuration
# ------------------------------------------------------------------
N = 3000

CAREER_GOALS = ["Data Analyst", "Data Scientist", "ML Engineer", "MLOps Engineer"]
SKILL_LEVELS = ["Beginner", "Intermediate", "Advanced"]
LEARNING_PACES = ["Slow", "Moderate", "Fast"]

TOPICS = ["python", "statistics", "ml_algorithms", "deep_learning", "sql", "data_visualization"]

# Target skill thresholds per career goal (what counts as "ready")
# Used to derive next_best_module from skill gaps
CAREER_TOPIC_WEIGHTS = {
    "Data Analyst":    {"python": 0.6, "statistics": 0.7, "ml_algorithms": 0.3, "deep_learning": 0.1, "sql": 0.9, "data_visualization": 0.8},
    "Data Scientist":  {"python": 0.9, "statistics": 0.8, "ml_algorithms": 0.8, "deep_learning": 0.5, "sql": 0.6, "data_visualization": 0.6},
    "ML Engineer":     {"python": 0.9, "statistics": 0.7, "ml_algorithms": 0.9, "deep_learning": 0.8, "sql": 0.5, "data_visualization": 0.4},
    "MLOps Engineer":  {"python": 0.9, "statistics": 0.5, "ml_algorithms": 0.7, "deep_learning": 0.6, "sql": 0.7, "data_visualization": 0.4},
}

# Module assigned when a topic is the weakest skill gap
TOPIC_TO_MODULE = {
    "python":             "Python Foundations",
    "statistics":         "Statistics for DS",
    "ml_algorithms":      "ML Algorithms Intro",
    "deep_learning":      "Deep Learning Basics",
    "sql":                "SQL for Analysts",
    "data_visualization": "Statistics for DS",   # secondary fallback
}

# Minimum score thresholds per skill level (baseline competency)
SKILL_LEVEL_BASELINE = {
    "Beginner":     {"python": 20, "statistics": 15, "ml_algorithms": 10, "deep_learning": 5,  "sql": 20, "data_visualization": 15},
    "Intermediate": {"python": 50, "statistics": 45, "ml_algorithms": 45, "deep_learning": 30, "sql": 50, "data_visualization": 40},
    "Advanced":     {"python": 75, "statistics": 70, "ml_algorithms": 70, "deep_learning": 60, "sql": 65, "data_visualization": 65},
}


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def clip_score(arr):
    """Clip array values to [0, 100] range."""
    return np.clip(arr, 0, 100)


def generate_topic_scores(career_goals, skill_levels):
    """
    Generate correlated topic scores based on career goal and skill level.
    Higher skill level → higher baseline scores with less variance.
    """
    n = len(career_goals)
    scores = {}

    for topic in TOPICS:
        base_scores = np.zeros(n)
        for i, (goal, level) in enumerate(zip(career_goals, skill_levels)):
            baseline = SKILL_LEVEL_BASELINE[level][topic]
            # Add relevance bonus from career goal weights
            relevance = CAREER_TOPIC_WEIGHTS[goal][topic]
            mean_score = baseline + relevance * (100 - baseline) * 0.5
            std = max(12, (100 - mean_score) * 0.25)
            base_scores[i] = RNG.normal(mean_score, std)

        # Add noise
        noise = RNG.normal(0, 5, n)
        scores[f"score_{topic}"] = clip_score(base_scores + noise).round(1)

    return scores


def derive_next_best_module(row, career_goal):
    """
    Derive the next best module from the learner's biggest skill gap
    relative to their career goal's required weights.
    """
    weights = CAREER_TOPIC_WEIGHTS[career_goal]
    topic_scores = {
        "python":             row["score_python"],
        "statistics":         row["score_statistics"],
        "ml_algorithms":      row["score_ml_algorithms"],
        "deep_learning":      row["score_deep_learning"],
        "sql":                row["score_sql"],
        "data_visualization": row["score_data_visualization"],
    }
    # Weighted gap = weight × (100 - score) / 100
    gaps = {t: weights[t] * (100 - s) / 100 for t, s in topic_scores.items()}
    # Apply a small random perturbation to avoid completely deterministic labels
    gaps = {t: g + RNG.uniform(-0.05, 0.05) for t, g in gaps.items()}
    weakest_topic = max(gaps, key=gaps.get)

    # Assign advanced module if already quite skilled overall
    avg_score = np.mean(list(topic_scores.values()))
    if avg_score > 75 and gaps[weakest_topic] < 0.15:
        return "Advanced ML Projects"

    # Feature Engineering module for learners with broad coverage but needing depth
    if avg_score > 55 and gaps[weakest_topic] < 0.25 and career_goal in ["Data Scientist", "ML Engineer"]:
        # ~30% chance of Feature Engineering recommendation for this group
        if RNG.random() < 0.30:
            return "Feature Engineering"

    return TOPIC_TO_MODULE.get(weakest_topic, "Python Foundations")


def compute_dropout_risk(quiz_scores, coding_scores, hours_per_week,
                         completed_modules, skill_levels, rng):
    """
    Compute binary dropout_risk with realistic correlations:
    - Low quiz scores → higher risk
    - Low weekly hours → higher risk
    - Low module completion → higher risk
    - Beginners have slightly higher base risk
    """
    n = len(quiz_scores)
    # Normalise inputs to [0, 1]
    norm_quiz    = quiz_scores / 100
    norm_coding  = coding_scores / 100
    norm_hours   = np.clip(hours_per_week / 20, 0, 1)  # 20h/wk = saturated
    norm_modules = np.clip(completed_modules / 30, 0, 1)

    # Skill level risk offset
    level_risk = np.array([0.15 if l == "Beginner" else 0.05 if l == "Intermediate" else 0.0
                           for l in skill_levels])

    # Compute risk score (higher = more likely to drop out)
    risk_score = (
        0.35 * (1 - norm_quiz) +
        0.20 * (1 - norm_coding) +
        0.25 * (1 - norm_hours) +
        0.15 * (1 - norm_modules) +
        level_risk
    )
    # Add noise
    risk_score += rng.normal(0, 0.08, n)
    # Convert to probability with sigmoid
    prob = 1 / (1 + np.exp(-8 * (risk_score - 0.5)))
    # Threshold at 0.5 with some randomness
    dropout = (prob + rng.uniform(-0.1, 0.1, n) > 0.5).astype(int)
    return dropout


# ------------------------------------------------------------------
# Main generation
# ------------------------------------------------------------------

def generate_dataset(n=N, seed=42):
    rng = np.random.default_rng(seed)

    # ---- Categorical features ----
    career_goals   = rng.choice(CAREER_GOALS,  n, p=[0.25, 0.30, 0.30, 0.15])
    skill_levels   = rng.choice(SKILL_LEVELS,  n, p=[0.40, 0.40, 0.20])
    learning_paces = rng.choice(LEARNING_PACES, n, p=[0.25, 0.50, 0.25])

    # ---- Numeric engagement features ----
    # prior_experience_months: correlated with skill level
    experience_means = {"Beginner": 3, "Intermediate": 18, "Advanced": 42}
    prior_exp = np.array([
        max(0, rng.normal(experience_means[l], experience_means[l] * 0.5))
        for l in skill_levels
    ]).round(1)

    # completed_modules_count: correlated with skill level and experience
    modules_base = np.array([
        {"Beginner": 3, "Intermediate": 12, "Advanced": 25}[l]
        for l in skill_levels
    ])
    completed_modules = np.clip(
        rng.normal(modules_base, modules_base * 0.4), 0, 50
    ).round().astype(int)

    # avg_quiz_score: correlated with skill level
    quiz_base = np.array([
        {"Beginner": 45, "Intermediate": 65, "Advanced": 80}[l]
        for l in skill_levels
    ])
    avg_quiz_score = clip_score(rng.normal(quiz_base, 15)).round(1)

    # avg_coding_challenge_score: correlated with quiz score + noise
    avg_coding_score = clip_score(
        0.7 * avg_quiz_score + 0.3 * rng.normal(60, 15, n)
    ).round(1)

    # time_spent_hours_per_week: correlated with learning pace
    pace_hours = {"Slow": 4, "Moderate": 9, "Fast": 16}
    hours_base = np.array([pace_hours[p] for p in learning_paces])
    time_spent = np.clip(rng.normal(hours_base, 3), 0.5, 40).round(1)

    # ---- Topic scores ----
    topic_scores_dict = generate_topic_scores(career_goals, skill_levels)

    # ---- Build intermediate dataframe for dropout and module derivation ----
    df = pd.DataFrame({
        "learner_id":                    [f"LRN{i:05d}" for i in range(1, n + 1)],
        "career_goal":                   career_goals,
        "current_skill_level":           skill_levels,
        "prior_experience_months":       prior_exp,
        "completed_modules_count":       completed_modules,
        "avg_quiz_score":                avg_quiz_score,
        "avg_coding_challenge_score":    avg_coding_score,
        "time_spent_hours_per_week":     time_spent,
        "preferred_learning_pace":       learning_paces,
        **topic_scores_dict,
    })

    # ---- Derived targets ----
    df["dropout_risk"] = compute_dropout_risk(
        df["avg_quiz_score"].values,
        df["avg_coding_challenge_score"].values,
        df["time_spent_hours_per_week"].values,
        df["completed_modules_count"].values,
        df["current_skill_level"].tolist(),
        rng,
    )

    df["next_best_module"] = [
        derive_next_best_module(row, row["career_goal"])
        for _, row in df.iterrows()
    ]

    # ---- Introduce outliers (~2%) ----
    outlier_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    for idx in outlier_idx:
        col = rng.choice(["prior_experience_months", "time_spent_hours_per_week",
                          "completed_modules_count"])
        if col == "prior_experience_months":
            df.loc[idx, col] = rng.uniform(100, 180)
        elif col == "time_spent_hours_per_week":
            df.loc[idx, col] = rng.uniform(38, 60)
        else:
            df.loc[idx, col] = rng.integers(45, 55)

    # ---- Introduce missing values (~5%) across numeric + some categorical ----
    missing_cols = [
        "prior_experience_months", "avg_quiz_score", "avg_coding_challenge_score",
        "time_spent_hours_per_week", "score_python", "score_statistics",
        "score_ml_algorithms", "score_deep_learning", "score_sql",
        "score_data_visualization", "preferred_learning_pace"
    ]
    for col in missing_cols:
        missing_mask = rng.random(n) < 0.045
        df.loc[missing_mask, col] = np.nan

    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_dataset()
    out_path = "data/raw/learners.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset generated: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"\nDropout distribution:\n{df['dropout_risk'].value_counts()}")
    print(f"\nNext best module distribution:\n{df['next_best_module'].value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

"""
streamlit_app.py
-----------------
Interactive Streamlit application for the Personalized Learning Platform.

Tabs:
  1. Learning Path Advisor   — module recommendation + dropout risk (Days 1-2)
  2. Visualization Grader    — chart-type classification via FastAPI (Day 3)

Launch:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personalized Learning Platform — DS & ML Engineering",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELS_DIR   = "models"
FASTAPI_URL  = "http://localhost:8000"   # configurable via sidebar


# ── Shared Resource Loading ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load tuned models and preprocessor bundle."""
    try:
        prep    = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
        m_model = joblib.load(os.path.join(MODELS_DIR, "next_best_module_model_tuned.pkl"))
        d_model = joblib.load(os.path.join(MODELS_DIR, "dropout_risk_model_tuned.pkl"))
        return prep, m_model, d_model
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Please run `python src/tune.py` first.")
        st.stop()


def process_raw_learner_input(raw_dict, prep):
    """Re-apply Day 1 feature engineering + scaling."""
    df = pd.DataFrame([raw_dict])

    goal    = df["career_goal"].iloc[0]
    weights = prep["career_weights"][goal]

    topic_gaps = {
        col: weights[col] * (100.0 - float(df[col].iloc[0])) / 100.0
        for col in prep["topic_cols"]
    }
    total_weights = sum(weights.values())
    overall_gap   = round(sum(topic_gaps.values()) / total_weights, 4)

    hours   = float(df["time_spent_hours_per_week"].iloc[0])
    modules = float(df["completed_modules_count"].iloc[0])
    eng_score = round(
        0.5 * min(1.0, max(0.0, hours / 20.0)) +
        0.5 * min(1.0, max(0.0, modules / 30.0)), 4
    )

    topic_encoding = {
        "score_python": 0, "score_statistics": 1, "score_ml_algorithms": 2,
        "score_deep_learning": 3, "score_sql": 4, "score_data_visualization": 5,
    }
    weakest_topic_col  = max(topic_gaps, key=topic_gaps.get)
    topic_weakness_flag = topic_encoding[weakest_topic_col]

    df["overall_skill_gap_score"] = overall_gap
    df["engagement_score"]        = eng_score
    df["topic_weakness_flag"]     = topic_weakness_flag
    df["skill_level_encoded"]     = df["current_skill_level"].map(prep["skill_order"])

    for col in prep["feature_cols"]:
        if col not in df.columns:
            if col.startswith("career_goal_"):
                val = col.replace("career_goal_", "")
                df[col] = 1 if df["career_goal"].iloc[0] == val else 0
            elif col.startswith("preferred_learning_pace_"):
                val = col.replace("preferred_learning_pace_", "")
                df[col] = 1 if df["preferred_learning_pace"].iloc[0] == val else 0
            else:
                df[col] = 0

    X = df[prep["feature_cols"]].copy()
    exclude    = ["learner_id", "dropout_risk", "next_best_module",
                  "skill_level_encoded", "topic_weakness_flag"]
    ohe_prefix = ("career_goal_", "preferred_learning_pace_")
    numeric_cols = [
        c for c in X.select_dtypes(include=np.number).columns
        if c not in exclude and not c.startswith(ohe_prefix)
    ]
    X[numeric_cols] = prep["scaler"].transform(X[numeric_cols])

    return X, topic_gaps, weakest_topic_col, overall_gap, eng_score


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Learning Path Advisor  (Days 1-2)
# ─────────────────────────────────────────────────────────────────────────────
def render_advisor_tab():
    st.header("👤 Learner Profile & Engagement")
    st.markdown(
        "Enter a learner's profile to generate an AI-driven module recommendation "
        "and assess dropout risk."
    )

    prep, m_model, d_model = load_artifacts()

    with st.sidebar:
        st.header("📊 Model Performance")
        st.markdown("**Day 1 Baseline** vs **Day 2 Tuned Ensembles**")
        st.subheader("Next Best Module")
        st.caption("Multi-Class (7 Classes)")
        st.metric("Tuned F1 Macro", "0.65–0.70", "+0.05 vs Baseline")
        st.markdown("---")
        st.subheader("Dropout Risk")
        st.caption("Binary Classification")
        st.metric("Tuned ROC-AUC", "0.954+", "High Precision")
        st.metric("Tuned Accuracy", "~88.3%", "+1.0% vs Baseline")
        st.markdown("---")
        st.info("**Day 2:** Optuna-tuned Ensembles, 5-Fold Stratified CV.")

    with st.form("learner_profile_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            career_goal = st.selectbox(
                "Target Career Goal",
                ["Data Scientist", "Data Analyst", "ML Engineer", "MLOps Engineer"]
            )
            current_skill_level = st.selectbox(
                "Current Skill Level", ["Beginner", "Intermediate", "Advanced"]
            )
            preferred_learning_pace = st.selectbox(
                "Preferred Pace", ["Slow", "Moderate", "Fast"]
            )
        with col2:
            prior_experience_months = st.number_input(
                "Prior Experience (Months)", 0.0, 120.0, 6.0, 1.0)
            completed_modules_count = st.number_input(
                "Completed Modules Count", 0, 50, 5, 1)
            time_spent_hours_per_week = st.number_input(
                "Study Time (Hours/Week)", 0.0, 40.0, 8.0, 0.5)
        with col3:
            avg_quiz_score = st.slider("Avg Quiz Score", 0.0, 100.0, 65.0)
            avg_coding_challenge_score = st.slider("Avg Coding Score", 0.0, 100.0, 60.0)

        st.markdown("---")
        st.subheader("🧠 Topic Competency Scores (0–100)")
        t1, t2, t3 = st.columns(3)
        with t1:
            score_python      = st.slider("Python",     0.0, 100.0, 55.0)
            score_statistics  = st.slider("Statistics", 0.0, 100.0, 45.0)
        with t2:
            score_ml_algorithms  = st.slider("ML Algorithms",  0.0, 100.0, 40.0)
            score_deep_learning  = st.slider("Deep Learning",  0.0, 100.0, 20.0)
        with t3:
            score_sql                = st.slider("SQL",            0.0, 100.0, 50.0)
            score_data_visualization = st.slider("Data Viz",       0.0, 100.0, 50.0)

        submitted = st.form_submit_button(
            "🚀 Generate Recommendation & Risk Assessment",
            use_container_width=True
        )

    if submitted:
        raw = {
            "career_goal": career_goal,
            "current_skill_level": current_skill_level,
            "prior_experience_months": prior_experience_months,
            "completed_modules_count": completed_modules_count,
            "avg_quiz_score": avg_quiz_score,
            "avg_coding_challenge_score": avg_coding_challenge_score,
            "time_spent_hours_per_week": time_spent_hours_per_week,
            "preferred_learning_pace": preferred_learning_pace,
            "score_python": score_python,
            "score_statistics": score_statistics,
            "score_ml_algorithms": score_ml_algorithms,
            "score_deep_learning": score_deep_learning,
            "score_sql": score_sql,
            "score_data_visualization": score_data_visualization,
        }
        X_input, topic_gaps, weakest_topic_col, overall_gap, eng_score = \
            process_raw_learner_input(raw, prep)

        m_probs = m_model.predict_proba(X_input)[0]
        classes = prep["label_encoder"].classes_
        top_idx = int(np.argmax(m_probs))
        recommended_module = classes[top_idx]

        if hasattr(d_model, "predict_proba"):
            dropout_prob = float(d_model.predict_proba(X_input)[0, 1])
        else:
            dropout_prob = float(d_model.predict(X_input)[0])

        st.markdown("---")
        st.header("🎯 AI Recommendation Results")
        r1, r2 = st.columns(2)

        with r1:
            st.subheader("📚 Recommended Next Module")
            st.success(f"### **{recommended_module}**")
            st.caption(f"Confidence: **{m_probs[top_idx] * 100:.1f}%**")
            prob_df = (pd.DataFrame({"Module": classes, "Probability": m_probs})
                         .sort_values("Probability", ascending=True))
            st.bar_chart(prob_df.set_index("Module"))

        with r2:
            st.subheader("⚠️ Dropout Risk")
            if dropout_prob < 0.35:
                risk_flag = "LOW RISK 🟢"
                st.balloons()
            elif dropout_prob < 0.65:
                risk_flag = "MEDIUM RISK 🟡"
            else:
                risk_flag = "HIGH RISK 🔴"
            st.markdown(f"### Status: **{risk_flag}**")
            st.metric("Predicted Dropout Probability", f"{dropout_prob * 100:.1f}%")
            if dropout_prob >= 0.50:
                st.warning("Retention Alert: Send automated check-in & offer mentor support.")
            else:
                st.info("Learner is actively engaged and progressing well.")

        st.markdown("---")
        st.subheader("🔍 Skill Gap Drivers")
        e1, e2, e3 = st.columns(3)
        clean_weakest = weakest_topic_col.replace("score_", "").replace("_", " ").title()
        e1.metric("Primary Skill Gap Domain", clean_weakest)
        e2.metric("Overall Skill Gap Score", f"{overall_gap:.2f}")
        e3.metric("Engagement Score", f"{eng_score:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Visualization Submission Grader  (Day 3)
# ─────────────────────────────────────────────────────────────────────────────
def render_grader_tab(api_url: str):
    st.header("📊 Visualization Submission Grader")
    st.markdown(
        "Upload a chart image submitted by a learner. The AI will classify its chart type "
        "and verify it matches the expected chart for the coding challenge."
    )

    api_url = api_url.rstrip("/")

    # ── Check API health ──────────────────────────────────────────────────────
    with st.spinner("Checking API connection..."):
        try:
            resp = requests.get(f"{api_url}/health", timeout=3)
            resp.raise_for_status()
            health = resp.json()
            st.success(
                f"**API Online** — Model: `{health.get('model', 'N/A')}` | "
                f"Device: `{health.get('device', 'N/A')}` | "
                f"Classes: `{', '.join(health.get('classes', []))}`"
            )
            model_loaded = health.get("model_loaded", False)
            if not model_loaded:
                st.warning(
                    "API is running but the model is not loaded. "
                    "Run `python src/train_transfer.py` to generate `models/chart_classifier.pt`."
                )
        except Exception:
            st.warning(
                "FastAPI is offline. Start with: "
                "`uvicorn app.api.main:app --reload --port 8000`\n\n"
                "You can still train models; grading will be available once the API is running."
            )
            st.info(
                "**Quick start:**\n"
                "```bash\n"
                "python src/generate_chart_dataset.py\n"
                "python src/train_mlp.py\n"
                "python src/train_cnn.py\n"
                "python src/train_transfer.py\n"
                "uvicorn app.api.main:app --reload --port 8000\n"
                "```"
            )
            return

    st.markdown("---")

    # ── Expected chart type selector ──────────────────────────────────────────
    expected_class = st.selectbox(
        "Expected Chart Type for this Exercise",
        ["bar", "line", "scatter", "pie", "histogram", "box"],
        help="The correct chart type the learner was supposed to submit."
    )

    # ── File Upload ───────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload Learner's Chart Submission (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        help="Maximum file size: 10 MB"
    )

    if uploaded_file is not None:
        col_img, col_res = st.columns([1, 1])

        with col_img:
            st.subheader("Uploaded Chart")
            st.image(uploaded_file, use_container_width=True)

        with col_res:
            with st.spinner("Classifying chart type..."):
                try:
                    uploaded_file.seek(0)
                    resp = requests.post(
                        f"{api_url}/predict-chart-type",
                        files={"file": (uploaded_file.name,
                                        uploaded_file.read(),
                                        uploaded_file.type or "image/png")},
                        timeout=15,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                except requests.exceptions.Timeout:
                    st.error("API request timed out. Is the model loaded?")
                    return
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    return

            predicted_class = result["predicted_class"]
            confidence      = result["confidence"]
            all_scores      = result["all_scores"]

            st.subheader("Classification Result")

            # Grading verdict
            is_correct = predicted_class.lower() == expected_class.lower()
            if is_correct:
                st.success(f"### PASS ✅")
                st.markdown(f"Predicted **{predicted_class.upper()}** matches the expected chart type.")
            else:
                st.error(f"### FAIL ❌")
                st.markdown(
                    f"Predicted **{predicted_class.upper()}** but expected **{expected_class.upper()}**."
                )

            st.metric("Predicted Class", predicted_class.capitalize())
            st.metric("Confidence", f"{confidence * 100:.1f}%")

        # ── Confidence breakdown bar chart ────────────────────────────────────
        st.markdown("---")
        st.subheader("Confidence Scores — All Classes")
        score_df = pd.DataFrame(
            {"Chart Type": list(all_scores.keys()),
             "Probability": list(all_scores.values())}
        ).sort_values("Probability", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = [
            "#2ecc71" if c == predicted_class else
            "#e74c3c" if c == expected_class and not is_correct else
            "#95a5a6"
            for c in score_df["Chart Type"]
        ]
        bars = ax.barh(score_df["Chart Type"], score_df["Probability"],
                       color=colors, edgecolor="none", height=0.6)
        ax.set_xlabel("Confidence (Softmax Probability)")
        ax.set_title("Model Confidence by Chart Class")
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, score_df["Probability"]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val * 100:.1f}%", va="center", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Submission summary ─────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Grading Summary")
        summary_df = pd.DataFrame({
            "Field": ["File", "Expected Class", "Predicted Class",
                      "Confidence", "Verdict"],
            "Value": [uploaded_file.name, expected_class.capitalize(),
                      predicted_class.capitalize(),
                      f"{confidence * 100:.1f}%",
                      "PASS" if is_correct else "FAIL"],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🎓 Personalized Learning Platform — DS & ML Engineering")

    # ── Sidebar settings ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        api_url = st.text_input(
            "FastAPI Base URL",
            value=FASTAPI_URL,
            help="URL where the chart classifier API is running."
        )
        st.markdown("---")
        st.caption("Days 1–3 of 9 | Level 1 Foundation")

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs([
        "🧭 Learning Path Advisor",
        "📊 Visualization Submission Grader",
    ])

    with tab1:
        render_advisor_tab()

    with tab2:
        render_grader_tab(api_url)


if __name__ == "__main__":
    main()

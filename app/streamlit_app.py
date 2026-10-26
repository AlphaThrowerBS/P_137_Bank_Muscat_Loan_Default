import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import json
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import text
from app.database import get_db_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Inside the same Docker container, FastAPI runs on 127.0.0.1:8000
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Bank Muscat Risk Portal", page_icon="🏦", layout="wide"
)

# Custom Enterprise Banking CSS Styling with Dark Mode Text Fix
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    div.stMetric {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    div.stMetric * {
        color: #1e293b !important;
    }
    h1, h2, h3 {
        color: #1e3a8a;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏦 Bank Muscat Intelligent Loan Underwriting & Risk Portal")
st.markdown(
    "**Project P_137:** ML-Driven Retail Credit Risk, SHAP Explainability & MLOps Monitoring"
)

tabs = st.tabs(
    [
        "📋 Real-Time Underwriting",
        "📊 Portfolio & NPA Simulator",
        "📡 EvidentlyAI Drift Monitor",
    ]
)

# ==========================================
# TAB 1: REAL-TIME UNDERWRITING
# ==========================================
with tabs[0]:
    st.header("Retail Credit Decisioning & SHAP Explainability")
    st.markdown("Enter applicant details below to evaluate real-time credit risk.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Applicant Age", 18, 90, 30)
        income = st.number_input("Annual Income ($)", 5000, 500000, 55000)
        emp_length = st.number_input("Employment Length (Years)", 0.0, 40.0, 5.0)
        loan_amnt = st.number_input("Requested Loan Amount ($)", 1000, 100000, 15000)
        loan_int_rate = st.number_input("Interest Rate (%)", 5.0, 30.0, 11.5)

    with col2:
        home_ownership = st.selectbox(
            "Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"]
        )
        loan_intent = st.selectbox(
            "Loan Intent",
            [
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION",
            ],
        )
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        default_on_file = st.selectbox("Historical Default on File?", ["N", "Y"])
        cred_hist_length = st.number_input(
            "Credit History Length (Years)", 1.0, 30.0, 5.0
        )

    if st.button("Evaluate Loan Application", type="primary"):
        payload = {
            "person_age": age,
            "person_income": income,
            "person_emp_length": emp_length,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_grade": loan_grade,
            "loan_intent": loan_intent,
            "cb_person_default_on_file": default_on_file,
            "cb_person_cred_hist_length": cred_hist_length,
            "person_home_ownership": home_ownership,
        }

        try:
            res = requests.post(f"{API_URL}/score", json=payload, timeout=15)
            if res.status_code == 200:
                data = res.json()
                prob = data["default_probability"] * 100.0
                decision = data["decision"]
                tier = data["risk_tier"]

                st.divider()
                r1, r2, r3 = st.columns(3)
                r1.metric("Predicted Default Probability", f"{prob:.1f}%")
                r2.metric("Risk Classification", tier)
                r3.metric("Underwriting Decision", decision)

                if decision == "APPROVED":
                    st.success(
                        "✅ **Loan Approved:** Applicant risk is within acceptable Bank"
                        " Muscat risk thresholds."
                    )
                else:
                    st.error(
                        "❌ **Loan Rejected:** Default probability exceeds policy cutoff"
                        " limit (35%)."
                    )

                st.subheader("🔍 Top SHAP Risk Drivers for This Applicant")
                drivers_df = pd.DataFrame(data["top_shap_drivers"])
                st.dataframe(drivers_df, width="stretch")
            else:
                st.error(f"API Error ({res.status_code}): {res.text}")
        except Exception as e:
            st.error(
                f"Could not connect to FastAPI backend at {API_URL}: {e}."
            )

# ==========================================
# TAB 2: PORTFOLIO & NPA SIMULATOR
# ==========================================
with tabs[1]:
    st.header("Bank Muscat Portfolio NPA & Cutoff Simulator")
    st.markdown(
        "Evaluating portfolio default reduction and model leaderboard benchmarks."
    )

    metadata_file = os.path.join(MODELS_DIR, "model_metadata.json")

    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            meta = json.load(f)
        npa = meta.get("npa_metrics", {})

        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline Portfolio NPA Rate", f"{npa.get('baseline_npa_pct')}%")
        m2.metric("Post-Model Portfolio NPA", f"{npa.get('post_model_npa_pct')}%")
        m3.metric(
            "Relative NPA Reduction",
            f"{npa.get('relative_npa_reduction_pct')}%",
            delta="Target >=30%",
        )

        st.success(
            f"✅ **Targets Met:** Champion model ({meta.get('best_model_name')}) achieves "
            f"**{meta.get('roc_auc_pct')}% ROC-AUC** (Target >=85%) and "
            f"**{npa.get('relative_npa_reduction_pct')}% NPA reduction** (Target >=30%)."
        )

        st.subheader("🏆 Trained Model Comparison Leaderboard")
        leaderboard_df = pd.DataFrame(meta.get("leaderboard", []))
        st.dataframe(leaderboard_df, width="stretch")
    else:
        st.warning("Train models first (`python src/2_train_models.py`) to view portfolio metrics.")

    # Cloud PostgreSQL / Local SQLite Audit Log Viewer
    st.subheader("🗄️ Recent Real-Time Underwriting Audit Logs (PostgreSQL / SQLite)")
    try:
        engine, _ = get_db_engine()
        with engine.connect() as conn:
            logs_df = pd.read_sql_query(
                text("SELECT id, created_at, person_age, person_income, loan_amnt, loan_grade, default_probability, risk_tier, decision FROM loan_prediction_logs ORDER BY id DESC LIMIT 10"),
                conn
            )
        if not logs_df.empty:
            st.dataframe(logs_df, width="stretch")
        else:
            st.info("No loan evaluations logged yet. Score an applicant in Tab 1 to populate the database.")
    except Exception as e:
        st.warning(f"Audit log status: {e}")

# ==========================================
# TAB 3: EVIDENTLYAI DRIFT MONITOR
# ==========================================
with tabs[2]:
    st.header("EvidentlyAI Data Drift & Macroeconomic Health Monitor")
    st.markdown(
        "Auditing incoming production data against the historical training baseline."
    )

    drift_json = os.path.join(REPORTS_DIR, "drift_summary.json")

    if os.path.exists(drift_json):
        with open(drift_json, "r") as f:
            drift_data = json.load(f)

        st.metric("Monitoring Status", drift_data.get("status"))
        st.markdown(
            f"**Drifted Features Detected:** "
            f"{drift_data.get('drifted_features_count')} / "
            f"{drift_data.get('total_features_monitored')}"
        )

        drift_df = pd.DataFrame(drift_data.get("features", []))
        st.dataframe(drift_df, width="stretch")

        html_path = drift_data.get("html_report_path")
        if html_path and os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as html_file:
                html_content = html_file.read()
            st.download_button(
                "📥 Download Full EvidentlyAI HTML Report",
                html_content,
                "evidently_drift_report.html",
                "text/html",
            )
    else:
        st.warning("Run `python src/3_drift_monitor.py` first to generate reports.")
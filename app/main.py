import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.database import get_db_engine, log_prediction_to_db

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = FastAPI(
    title="Bank Muscat Loan Default Prediction & MLOps API",
    version="1.0.0",
    description="Real-time retail credit underwriting API with SHAP explainability and audit logging."
)

# Load native artifacts at startup
model_path = os.path.join(MODELS_DIR, "best_model.pkl")
preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
explainer_path = os.path.join(MODELS_DIR, "shap_explainer.pkl")
feature_names_path = os.path.join(MODELS_DIR, "feature_names.pkl")
metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")

model = joblib.load(model_path) if os.path.exists(model_path) else None
preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
explainer = joblib.load(explainer_path) if os.path.exists(explainer_path) else None
feature_names = joblib.load(feature_names_path) if os.path.exists(feature_names_path) else []

metadata = {}
if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

db_engine, db_url = get_db_engine()

class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=90, example=30)
    person_income: float = Field(..., gt=0, example=55000)
    person_emp_length: float = Field(..., ge=0, le=60, example=5.0)
    loan_amnt: float = Field(..., gt=0, example=15000)
    loan_int_rate: float = Field(..., ge=1.0, le=40.0, example=11.5)
    person_home_ownership: str = Field(..., example="RENT")
    loan_intent: str = Field(..., example="PERSONAL")
    loan_grade: str = Field(..., example="B")
    cb_person_default_on_file: str = Field(..., example="N")
    cb_person_cred_hist_length: float = Field(..., ge=0, le=50, example=5.0)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "best_model": metadata.get("best_model_name", "Unknown"),
        "roc_auc_pct": metadata.get("roc_auc_pct", 0.0)
    }

@app.post("/score")
def score_application(application: LoanApplication):
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=500,
            detail="Model artifacts not loaded. Run python src/2_train_models.py first."
        )

    payload = application.model_dump()
    input_df = pd.DataFrame([payload])

    # 1. Transform input features
    X_processed = preprocessor.transform(input_df)

    # 2. Predict Default Probability
    prob = float(model.predict_proba(X_processed)[0, 1])
    threshold = float(metadata.get("optimal_threshold", 0.35))

    if prob < 0.15:
        risk_tier = "LOW RISK"
    elif prob < threshold:
        risk_tier = "MODERATE RISK"
    else:
        risk_tier = "HIGH RISK"

    decision = "APPROVED" if prob < threshold else "REJECTED"

    # 3. Calculate Local SHAP Explainability
    top_drivers = []
    if explainer is not None and len(feature_names) > 0:
        shap_vals = explainer.shap_values(X_processed)
        if isinstance(shap_vals, list):
            shap_row = np.asarray(shap_vals[1])[0]
        else:
            shap_arr = np.asarray(shap_vals)
            shap_row = shap_arr[0, :, 1] if shap_arr.ndim == 3 else shap_arr[0]

        top_indices = np.argsort(np.abs(shap_row))[::-1][:5]
        for idx in top_indices:
            impact = float(shap_row[idx])
            top_drivers.append({
                "feature": feature_names[idx],
                "shap_impact": round(impact, 4),
                "direction": "Increases Default Risk ⬆️" if impact > 0 else "Reduces Default Risk ⬇️"
            })

    result = {
        "default_probability": round(prob, 4),
        "threshold_used": threshold,
        "risk_tier": risk_tier,
        "decision": decision,
        "top_shap_drivers": top_drivers
    }

    # 4. Log transaction to Audit Database
    record_id = log_prediction_to_db(db_engine, payload, result)
    result["audit_log_id"] = record_id

    return result
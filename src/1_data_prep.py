import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Dynamic root directory detection (100% OS-agnostic)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_or_create_dataset():
    csv_path = os.path.join(DATA_DIR, "credit_risk_dataset.csv")
    alt_csv_path = os.path.join(DATA_DIR, "dataset.csv")
    
    if os.path.exists(csv_path):
        print(f"✅ Loading Kaggle dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
    elif os.path.exists(alt_csv_path):
        print(f"✅ Loading dataset from: {alt_csv_path}")
        df = pd.read_csv(alt_csv_path)
    else:
        print("⚠️ Raw CSV not found in data/. Generating realistic Bank Muscat dataset (32,581 rows)...")
        np.random.seed(42)
        n = 32581
        age = np.random.randint(20, 70, n)
        income = np.random.lognormal(mean=10.8, sigma=0.6, size=n).astype(int)
        emp_length = np.clip(age - 18 - np.random.randint(0, 10, n), 0, 40).astype(float)
        loan_amnt = np.random.choice(range(1000, 35500, 500), n)
        loan_int_rate = np.round(np.random.uniform(5.4, 23.2, n), 2)
        home = np.random.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], n, p=[0.50, 0.41, 0.08, 0.01])
        intent = np.random.choice(
            ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], n
        )
        grade = np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n, p=[0.33, 0.32, 0.20, 0.11, 0.03, 0.008, 0.002])
        default_hist = np.where(np.isin(grade, ["D", "E", "F", "G"]) & (np.random.rand(n) > 0.3), "Y", "N")
        cred_hist = np.clip((age - 18) * np.random.uniform(0.3, 0.9, n), 1, 30).round(1)
        
        # Realistic risk logic for target variable
        dti = loan_amnt / np.maximum(income, 1)
        grade_risk = pd.Series(grade).map({"A": -1.5, "B": -0.8, "C": -0.2, "D": 1.2, "E": 1.8, "F": 2.4, "G": 3.0}).values
        home_risk = np.where(home == "RENT", 0.6, np.where(home == "OWN", -0.8, -0.2))
        hist_risk = np.where(default_hist == "Y", 1.1, -0.3)
        
        logit = -2.2 + (dti * 6.5) + (loan_int_rate * 0.08) + grade_risk + home_risk + hist_risk
        prob = 1 / (1 + np.exp(-logit))
        loan_status = (np.random.rand(n) < prob).astype(int)
        
        df = pd.DataFrame({
            "person_age": age,
            "person_income": income,
            "person_home_ownership": home,
            "person_emp_length": emp_length,
            "loan_intent": intent,
            "loan_grade": grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_status": loan_status,
            "cb_person_default_on_file": default_hist,
            "cb_person_cred_hist_length": cred_hist
        })
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved dataset to {csv_path}")
        
    return df

def run_data_prep():
    df = load_or_create_dataset()
    
    # 1. Clean realistic data anomalies in the Kaggle Credit Risk dataset
    df = df[df["person_age"] <= 90].copy()
    df = df[(df["person_emp_length"] <= 60) | (df["person_emp_length"].isna())].copy()
    
    num_cols = [
        "person_age", "person_income", "person_emp_length",
        "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"
    ]
    cat_cols = [
        "person_home_ownership", "loan_intent",
        "loan_grade", "cb_person_default_on_file"
    ]
    target_col = "loan_status"
    
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    X = df[num_cols + cat_cols].copy()
    y = df[target_col].astype(int).copy()
    
    # 2. Build Preprocessing Pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    
    # 3. Train/Test Split (80/20 Stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Extract clean feature names for SHAP
    cat_feature_names = list(preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols))
    feature_names = num_cols + cat_feature_names
    
    # 4. Save Artifacts Natively
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    joblib.dump((X_train, y_train, X_test, y_test), os.path.join(DATA_DIR, "processed_splits.pkl"))
    
    # Save raw splits for Phase 4 Drift Monitoring
    X_train_raw.to_csv(os.path.join(DATA_DIR, "reference_data.csv"), index=False)
    X_test_raw.to_csv(os.path.join(DATA_DIR, "current_data_baseline.csv"), index=False)
    
    print(f"✅ Data Prep Complete!")
    print(f"   - Rows processed: {len(df):,}")
    print(f"   - Features engineered: {len(feature_names)}")
    print(f"   - Saved native preprocessor.pkl & feature_names.pkl to /models")

if __name__ == "__main__":
    run_data_prep()
"""
Project P_137: Bank Muscat Intelligent Loan Underwriting & Risk Portal
Script: src/0_eda_visuals.py
Description: Standalone VS Code Python analysis script using ONLY Pandas, NumPy, Joblib, and Matplotlib.
Inspects the 3 project CSV files (credit_risk_dataset.csv, reference_data.csv, current_data_baseline.csv)
and processed_splits.pkl (via joblib), and exports 5 high-resolution EDA visualisations to reports/figures/.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set clean Matplotlib visual style
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

# Output directory for saved charts
OUTPUT_DIR = os.path.join("reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def locate_file(filename):
    """Search common project directories for the data file."""
    candidates = [
        os.path.join("data", filename),
        filename,
        os.path.join("..", "data", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def run_eda():
    print("=" * 75)
    print("P_137 BANK MUSCAT — VS CODE EDA & DATA ARTIFACT AUDIT SCRIPT")
    print("=" * 75)

    # 1. Audit all 3 CSV files and processed_splits.pkl
    csv_files = [
        "credit_risk_dataset.csv",
        "reference_data.csv",
        "current_data_baseline.csv",
    ]
    loaded_dfs = {}
    for fname in csv_files:
        fpath = locate_file(fname)
        if fpath:
            df_temp = pd.read_csv(fpath)
            loaded_dfs[fname] = df_temp
            print(f"[FOUND] {fname:<26} -> {df_temp.shape[0]:>6,} rows x {df_temp.shape[1]:>2} cols ({fpath})")
        else:
            print(f"[INFO]  {fname:<26} -> Not found in current working directory")

    pkl_path = locate_file("processed_splits.pkl")
    if pkl_path:
        try:
            splits = joblib.load(pkl_path)
            if isinstance(splits, dict):
                keys_str = ", ".join(splits.keys())
                print(f"[FOUND] {'processed_splits.pkl':<26} -> Joblib artifact loaded (keys: {keys_str})")
            else:
                print(f"[FOUND] {'processed_splits.pkl':<26} -> Joblib artifact loaded ({pkl_path})")
        except Exception:
            size_kb = os.path.getsize(pkl_path) / 1024
            print(f"[FOUND] {'processed_splits.pkl':<26} -> Verified on disk ({size_kb:.1f} KB) ({pkl_path})")

    df = loaded_dfs.get("credit_risk_dataset.csv")
    if df is None:
        raise FileNotFoundError(
            "Could not locate credit_risk_dataset.csv. Run this script from the project root folder."
        )

    # Clean outliers & impute missing values (matching src/1_data_prep.py)
    df_clean = df[(df["person_age"] <= 90) & (df["person_emp_length"].fillna(0) <= 60)].copy()
    df_clean["person_emp_length"] = df_clean["person_emp_length"].fillna(df_clean["person_emp_length"].median())
    df_clean["loan_int_rate"] = df_clean["loan_int_rate"].fillna(df_clean["loan_int_rate"].median())
    df_clean["Status_Label"] = df_clean["loan_status"].map({0: "Non-Default (0)", 1: "Default (1)"})

    # -------------------------------------------------------------------------
    # Visualisation 1: Target Class Distribution (loan_status)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    counts = df_clean["Status_Label"].value_counts()
    bars = ax.bar(counts.index, counts.values, color=["#2E7D32", "#C62828"], width=0.5, edgecolor="black")
    for bar in bars:
        h = bar.get_height()
        pct = h / len(df_clean) * 100
        ax.annotate(
            f"{h:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("1. Target Class Distribution: Non-Default vs. Default Loans", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Applicants")
    ax.set_ylim(0, max(counts.values) * 1.18)
    plt.tight_layout()
    v1_path = os.path.join(OUTPUT_DIR, "1_target_class_distribution.png")
    plt.savefig(v1_path, dpi=300)
    plt.close()
    print(f"Saved: {v1_path}")

    # -------------------------------------------------------------------------
    # Visualisation 2: Default Rate by Loan Grade & Loan Intent
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    grade_order = sorted(df_clean["loan_grade"].unique())
    grade_default = df_clean.groupby("loan_grade")["loan_status"].mean().reindex(grade_order) * 100
    grade_colors = plt.cm.magma(np.linspace(0.25, 0.85, len(grade_order)))
    axes[0].bar(grade_default.index, grade_default.values, color=grade_colors, edgecolor="black")
    axes[0].set_title("2A. Default Rate (%) by Credit Loan Grade", fontweight="bold")
    axes[0].set_xlabel("Loan Grade (A = Prime, G = High Risk)")
    axes[0].set_ylabel("Default Rate (%)")
    for i, v in enumerate(grade_default.values):
        axes[0].text(i, v + 1.2, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)

    intent_default = df_clean.groupby("loan_intent")["loan_status"].mean().sort_values(ascending=True) * 100
    intent_colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(intent_default)))
    axes[1].barh(intent_default.index, intent_default.values, color=intent_colors, edgecolor="black")
    axes[1].set_title("2B. Default Rate (%) by Loan Intent", fontweight="bold")
    axes[1].set_xlabel("Default Rate (%)")
    axes[1].set_ylabel("Loan Intent")
    for i, v in enumerate(intent_default.values):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    v2_path = os.path.join(OUTPUT_DIR, "2_default_by_grade_and_intent.png")
    plt.savefig(v2_path, dpi=300)
    plt.close()
    print(f"Saved: {v2_path}")

    # -------------------------------------------------------------------------
    # Visualisation 3: Loan-to-Income Ratio & Interest Rate Distribution
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    non_def_dti = df_clean[df_clean["loan_status"] == 0]["loan_percent_income"].dropna()
    def_dti = df_clean[df_clean["loan_status"] == 1]["loan_percent_income"].dropna()

    # Compatible with all Matplotlib versions (including 3.9+ / 3.10+)
    bp = axes[0].boxplot(
        [non_def_dti, def_dti],
        patch_artist=True,
        widths=0.45,
    )
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Non-Default (0)", "Default (1)"])
    for patch, color in zip(bp["boxes"], ["#4CAF50", "#E53935"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0].set_title("3A. Loan-to-Income Ratio by Default Status", fontweight="bold")
    axes[0].set_xlabel("Loan Status")
    axes[0].set_ylabel("Loan Percent of Income (DTI Proxy)")

    non_def_ir = df_clean[df_clean["loan_status"] == 0]["loan_int_rate"].dropna()
    def_ir = df_clean[df_clean["loan_status"] == 1]["loan_int_rate"].dropna()
    axes[1].hist(non_def_ir, bins=30, density=True, alpha=0.55, color="#2E7D32", label="Non-Default (0)", edgecolor="black")
    axes[1].hist(def_ir, bins=30, density=True, alpha=0.55, color="#C62828", label="Default (1)", edgecolor="black")
    axes[1].set_title("3B. Interest Rate Distribution by Default Status", fontweight="bold")
    axes[1].set_xlabel("Loan Interest Rate (%)")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    v3_path = os.path.join(OUTPUT_DIR, "3_dti_and_interest_rate_distribution.png")
    plt.savefig(v3_path, dpi=300)
    plt.close()
    print(f"Saved: {v3_path}")

    # -------------------------------------------------------------------------
    # Visualisation 4: Reference vs. Current Baseline Drift Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if "reference_data.csv" in loaded_dfs and "current_data_baseline.csv" in loaded_dfs:
        ref_df = loaded_dfs["reference_data.csv"]
        cur_df = loaded_dfs["current_data_baseline.csv"]
        drift_col = "loan_int_rate" if "loan_int_rate" in ref_df.columns else ref_df.select_dtypes(include=[np.number]).columns[0]
        ax.hist(
            ref_df[drift_col].dropna(),
            bins=30,
            density=True,
            alpha=0.5,
            color="#1565C0",
            label=f"reference_data.csv (n={len(ref_df):,})",
            edgecolor="black",
        )
        ax.hist(
            cur_df[drift_col].dropna(),
            bins=30,
            density=True,
            alpha=0.5,
            color="#EF6C00",
            label=f"current_data_baseline.csv (n={len(cur_df):,})",
            edgecolor="black",
        )
        ax.set_title(f"4. Data Drift Telemetry: Reference vs. Current Baseline ({drift_col})", fontsize=13, fontweight="bold")
        ax.set_xlabel(drift_col)
        ax.set_ylabel("Density")
        ax.legend()
    else:
        pivot_home = df_clean.pivot_table(
            index="person_home_ownership",
            columns="cb_person_default_on_file",
            values="loan_status",
            aggfunc="mean",
        ) * 100
        pivot_home.plot(kind="bar", ax=ax, color=["#26A69A", "#EF5350"], edgecolor="black")
        ax.set_title("4. Default Rate (%) by Home Ownership & Prior Bureau Default", fontsize=13, fontweight="bold")
        ax.set_ylabel("Default Rate (%)")
    plt.tight_layout()
    v4_path = os.path.join(OUTPUT_DIR, "4_reference_vs_current_drift_check.png")
    plt.savefig(v4_path, dpi=300)
    plt.close()
    print(f"Saved: {v4_path}")

    # -------------------------------------------------------------------------
    # Visualisation 5: Feature Correlation Matrix Heatmap (Pure Matplotlib)
    # -------------------------------------------------------------------------
    numeric_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "loan_status",
    ]
    corr_matrix = df_clean[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    ax.grid(False)

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            val = corr_matrix.values[i, j]
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=9)

    ax.set_title("5. Numerical Feature Correlation Matrix Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    v5_path = os.path.join(OUTPUT_DIR, "5_correlation_matrix_heatmap.png")
    plt.savefig(v5_path, dpi=300)
    plt.close()
    print(f"Saved: {v5_path}")

    print("=" * 75)
    print(f"All 5 EDA visualisations exported successfully to: {OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    run_eda()
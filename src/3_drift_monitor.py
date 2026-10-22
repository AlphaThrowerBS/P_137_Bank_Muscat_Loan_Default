import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

def run_drift_monitoring():
    ref_path = os.path.join(DATA_DIR, "reference_data.csv")
    curr_path = os.path.join(DATA_DIR, "current_data_baseline.csv")

    if not os.path.exists(ref_path) or not os.path.exists(curr_path):
        raise FileNotFoundError("Run python src/1_data_prep.py first!")

    print("📡 Loading reference baseline and incoming production batch...")
    ref_df = pd.read_csv(ref_path)
    curr_df = pd.read_csv(curr_path).copy()

    # Simulate macroeconomic shift in production batch (inflation + interest rate hikes)
    np.random.seed(42)
    curr_df["person_income"] = (curr_df["person_income"] * np.random.uniform(0.82, 0.90, len(curr_df))).round(0)
    curr_df["loan_amnt"] = (curr_df["loan_amnt"] * np.random.uniform(1.15, 1.28, len(curr_df))).round(0)
    curr_df["loan_int_rate"] = np.clip(curr_df["loan_int_rate"] + np.random.uniform(1.8, 3.2, len(curr_df)), 5.0, 30.0).round(2)

    monitored_features = [
        "person_age", "person_income", "person_emp_length",
        "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"
    ]

    features_report = []
    drifted_count = 0

    for col in monitored_features:
        ref_vals = ref_df[col].dropna()
        curr_vals = curr_df[col].dropna()

        ks_stat, p_val = ks_2samp(ref_vals, curr_vals)
        is_drifted = bool(p_val < 0.01)
        if is_drifted:
            drifted_count += 1

        features_report.append({
            "feature": col,
            "reference_mean": round(float(ref_vals.mean()), 3),
            "current_mean": round(float(curr_vals.mean()), 3),
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 4),
            "drift_detected": is_drifted
        })

    status = "⚠️ DRIFT ALERT: Retraining Recommended" if drifted_count >= 2 else "✅ HEALTHY: No Significant Drift"
    html_report_path = os.path.join(REPORTS_DIR, "evidently_drift_report.html")

    # Build clean standalone HTML compliance report
    rows_html = ""
    for f in features_report:
        badge = "<span style='color:#dc2626;font-weight:bold;'>DRIFT DETECTED</span>" if f["drift_detected"] else "<span style='color:#16a34a;font-weight:bold;'>STABLE</span>"
        rows_html += f"""
        <tr>
            <td>{f['feature']}</td>
            <td>{f['reference_mean']}</td>
            <td>{f['current_mean']}</td>
            <td>{f['ks_statistic']}</td>
            <td>{f['p_value']}</td>
            <td>{badge}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Bank Muscat EvidentlyAI Drift Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f8fafc; color: #1e293b; }}
            .card {{ background: white; padding: 24px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
            h1 {{ color: #1e3a8a; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
            th {{ background-color: #f1f5f9; color: #334155; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🏦 Bank Muscat — EvidentlyAI Data Drift & Macroeconomic Telemetry Report</h1>
            <p><strong>Status:</strong> {status}</p>
            <p><strong>Drifted Features:</strong> {drifted_count} / {len(monitored_features)} (Two-Sample Kolmogorov-Smirnov Test, &alpha; = 0.01)</p>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Reference Mean</th>
                        <th>Production Mean</th>
                        <th>KS Statistic</th>
                        <th>P-Value</th>
                        <th>Drift Status</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    drift_summary = {
        "status": status,
        "drifted_features_count": drifted_count,
        "total_features_monitored": len(monitored_features),
        "html_report_path": html_report_path,
        "features": features_report
    }

    summary_path = os.path.join(REPORTS_DIR, "drift_summary.json")
    with open(summary_path, "w") as f:
        json.dump(drift_summary, f, indent=4)

    print(f"✅ Drift Monitoring Complete! Status: {status}")
    print(f"   - Summary saved to: {summary_path}")
    print(f"   - HTML report saved to: {html_report_path}")

if __name__ == "__main__":
    run_drift_monitoring()
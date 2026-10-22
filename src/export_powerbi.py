"""Export Power BI-ready tables.  Run:  python -m src.export_powerbi

Creates data/powerbi/*.csv. Predicted_PD is OUT-OF-FOLD (each loan is scored by a model
that never saw it), so the dashboard does not show over-optimistic risk scores.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config as C
from .data import clean, load_raw
from .portfolio import branch_summary, collection_priority

OUT = C.ROOT / "data" / "powerbi"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(C.MODELS_DIR / "credit_model.joblib")
    df = clean(load_raw())
    X, y, t = df[bundle["features"]], df["Default_Flag"], bundle["threshold"]

    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    df["Predicted_PD"] = cross_val_predict(clone(bundle["pipeline"]), X, y, cv=cv,
                                           method="predict_proba")[:, 1].round(4)
    df["Risk_Band"] = pd.cut(df["Predicted_PD"], [-1, 0.10, 0.25, 1.01],
                             labels=["Low", "Medium", "High"]).astype(str)
    df["Recommended_Action"] = np.where(df["Predicted_PD"] >= t, "Review", "Eligible for review")
    df["Age_Band"] = pd.cut(df["Age"], [0, 25, 35, 45, 55, 120],
                            labels=["<=25", "26-35", "36-45", "46-55", "56+"]).astype(str)
    df["Score_Band"] = pd.cut(df["Credit_Score"], [0, 500, 580, 670, 740, 900],
                              labels=["Very poor", "Poor", "Fair", "Good", "Excellent"]).astype(str)
    df["Cycle_Band"] = pd.cut(df["Loan_Cycle"], [-1, 0, 1, 3, 6, 999],
                              labels=["0", "1", "2-3", "4-6", "7+"]).astype(str)
    df["Loan_to_Income"] = df["Loan_to_Income"].round(3)

    cols = ["Client_ID", "Branch", "Loan_Officer", "Age", "Age_Band", "Income", "Loan_Amount",
            "Loan_to_Income", "Credit_Score", "Score_Band", "Loan_Cycle", "Cycle_Band",
            "Outstanding_Balance", "Total_Paid", "Loan_Status", "Default_Flag",
            "Predicted_PD", "Risk_Band", "Recommended_Action"]
    df[cols].to_csv(OUT / "fact_loans.csv", index=False)
    branch_summary(df).reset_index().to_csv(OUT / "dim_branch_summary.csv", index=False)
    collection_priority(df, top=200).to_csv(OUT / "collection_priority.csv", index=False)
    pd.read_csv(C.REPORTS_DIR / "model_comparison.csv").to_csv(OUT / "model_comparison.csv", index=False)
    imp = json.loads((C.REPORTS_DIR / "metrics.json").read_text())["importance"]
    pd.DataFrame({"Feature": imp.keys(), "Importance": imp.values()}).to_csv(
        OUT / "feature_importance.csv", index=False)
    print(f"Exported {len(df):,} loans + 4 supporting tables to {OUT}")
    print(df["Risk_Band"].value_counts().to_string())


if __name__ == "__main__":
    main()

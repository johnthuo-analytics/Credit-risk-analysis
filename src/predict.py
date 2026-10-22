"""Score a single applicant with the saved model."""
import joblib
import pandas as pd

from .config import MODELS_DIR, RISK_LOW_MAX, RISK_MEDIUM_MAX


def load_model():
    return joblib.load(MODELS_DIR / "credit_model.joblib")


def score(applicant: dict, bundle=None) -> dict:
    bundle = bundle or load_model()
    row = dict(applicant)
    row["Loan_to_Income"] = row["Loan_Amount"] / row["Income"]
    X = pd.DataFrame([row])[bundle["features"]]
    p = float(bundle["pipeline"].predict_proba(X)[0, 1])
    band = "Low" if p < RISK_LOW_MAX else "Medium" if p < RISK_MEDIUM_MAX else "High"
    return {"default_probability": p, "risk_band": band,
            "decision": "Review" if p >= bundle["threshold"] else "Eligible for review",
            "decision_note": "This is a demonstration model, not an automated lending decision."}

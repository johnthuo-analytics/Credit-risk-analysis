"""Streamlit demo: score an applicant and explore branch risk.  Run: streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.config import MODELS_DIR
from src.data import clean, load_raw
from src.portfolio import branch_summary
from src.predict import load_model, score

st.set_page_config(page_title="Credit Risk Dashboard", page_icon="🏦", layout="wide")
st.title("🏦 Credit Risk Scoring & Portfolio Monitor")

if not (MODELS_DIR / "credit_model.joblib").exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

tab1, tab2 = st.tabs(["Applicant scoring", "Portfolio monitor"])

with tab1:
    bundle = load_model()
    df = clean(load_raw())
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Age", 18, 70, 35)
    income = c2.number_input("Monthly income", 5_000, 1_000_000, 60_000, step=1_000)
    loan = c3.number_input("Loan amount", 5_000, 3_000_000, 150_000, step=5_000)
    credit = c1.slider("Credit score", 300, 850, 620)
    cycle = c2.slider("Previous loans (cycle)", 0, 10, 1)
    branch = c3.selectbox("Branch", sorted(df["Branch"].unique()))
    if st.button("Score applicant", type="primary"):
        r = score({"Age": age, "Income": income, "Loan_Amount": loan,
                   "Credit_Score": credit, "Loan_Cycle": cycle, "Branch": branch}, bundle)
        a, b, c = st.columns(3)
        a.metric("Default probability", f"{r['default_probability']:.1%}")
        b.metric("Risk band", r["risk_band"])
        c.metric("Suggested action", r["decision"])
        st.caption(f"Model: {bundle['model_name']} | decision threshold {bundle['threshold']:.2f}")

with tab2:
    bs = branch_summary(clean(load_raw()))
    k1, k2, k3 = st.columns(3)
    k1.metric("Loans", f"{int(bs['loans'].sum()):,}")
    k2.metric("Outstanding", f"{bs['outstanding'].sum():,.0f}")
    k3.metric("Avg default rate", f"{(bs['default_rate'] * bs['loans']).sum() / bs['loans'].sum():.1%}")
    st.bar_chart(bs["default_rate"])
    st.dataframe(bs.style.format({"default_rate": "{:.1%}", "collection_efficiency": "{:.1%}",
                                  "share_of_outstanding": "{:.1%}", "share_of_defaults": "{:.1%}",
                                  "disbursed": "{:,.0f}", "outstanding": "{:,.0f}"}))

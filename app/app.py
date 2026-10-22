"""Streamlit demo for applicant risk scoring and loan-portfolio monitoring.

Run: python -m streamlit run app/app.py
This is an educational demo using synthetic data, not a production credit decision system.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.config import MODELS_DIR, REPORTS_DIR, RISK_LOW_MAX, RISK_MEDIUM_MAX
from src.data import clean, load_raw
from src.portfolio import branch_summary, cycle_summary, collection_priority
from src.predict import load_model, score

st.set_page_config(page_title="Credit Risk Analytics", page_icon="🏦", layout="wide")
st.title("Credit Risk Analytics")
st.caption("Applicant scoring · Portfolio monitoring · Risk drivers")

st.warning(
    "Educational portfolio demo. The default dataset is synthetic. Scores and suggested actions "
    "must not be used as the sole basis for real lending decisions."
)

if not (MODELS_DIR / "credit_model.joblib").exists():
    st.error("No trained model found. From the repository root, run `make all` first.")
    st.stop()

try:
    df = clean(load_raw())
    bundle = load_model()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load a valid dataset/model: {exc}")
    st.stop()

tab_score, tab_portfolio, tab_risk = st.tabs(
    ["Applicant scoring", "Portfolio overview", "Risk drivers & collections"]
)

with tab_score:
    st.subheader("Estimate default probability")
    st.caption("Enter application-time information only. Post-disbursement repayment fields are excluded.")
    branches = sorted(df["Branch"].dropna().unique().tolist())
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Applicant age", 18, 100, 35)
    income = c2.number_input("Monthly income (currency units)", min_value=1_000, max_value=10_000_000,
                              value=60_000, step=1_000)
    loan = c3.number_input("Requested loan amount", min_value=1_000, max_value=20_000_000,
                           value=150_000, step=5_000)
    credit = c1.slider("Credit score", 300, 850, 620)
    cycle = c2.slider("Previous loan cycle", 0, 30, 1)
    branch = c3.selectbox("Branch", branches)
    st.caption(f"Risk bands: Low < {RISK_LOW_MAX:.0%}; Medium < {RISK_MEDIUM_MAX:.0%}; High otherwise.")
    if st.button("Estimate risk", type="primary"):
        result = score({
            "Age": age, "Income": income, "Loan_Amount": loan,
            "Credit_Score": credit, "Loan_Cycle": cycle, "Branch": branch,
        }, bundle)
        a, b, c = st.columns(3)
        a.metric("Estimated default probability", f"{result['default_probability']:.1%}")
        b.metric("Probability risk band", result["risk_band"])
        c.metric("Threshold result", result["decision"])
        st.info(result["decision_note"])
        st.caption(
            f"Model: {bundle['model_name']} · Illustrative review threshold: "
            f"{bundle['threshold']:.2f}. This threshold is based on assumed costs and needs "
            "validation against real loss, pricing, policy and fairness requirements."
        )

with tab_portfolio:
    st.subheader("Portfolio overview")
    branch_options = ["All branches"] + sorted(df["Branch"].unique().tolist())
    selected_branch = st.selectbox("Filter branch", branch_options, key="portfolio_branch")
    view = df if selected_branch == "All branches" else df[df["Branch"] == selected_branch]
    bs = branch_summary(view)
    total_loans = len(view)
    default_rate = float(view["Default_Flag"].mean()) if total_loans else 0
    paid = float(view["Total_Paid"].sum())
    outstanding = float(view["Outstanding_Balance"].sum())
    collection_efficiency = paid / (paid + outstanding) if paid + outstanding else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Loans", f"{total_loans:,}")
    k2.metric("Disbursed", f"{view['Loan_Amount'].sum():,.0f}")
    k3.metric("Default rate", f"{default_rate:.1%}")
    k4.metric("Collection efficiency", f"{collection_efficiency:.1%}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Default rate by branch**")
        st.bar_chart(bs["default_rate"])
    with right:
        st.markdown("**Outstanding balance by branch**")
        st.bar_chart(bs["outstanding"])
    st.markdown("**Branch performance table**")
    st.dataframe(
        bs.style.format({
            "default_rate": "{:.1%}", "collection_efficiency": "{:.1%}",
            "share_of_outstanding": "{:.1%}", "share_of_defaults": "{:.1%}",
            "disbursed": "{:,.0f}", "outstanding": "{:,.0f}",
        }),
        use_container_width=True,
    )

with tab_risk:
    st.subheader("Risk patterns")
    left, right = st.columns(2)
    with left:
        st.markdown("**Default rate by credit score band**")
        score_band = pd.cut(
            df["Credit_Score"], [0, 500, 580, 670, 740, 900],
            labels=["Very poor", "Poor", "Fair", "Good", "Excellent"],
        )
        score_summary = df.assign(Score_Band=score_band).groupby(
            "Score_Band", observed=True
        )["Default_Flag"].mean()
        st.bar_chart(score_summary)
    with right:
        st.markdown("**Default rate by previous loan cycle**")
        st.bar_chart(cycle_summary(df)["default_rate"])
    st.markdown("**Collection follow-up candidates**")
    st.caption("Synthetic demo records only. In production, collection priority must follow approved policy and privacy rules.")
    priority = collection_priority(df, top=20)
    st.dataframe(priority, use_container_width=True, hide_index=True)
    st.download_button(
        "Download demo collection-priority CSV",
        data=priority.to_csv(index=False).encode("utf-8"),
        file_name="demo_collection_priority.csv",
        mime="text/csv",
    )

    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        decision_metrics = metrics.get("decision", {})
        st.markdown("**Hold-out evaluation summary**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Default recall", f"{decision_metrics.get('recall', 0):.1%}")
        m2.metric("Precision", f"{decision_metrics.get('precision', 0):.1%}")
        m3.metric("Approval below threshold", f"{decision_metrics.get('approval_rate', 0):.1%}")
        st.caption("Metrics reflect the synthetic hold-out test set and should not be generalized to real borrowers.")

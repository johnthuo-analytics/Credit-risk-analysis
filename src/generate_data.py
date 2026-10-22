"""Generate a SYNTHETIC loan dataset that follows the schema in data/README.md.

This exists so the whole pipeline runs end-to-end out of the box. It is NOT real
client data. Replace data/raw/loans.csv with your real export to get real results.
"""
import numpy as np
import pandas as pd

from .config import DATA_FILE, RANDOM_STATE

BRANCHES = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Thika"]
BRANCH_RISK = dict(zip(BRANCHES, [0.0, 0.35, 0.15, -0.1, 0.25, -0.2]))


def generate(n: int = 5000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 66, n)
    income = np.clip(rng.lognormal(10.6, 0.5, n), 8_000, 400_000).round(-2)
    credit = np.clip(rng.normal(620, 80, n), 300, 850).round()
    cycle = rng.poisson(2.0, n)
    branch = rng.choice(BRANCHES, n, p=[0.28, 0.17, 0.14, 0.14, 0.12, 0.15])
    officer = [f"LO_{b[:3].upper()}_{rng.integers(1, 6)}" for b in branch]
    loan = np.clip(income * rng.uniform(0.5, 6.0, n), 5_000, 1_500_000).round(-2)

    lti = loan / income
    logit = (
        -1.6
        + 0.55 * (lti - 3) / 1.5
        - 0.9 * (credit - 620) / 80
        - 0.25 * np.minimum(cycle, 6) / 2
        - 0.15 * (age - 40) / 12
        + np.array([BRANCH_RISK[b] for b in branch])
        + rng.normal(0, 0.35, n)
    )
    p_default = 1 / (1 + np.exp(-logit))
    default = rng.random(n) < p_default

    # Repayment behaviour (consequence of the outcome -> leakage for scoring models)
    paid_share = np.where(default, rng.beta(2, 4, n), rng.beta(18, 1.2, n)).clip(0, 1)
    paid_share = np.where(default, paid_share, np.where(rng.random(n) < 0.6, 1.0, paid_share))
    total_paid = (loan * paid_share * 1.18).round(-1)
    outstanding = np.maximum(loan * 1.18 - total_paid, 0).round(-1)

    return pd.DataFrame({
        "Client_ID": [f"C{100000 + i}" for i in range(n)],
        "Age": age, "Income": income, "Loan_Amount": loan,
        "Credit_Score": credit, "Branch": branch, "Loan_Officer": officer,
        "Loan_Cycle": cycle, "Outstanding_Balance": outstanding,
        "Total_Paid": total_paid,
        "Loan_Status": np.where(default, "Default", "Paid"),
    })


if __name__ == "__main__":
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(DATA_FILE, index=False)
    print(f"Wrote {len(df):,} synthetic rows to {DATA_FILE} "
          f"(default rate {(df.Loan_Status == 'Default').mean():.1%})")

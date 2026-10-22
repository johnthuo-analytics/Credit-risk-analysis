"""Portfolio monitoring metrics (uses post-disbursement fields, by design)."""
import numpy as np
import pandas as pd


def branch_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Branch")
    out = pd.DataFrame({
        "loans": g.size(),
        "disbursed": g["Loan_Amount"].sum(),
        "outstanding": g["Outstanding_Balance"].sum(),
        "default_rate": g["Default_Flag"].mean(),
    })
    repaid = g["Total_Paid"].sum()
    out["collection_efficiency"] = repaid / (repaid + out["outstanding"])
    out["share_of_outstanding"] = out["outstanding"] / out["outstanding"].sum()
    out["share_of_defaults"] = g["Default_Flag"].sum() / df["Default_Flag"].sum()
    return out.sort_values("default_rate", ascending=False)


def officer_summary(df: pd.DataFrame, min_loans: int = 30) -> pd.DataFrame:
    g = df.groupby("Loan_Officer")
    out = pd.DataFrame({"loans": g.size(), "default_rate": g["Default_Flag"].mean(),
                        "outstanding": g["Outstanding_Balance"].sum()})
    return out[out["loans"] >= min_loans].sort_values("default_rate", ascending=False)


def concentration_hhi(df: pd.DataFrame) -> float:
    """Herfindahl-Hirschman index of outstanding balance by branch (0-1)."""
    share = df.groupby("Branch")["Outstanding_Balance"].sum()
    share = share / share.sum()
    return float((share ** 2).sum())


def cycle_summary(df: pd.DataFrame) -> pd.DataFrame:
    bins = pd.cut(df["Loan_Cycle"], [-1, 0, 1, 3, 6, np.inf],
                  labels=["0", "1", "2-3", "4-6", "7+"])
    return df.groupby(bins, observed=True)["Default_Flag"].agg(["size", "mean"]).rename(
        columns={"size": "loans", "mean": "default_rate"})


def collection_priority(df: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Partially-paid, still-owing accounts ranked by balance: recovery targets."""
    paid_share = df["Total_Paid"] / (df["Total_Paid"] + df["Outstanding_Balance"]).replace(0, np.nan)
    mask = (df["Outstanding_Balance"] > 0) & (paid_share > 0.2)
    cols = ["Client_ID", "Branch", "Loan_Officer", "Outstanding_Balance", "Total_Paid"]
    return df.loc[mask, cols].assign(paid_share=paid_share[mask]).sort_values(
        "Outstanding_Balance", ascending=False).head(top)

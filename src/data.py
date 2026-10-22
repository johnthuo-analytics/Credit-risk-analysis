"""Loading, validation and cleaning."""
import pandas as pd

from .config import DATA_FILE, ID_COL, TARGET, POSITIVE_CLASS

REQUIRED_COLUMNS = [
    "Client_ID", "Age", "Income", "Loan_Amount", "Credit_Score", "Branch",
    "Loan_Officer", "Loan_Cycle", "Outstanding_Balance", "Total_Paid", "Loan_Status",
]


def load_raw(path=DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, fix types, impute, and add engineered features."""
    df = df.drop_duplicates(subset=ID_COL).copy()
    df["Branch"] = df["Branch"].astype(str).str.strip().str.title()
    df["Loan_Officer"] = df["Loan_Officer"].astype(str).str.strip()

    num = ["Age", "Income", "Loan_Amount", "Credit_Score", "Loan_Cycle",
           "Outstanding_Balance", "Total_Paid"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    df = df[(df["Income"] > 0) & (df["Loan_Amount"] > 0)]
    df["Loan_to_Income"] = df["Loan_Amount"] / df["Income"]
    df["Default_Flag"] = (df[TARGET] == POSITIVE_CLASS).astype(int)
    return df.reset_index(drop=True)

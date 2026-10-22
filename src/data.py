"""Loading, schema validation and conservative cleaning for loan data."""
import pandas as pd

from .config import DATA_FILE, ID_COL, TARGET, POSITIVE_CLASS

REQUIRED_COLUMNS = [
    "Client_ID", "Age", "Income", "Loan_Amount", "Credit_Score", "Branch",
    "Loan_Officer", "Loan_Cycle", "Outstanding_Balance", "Total_Paid", "Loan_Status",
]
NUMERIC_COLUMNS = [
    "Age", "Income", "Loan_Amount", "Credit_Score", "Loan_Cycle",
    "Outstanding_Balance", "Total_Paid",
]


def load_raw(path=DATA_FILE) -> pd.DataFrame:
    """Load a CSV and fail early when the expected schema is not present."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate target/IDs, normalize types, impute numeric gaps and engineer features.

    Exact duplicate rows are removed. Conflicting rows with the same Client_ID are
    rejected rather than silently keeping an arbitrary record. Imputation here is
    for descriptive/portfolio preparation; model preprocessing is fitted inside
    the sklearn pipeline on training folds only.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.copy()
    df = df.drop_duplicates()

    if df[ID_COL].isna().any() or df[ID_COL].astype(str).str.strip().eq("").any():
        raise ValueError("Client_ID contains blank values.")
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    duplicated_ids = df.loc[df[ID_COL].duplicated(keep=False), ID_COL].unique()
    if len(duplicated_ids):
        examples = ", ".join(duplicated_ids[:5])
        raise ValueError(f"Conflicting records share Client_ID values (e.g. {examples}). Resolve them before analysis.")

    status = df[TARGET].astype("string").str.strip().str.title()
    allowed = {"Paid", POSITIVE_CLASS}
    unexpected = sorted(set(status.dropna().unique()) - allowed)
    if status.isna().any() or unexpected:
        raise ValueError(f"{TARGET} must contain only 'Paid' or '{POSITIVE_CLASS}'. Unexpected values: {unexpected}")
    df[TARGET] = status

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().sum() == 0:
            raise ValueError(f"{col} has no usable numeric values.")

    # Impute post-disbursement fields for portfolio totals only. Application-time
    # predictor gaps are deliberately left missing so sklearn imputes them inside
    # each training fold and cannot learn from the hold-out set's distribution.
    for col in ("Outstanding_Balance", "Total_Paid"):
        df[col] = df[col].fillna(df[col].median())

    for col in ("Branch", "Loan_Officer"):
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA).fillna("Unknown").astype(str)

    # Invalid core amounts cannot support a meaningful ratio or portfolio metric.
    df = df[df["Income"].notna() & df["Loan_Amount"].notna()
            & (df["Income"] > 0) & (df["Loan_Amount"] > 0)].copy()
    if df.empty:
        raise ValueError("No rows remain after excluding non-positive income or loan amounts.")
    if ((df["Age"].dropna() < 18).any() or (df["Age"].dropna() > 100).any()):
        raise ValueError("Age contains values outside the supported 18-100 range.")
    if (df["Loan_Cycle"].dropna() < 0).any():
        raise ValueError("Loan_Cycle cannot be negative.")

    df["Loan_to_Income"] = df["Loan_Amount"] / df["Income"]
    df["Default_Flag"] = (df[TARGET] == POSITIVE_CLASS).astype(int)
    return df.reset_index(drop=True)

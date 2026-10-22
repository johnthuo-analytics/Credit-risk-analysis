"""Central configuration: paths, column names, modelling constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

# Swap this for your real file. Schema is documented in data/README.md.
DATA_FILE = RAW_DIR / "loans.csv"

TARGET = "Loan_Status"
POSITIVE_CLASS = "Default"
ID_COL = "Client_ID"

# Known only at application time -> safe for an application-scoring model.
NUMERIC_FEATURES = [
    "Age", "Income", "Loan_Amount", "Credit_Score", "Loan_Cycle",
    "Loan_to_Income",          # engineered
]
CATEGORICAL_FEATURES = ["Branch"]

# Observed AFTER disbursement. Using them to predict default is target leakage,
# so they are used for portfolio monitoring only, never for the scoring model.
LEAKAGE_COLS = ["Outstanding_Balance", "Total_Paid"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3

# Illustrative business costs for threshold selection (relative units).
COST_FALSE_NEGATIVE = 5.0   # approve a loan that defaults (loss of principal)
COST_FALSE_POSITIVE = 1.0   # reject a loan that would have repaid (lost interest)

# Probability-based reporting bands. The approval/review threshold is a separate
# business decision calibrated to assumed error costs; do not conflate the two.
RISK_LOW_MAX = 0.10
RISK_MEDIUM_MAX = 0.25

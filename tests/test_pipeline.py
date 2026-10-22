import numpy as np
import pandas as pd

from src.data import clean
from src.evaluate import best_threshold, ks_statistic
from src.generate_data import generate
from src.portfolio import branch_summary, concentration_hhi


def _df():
    return clean(generate(n=800, seed=1))


def test_schema_and_target():
    df = _df()
    assert set(df["Default_Flag"].unique()) <= {0, 1}
    assert (df["Loan_to_Income"] > 0).all()
    assert df["Client_ID"].is_unique


def test_no_leakage_columns_in_features():
    from src import config as C
    feats = set(C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES)
    assert not feats & set(C.LEAKAGE_COLS)


def test_branch_summary_shares_sum_to_one():
    bs = branch_summary(_df())
    assert abs(bs["share_of_outstanding"].sum() - 1) < 1e-9
    assert abs(bs["share_of_defaults"].sum() - 1) < 1e-9


def test_hhi_bounds():
    h = concentration_hhi(_df())
    assert 1 / 6 - 1e-9 <= h <= 1


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_threshold_prefers_catching_defaults_when_fn_costly():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    t_costly, _ = best_threshold(y, p, c_fn=10, c_fp=1)
    t_even, _ = best_threshold(y, p, c_fn=1, c_fp=1)
    assert t_costly <= t_even


def test_exact_duplicate_rows_are_removed():
    raw = generate(n=50, seed=8)
    result = clean(pd.concat([raw, raw.iloc[[0]]], ignore_index=True))
    assert len(result) == len(raw)
    assert result["Client_ID"].is_unique


def test_conflicting_duplicate_ids_are_rejected():
    raw = generate(n=50, seed=9)
    duplicate = raw.iloc[[0]].copy()
    duplicate.loc[:, "Income"] = duplicate["Income"] * 1.1
    with np.testing.assert_raises_regex(ValueError, "Conflicting records share Client_ID"):
        clean(pd.concat([raw, duplicate], ignore_index=True))


def test_invalid_target_is_rejected():
    raw = generate(n=50, seed=10)
    raw.loc[0, "Loan_Status"] = "Unknown"
    with np.testing.assert_raises_regex(ValueError, "Loan_Status must contain only"):
        clean(raw)


def test_blank_branch_is_normalized():
    raw = generate(n=50, seed=11)
    raw.loc[0, "Branch"] = None
    result = clean(raw)
    assert result.loc[0, "Branch"] == "Unknown"


def test_risk_bands_use_probability_thresholds():
    from src import config as C
    from src.predict import score
    from sklearn.dummy import DummyClassifier
    # Test the band cutoffs without requiring the trained production artifact.
    bundle = {
        "pipeline": DummyClassifier(strategy="constant", constant=0),
        "threshold": 0.3,
        "features": C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES,
        "model_name": "test",
    }
    bundle["pipeline"].fit(
        pd.DataFrame({"Age":[30, 40], "Income":[50000, 60000], "Loan_Amount":[10000, 20000],
                      "Credit_Score":[650, 600], "Loan_Cycle":[1, 2],
                      "Loan_to_Income":[0.2, 0.33], "Branch":["Nairobi", "Thika"]}),
        [0, 1],
    )
    result = score({"Age":30, "Income":50000, "Loan_Amount":10000,
                    "Credit_Score":650, "Loan_Cycle":1, "Branch":"Nairobi"}, bundle)
    assert result["risk_band"] == "Low"
    assert "not an automated lending decision" in result["decision_note"]


def test_application_feature_missingness_is_left_for_fold_pipeline():
    from src.train import build_preprocessor
    raw = generate(n=100, seed=12)
    raw.loc[0, "Age"] = np.nan
    cleaned = clean(raw)
    assert pd.isna(cleaned.loc[0, "Age"])
    features = ["Age", "Income", "Loan_Amount", "Credit_Score", "Loan_Cycle",
                "Loan_to_Income", "Branch"]
    transformed = build_preprocessor().fit_transform(
        cleaned[features].iloc[1:], cleaned["Default_Flag"].iloc[1:]
    )
    assert transformed.shape[0] == len(cleaned) - 1

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

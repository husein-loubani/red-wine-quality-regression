"""
conftest.py
Shared pytest fixtures for the Red Wine Quality Regression test suite.
"""

# Standard library
import sys
from pathlib import Path

# Make src importable when running pytest from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from red_wine_quality.config import FULL_OLS_FORMULA, REDUCED_OLS_FORMULA
from red_wine_quality.data_processing import add_log_features


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Load the raw wine CSV once for the whole test session."""
    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "winequality-red.csv"
    from red_wine_quality.data_processing import load_data
    return load_data(path)


@pytest.fixture(scope="session")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicated DataFrame with free_sulfur_dioxide dropped."""
    df = raw_df.drop_duplicates().copy()
    df = df.drop(columns=["free_sulfur_dioxide"])
    return df


@pytest.fixture(scope="session")
def split_dfs(clean_df: pd.DataFrame):
    """Return (train_df, test_df) with stratified split."""
    from red_wine_quality.data_processing import split_data
    return split_data(clean_df)


@pytest.fixture(scope="session")
def train_df(split_dfs):
    return split_dfs[0]


@pytest.fixture(scope="session")
def test_df(split_dfs):
    return split_dfs[1]


@pytest.fixture(scope="session")
def train_rdf(train_df: pd.DataFrame) -> pd.DataFrame:
    """Training DataFrame with log features added."""
    return add_log_features(train_df)


@pytest.fixture(scope="session")
def test_rdf(test_df: pd.DataFrame) -> pd.DataFrame:
    """Hold-out DataFrame with log features added."""
    return add_log_features(test_df)


@pytest.fixture(scope="session")
def full_model(train_rdf: pd.DataFrame):
    """Fitted full OLS model."""
    from red_wine_quality.statistics import fit_ols
    return fit_ols(train_rdf, formula=FULL_OLS_FORMULA)


@pytest.fixture(scope="session")
def reduced_model(train_rdf: pd.DataFrame):
    """Fitted reduced OLS model."""
    from red_wine_quality.statistics import fit_ols
    return fit_ols(train_rdf, formula=REDUCED_OLS_FORMULA)

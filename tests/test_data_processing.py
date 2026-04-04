"""
test_data_processing.py
Tests for src/data_processing.py
"""

import numpy as np
import pandas as pd
import pytest

from red_wine_quality.data_processing import (
    load_data,
    assess_data,
    descriptive_stats,
    add_log_features,
    split_data,
    compute_vif,
)
from red_wine_quality.config import SKEW_CANDIDATES, RANDOM_SEED


class TestLoadData:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_shape(self, raw_df):
        assert raw_df.shape == (1599, 12)

    def test_columns_renamed(self, raw_df):
        assert "volatile_acidity" in raw_df.columns
        assert "volatile acidity" not in raw_df.columns

    def test_quality_column_present(self, raw_df):
        assert "quality" in raw_df.columns

    def test_no_unexpected_nulls(self, raw_df):
        assert raw_df.isna().sum().sum() == 0


class TestAssessData:
    def test_returns_dict(self, raw_df):
        report = assess_data(raw_df)
        assert isinstance(report, dict)

    def test_report_keys(self, raw_df):
        report = assess_data(raw_df)
        for key in ("n_rows", "n_cols", "missing_per_column", "n_duplicates", "dtypes"):
            assert key in report

    def test_duplicate_count(self, raw_df):
        report = assess_data(raw_df)
        assert report["n_duplicates"] == 240


class TestDescriptiveStats:
    def test_returns_dataframe(self, clean_df):
        result = descriptive_stats(clean_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_extra_columns(self, clean_df):
        result = descriptive_stats(clean_df)
        for col in ("cv", "skew", "kurt"):
            assert col in result.columns

    def test_index_matches_columns(self, clean_df):
        result = descriptive_stats(clean_df)
        assert set(result.index) == set(clean_df.columns)


class TestAddLogFeatures:
    def test_log_columns_created(self, train_df):
        result = add_log_features(train_df)
        for col in SKEW_CANDIDATES:
            assert f"log_{col}" in result.columns

    def test_original_columns_preserved(self, train_df):
        result = add_log_features(train_df)
        for col in train_df.columns:
            assert col in result.columns

    def test_no_mutation(self, train_df):
        original_cols = list(train_df.columns)
        _ = add_log_features(train_df)
        assert list(train_df.columns) == original_cols

    def test_log_values_finite(self, train_df):
        result = add_log_features(train_df)
        for col in SKEW_CANDIDATES:
            assert np.isfinite(result[f"log_{col}"]).all()

    def test_skewness_reduced(self, train_df):
        result = add_log_features(train_df)
        for col in SKEW_CANDIDATES:
            original_skew = abs(train_df[col].skew())
            log_skew = abs(result[f"log_{col}"].skew())
            assert log_skew < original_skew, (
                f"log1p transform did not reduce skewness for {col}: "
                f"{original_skew:.2f} -> {log_skew:.2f}"
            )


class TestSplitData:
    def test_sizes(self, clean_df, split_dfs):
        train_df, test_df = split_dfs
        total = len(clean_df)
        assert len(test_df) == pytest.approx(total * 0.20, abs=5)
        assert len(train_df) + len(test_df) == total

    def test_no_overlap(self, split_dfs):
        train_df, test_df = split_dfs
        train_idx = set(train_df.index)
        test_idx = set(test_df.index)
        assert train_idx.isdisjoint(test_idx)

    def test_stratification(self, split_dfs, clean_df):
        train_df, test_df = split_dfs
        train_dist = train_df["quality"].value_counts(normalize=True).sort_index()
        test_dist = test_df["quality"].value_counts(normalize=True).sort_index()
        for score in train_dist.index:
            if score in test_dist.index:
                assert abs(train_dist[score] - test_dist[score]) < 0.05, (
                    f"Quality score {score} is not balanced: "
                    f"train={train_dist[score]:.3f}, test={test_dist[score]:.3f}"
                )

    def test_reproducibility(self, clean_df):
        train1, test1 = split_data(clean_df, random_state=RANDOM_SEED)
        train2, test2 = split_data(clean_df, random_state=RANDOM_SEED)
        pd.testing.assert_frame_equal(train1.reset_index(drop=True),
                                      train2.reset_index(drop=True))


class TestComputeVif:
    def test_returns_dataframe(self, train_rdf):
        from red_wine_quality.config import FULL_PREDICTORS
        result = compute_vif(train_rdf, FULL_PREDICTORS)
        assert isinstance(result, pd.DataFrame)

    def test_has_vif_column(self, train_rdf):
        from red_wine_quality.config import FULL_PREDICTORS
        result = compute_vif(train_rdf, FULL_PREDICTORS)
        assert "VIF" in result.columns

    def test_no_const_row(self, train_rdf):
        from red_wine_quality.config import FULL_PREDICTORS
        result = compute_vif(train_rdf, FULL_PREDICTORS)
        assert "const" not in result.index

    def test_all_vif_positive(self, train_rdf):
        from red_wine_quality.config import FULL_PREDICTORS
        result = compute_vif(train_rdf, FULL_PREDICTORS)
        assert (result["VIF"] > 0).all()

    def test_density_raises_vif(self, train_rdf):
        from red_wine_quality.config import FULL_PREDICTORS
        vif_without = compute_vif(train_rdf, FULL_PREDICTORS)
        vif_with = compute_vif(train_rdf, FULL_PREDICTORS + ["density"])
        max_without = vif_without["VIF"].max()
        max_with = vif_with["VIF"].max()
        assert max_with > max_without

"""
test_plotting.py
Tests for src/plotting.py. Verifies each function returns a valid Figure
and saves correctly to disk.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for CI

import matplotlib.pyplot as plt
import pytest
from pathlib import Path

from red_wine_quality.config import SKEW_CANDIDATES, FULL_PREDICTORS
from red_wine_quality.data_processing import add_log_features, compute_vif
from red_wine_quality.statistics import (
    fit_ols, coefficient_table, standardized_coefficients, evaluate_holdout,
)
from red_wine_quality.plotting import (
    save_figure,
    plot_target_distribution,
    plot_univariate_distributions,
    plot_predictors_vs_target,
    plot_correlation_bar,
    plot_correlation_heatmap,
    plot_log_transform_comparison,
    plot_vif,
    plot_coefficients,
    plot_standardized_coefficients,
    plot_residual_diagnostics,
    plot_holdout_evaluation,
    plot_violin_pair,
    plot_stratification_check,
)
from red_wine_quality.config import FULL_OLS_FORMULA, REDUCED_OLS_FORMULA


@pytest.fixture(scope="module")
def coef_df(full_model):
    return coefficient_table(full_model)


@pytest.fixture(scope="module")
def std_coefs(full_model, train_rdf):
    return standardized_coefficients(full_model, train_rdf)


@pytest.fixture(scope="module")
def holdout(reduced_model, test_rdf):
    return evaluate_holdout(reduced_model, test_rdf)


@pytest.fixture(scope="module")
def vif_pair(train_rdf):
    before = compute_vif(train_rdf, FULL_PREDICTORS + ["density"])
    after = compute_vif(train_rdf, FULL_PREDICTORS)
    return before, after


def assert_figure(fig) -> None:
    """Helper: confirm the return value is a Matplotlib Figure with at least one axes."""
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    plt.close(fig)


class TestPlotFunctions:
    def test_target_distribution(self, clean_df):
        fig = plot_target_distribution(clean_df)
        assert_figure(fig)

    def test_univariate_distributions(self, train_df):
        predictors = [c for c in train_df.columns if c != "quality"]
        fig = plot_univariate_distributions(train_df, predictors)
        assert_figure(fig)

    def test_predictors_vs_target(self, train_df):
        predictors = [c for c in train_df.columns if c != "quality"]
        fig = plot_predictors_vs_target(train_df, predictors)
        assert_figure(fig)

    def test_correlation_bar(self, train_df):
        fig = plot_correlation_bar(train_df)
        assert_figure(fig)

    def test_correlation_heatmap(self, train_df):
        fig = plot_correlation_heatmap(train_df)
        assert_figure(fig)

    def test_log_transform_comparison(self, train_df):
        fig = plot_log_transform_comparison(train_df, SKEW_CANDIDATES)
        assert_figure(fig)

    def test_vif(self, vif_pair):
        fig = plot_vif(*vif_pair)
        assert_figure(fig)

    def test_coefficients(self, coef_df):
        fig = plot_coefficients(coef_df)
        assert_figure(fig)

    def test_standardized_coefficients(self, std_coefs):
        fig = plot_standardized_coefficients(std_coefs)
        assert_figure(fig)

    def test_residual_diagnostics(self, reduced_model):
        fig = plot_residual_diagnostics(reduced_model)
        assert_figure(fig)

    def test_holdout_evaluation(self, holdout):
        fig = plot_holdout_evaluation(
            holdout.actuals, holdout.predictions, holdout.rmse, holdout.r2
        )
        assert_figure(fig)

    def test_violin_pair(self, train_df):
        fig = plot_violin_pair(train_df, "alcohol", "volatile_acidity")
        assert_figure(fig)

    def test_stratification_check(self, train_df, test_df):
        fig = plot_stratification_check(train_df, test_df)
        assert_figure(fig)


class TestSaveFigure:
    def test_creates_file(self, clean_df, tmp_path):
        fig = plot_target_distribution(clean_df)
        save_figure(fig, "test_plot", tmp_path)
        assert (tmp_path / "test_plot.png").exists()
        plt.close(fig)

    def test_creates_directory_if_missing(self, clean_df, tmp_path):
        nested = tmp_path / "new_dir" / "sub"
        fig = plot_target_distribution(clean_df)
        save_figure(fig, "test_plot", nested)
        assert (nested / "test_plot.png").exists()
        plt.close(fig)

    def test_file_non_empty(self, clean_df, tmp_path):
        fig = plot_target_distribution(clean_df)
        save_figure(fig, "test_plot", tmp_path)
        size = (tmp_path / "test_plot.png").stat().st_size
        assert size > 1000, "Saved PNG is suspiciously small"
        plt.close(fig)

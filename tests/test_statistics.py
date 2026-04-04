"""
test_statistics.py
Tests for src/statistics.py
"""

import numpy as np
import pandas as pd
import pytest

from red_wine_quality.statistics import (
    fit_ols,
    coefficient_table,
    standardized_coefficients,
    evaluate_holdout,
    compare_models,
    compare_shared_coefficients,
)
from red_wine_quality.config import FULL_OLS_FORMULA, REDUCED_OLS_FORMULA, ALPHA


class TestFitOls:
    def test_returns_model(self, train_rdf):
        model = fit_ols(train_rdf, formula=FULL_OLS_FORMULA)
        assert model is not None

    def test_rsquared_in_range(self, full_model):
        assert 0 < full_model.rsquared < 1

    def test_nobs_matches_training_size(self, full_model, train_rdf):
        assert int(full_model.nobs) == len(train_rdf)

    def test_f_pvalue_significant(self, full_model):
        assert full_model.f_pvalue < ALPHA

    def test_key_predictors_significant(self, full_model):
        pvals = full_model.pvalues
        for var in ("volatile_acidity", "alcohol", "log_sulphates"):
            assert pvals[var] < ALPHA, f"{var} should be significant"

    def test_alcohol_positive(self, full_model):
        assert full_model.params["alcohol"] > 0

    def test_volatile_acidity_negative(self, full_model):
        assert full_model.params["volatile_acidity"] < 0


class TestCoefficientTable:
    def test_returns_dataframe(self, full_model):
        result = coefficient_table(full_model)
        assert isinstance(result, pd.DataFrame)

    def test_no_intercept(self, full_model):
        result = coefficient_table(full_model)
        assert "Intercept" not in result.index

    def test_expected_columns(self, full_model):
        result = coefficient_table(full_model)
        for col in ("coef", "se", "t", "pvalue", "ci_low", "ci_high", "significant"):
            assert col in result.columns

    def test_ci_contains_coef(self, full_model):
        result = coefficient_table(full_model)
        assert (result["ci_low"] <= result["coef"]).all()
        assert (result["coef"] <= result["ci_high"]).all()

    def test_significant_flag(self, full_model):
        result = coefficient_table(full_model)
        expected = result["pvalue"] < ALPHA
        pd.testing.assert_series_equal(result["significant"], expected, check_names=False)


class TestStandardizedCoefficients:
    def test_returns_dataframe(self, full_model, train_rdf):
        result = standardized_coefficients(full_model, train_rdf)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, full_model, train_rdf):
        result = standardized_coefficients(full_model, train_rdf)
        for col in ("raw_coef", "std_coef", "abs_std_coef"):
            assert col in result.columns

    def test_no_intercept(self, full_model, train_rdf):
        result = standardized_coefficients(full_model, train_rdf)
        assert "Intercept" not in result.index

    def test_sorted_by_abs_desc(self, full_model, train_rdf):
        result = standardized_coefficients(full_model, train_rdf)
        vals = result["abs_std_coef"].values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_alcohol_top_positive(self, full_model, train_rdf):
        result = standardized_coefficients(full_model, train_rdf)
        top_positive = result[result["std_coef"] > 0].head(1).index[0]
        assert top_positive == "alcohol"


class TestEvaluateHoldout:
    def test_rmse_positive(self, reduced_model, test_rdf):
        result = evaluate_holdout(reduced_model, test_rdf)
        assert result.rmse > 0

    def test_mae_positive(self, reduced_model, test_rdf):
        result = evaluate_holdout(reduced_model, test_rdf)
        assert result.mae > 0

    def test_r2_reasonable(self, reduced_model, test_rdf):
        result = evaluate_holdout(reduced_model, test_rdf)
        assert 0.1 < result.r2 < 0.8

    def test_mae_less_than_rmse(self, reduced_model, test_rdf):
        result = evaluate_holdout(reduced_model, test_rdf)
        assert result.mae <= result.rmse

    def test_predictions_length(self, reduced_model, test_rdf):
        result = evaluate_holdout(reduced_model, test_rdf)
        assert len(result.predictions) == len(test_rdf)

    def test_r2_gap_small(self, reduced_model, test_rdf):
        result = evaluate_holdout(reduced_model, test_rdf)
        assert abs(result.r2_gap) < 0.15, "Train-holdout R2 gap is suspiciously large"


class TestCompareModels:
    def test_returns_dataframe(self, full_model, reduced_model, train_rdf, test_rdf):
        result = compare_models(full_model, reduced_model, train_rdf, test_rdf)
        assert isinstance(result, pd.DataFrame)

    def test_has_both_model_columns(self, full_model, reduced_model, train_rdf, test_rdf):
        result = compare_models(full_model, reduced_model, train_rdf, test_rdf)
        assert "Full model" in result.columns
        assert "Reduced model" in result.columns

    def test_reduced_fewer_predictors(self, full_model, reduced_model, train_rdf, test_rdf):
        result = compare_models(full_model, reduced_model, train_rdf, test_rdf)
        full_n = result.loc["No. predictors", "Full model"]
        red_n = result.loc["No. predictors", "Reduced model"]
        assert red_n < full_n

    def test_reduced_lower_bic(self, full_model, reduced_model, train_rdf, test_rdf):
        result = compare_models(full_model, reduced_model, train_rdf, test_rdf)
        assert result.loc["BIC", "Reduced model"] < result.loc["BIC", "Full model"]


class TestCompareSharedCoefficients:
    def test_returns_dataframe(self, full_model, reduced_model):
        result = compare_shared_coefficients(full_model, reduced_model)
        assert isinstance(result, pd.DataFrame)

    def test_only_shared_vars(self, full_model, reduced_model):
        result = compare_shared_coefficients(full_model, reduced_model)
        full_vars = set(full_model.params.drop("Intercept", errors="ignore").index)
        red_vars = set(reduced_model.params.drop("Intercept", errors="ignore").index)
        shared = full_vars & red_vars
        assert set(result.index) == shared

    def test_alcohol_stable(self, full_model, reduced_model):
        result = compare_shared_coefficients(full_model, reduced_model)
        change_pct = abs(result.loc["alcohol", "Change (%)"])
        assert change_pct < 10, f"Alcohol coefficient changed by {change_pct:.1f}% — possible confounding"

    def test_volatile_acidity_stable(self, full_model, reduced_model):
        result = compare_shared_coefficients(full_model, reduced_model)
        change_pct = abs(result.loc["volatile_acidity", "Change (%)"])
        assert change_pct < 10

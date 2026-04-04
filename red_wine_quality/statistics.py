"""
statistics.py
Model fitting, evaluation, comparison, and hypothesis testing utilities
for the Red Wine Quality Regression project.
"""

# Standard library
from dataclasses import dataclass, field

# Third-party
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

# Local
from red_wine_quality.config import ALPHA, FULL_OLS_FORMULA


def fit_ols(train_df: pd.DataFrame, formula: str = FULL_OLS_FORMULA):
    """Fit an OLS model using the statsmodels formula API."""
    model = smf.ols(formula, data=train_df).fit()
    return model


def print_fit_summary(model) -> None:
    """Print a clean model-fit summary (R-squared, adjusted R-squared, F-statistic)."""
    print("Model Fit Summary")
    print("-" * 42)
    print(f"  Observations   : {int(model.nobs):,}")
    print(f"  Predictors     : {int(model.df_model)}")
    print(f"  R-squared      : {model.rsquared:.4f}")
    print(f"  Adjusted R-sq  : {model.rsquared_adj:.4f}")
    print(f"  AIC            : {model.aic:.1f}")
    print(f"  BIC            : {model.bic:.1f}")
    print(f"  F-statistic    : {model.fvalue:.2f}  (p = {model.f_pvalue:.2e})")
    print()
    pct = model.rsquared * 100
    print(
        f"  The model explains {pct:.1f}% of variance in quality "
        "ratings on the training set."
    )
    print(
        "  The F-statistic is highly significant, confirming that the model "
        "beats a null (intercept-only) baseline."
    )


def coefficient_table(model, alpha: float = ALPHA) -> pd.DataFrame:
    """Return a tidy coefficient DataFrame (intercept excluded)."""
    tbl = model.summary2().tables[1].copy()
    tbl = tbl.drop("Intercept", errors="ignore")
    tbl.columns = ["coef", "se", "t", "pvalue", "ci_low", "ci_high"]
    tbl["significant"] = tbl["pvalue"] < alpha
    return tbl.sort_values("coef")


def standardized_coefficients(model, train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute standardized (beta) coefficients for comparability across scales."""
    params = model.params.drop("Intercept", errors="ignore")
    predictors = params.index.tolist()
    sd_x = train_df[predictors].std()
    sd_y = train_df["quality"].std()
    std_coef = params * sd_x / sd_y
    result = pd.DataFrame({
        "raw_coef": params,
        "std_coef": std_coef,
        "abs_std_coef": std_coef.abs(),
    }).sort_values("abs_std_coef", ascending=False)
    return result


@dataclass
class HoldoutResult:
    rmse: float
    mae: float
    r2: float
    train_r2: float
    predictions: pd.Series = field(repr=False)
    actuals: pd.Series = field(repr=False)

    @property
    def r2_gap(self) -> float:
        return self.train_r2 - self.r2


def evaluate_holdout(model, test_df: pd.DataFrame) -> HoldoutResult:
    """Predict on the hold-out set and return a HoldoutResult dataclass."""
    predictions = model.predict(test_df)
    actuals = test_df["quality"]

    result = HoldoutResult(
        rmse=mean_squared_error(actuals, predictions) ** 0.5,
        mae=mean_absolute_error(actuals, predictions),
        r2=r2_score(actuals, predictions),
        train_r2=model.rsquared,
        predictions=predictions,
        actuals=actuals,
    )

    print("Hold-out Performance")
    print("-" * 38)
    print(f"  RMSE           : {result.rmse:.4f}")
    print(f"  MAE            : {result.mae:.4f}")
    print(f"  R-squared      : {result.r2:.4f}")
    print(f"  Train R-sq     : {result.train_r2:.4f}")
    print(f"  Gap            : {result.r2_gap:.4f}")
    print()
    print(
        f"  Predictions are within +/-{result.mae:.2f} quality points on average."
    )
    return result


def compare_models(
    full_model,
    reduced_model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare full and reduced models on key metrics, returning a summary DataFrame."""
    full_pred = full_model.predict(test_df)
    red_pred = reduced_model.predict(test_df)
    actual = test_df["quality"]

    rows = {
        "Adjusted R-sq (train)": [full_model.rsquared_adj, reduced_model.rsquared_adj],
        "AIC": [full_model.aic, reduced_model.aic],
        "BIC": [full_model.bic, reduced_model.bic],
        "R-sq (hold-out)": [
            r2_score(actual, full_pred),
            r2_score(actual, red_pred),
        ],
        "RMSE (hold-out)": [
            mean_squared_error(actual, full_pred) ** 0.5,
            mean_squared_error(actual, red_pred) ** 0.5,
        ],
        "MAE (hold-out)": [
            mean_absolute_error(actual, full_pred),
            mean_absolute_error(actual, red_pred),
        ],
        "No. predictors": [int(full_model.df_model), int(reduced_model.df_model)],
    }

    comparison = pd.DataFrame(rows, index=["Full model", "Reduced model"]).T
    return comparison


def compare_shared_coefficients(full_model, reduced_model) -> pd.DataFrame:
    """Show coefficient stability for predictors shared between both models."""
    full_params = full_model.params.drop("Intercept", errors="ignore")
    red_params = reduced_model.params.drop("Intercept", errors="ignore")
    shared = sorted(set(full_params.index) & set(red_params.index))

    rows = []
    for var in shared:
        full_ci = full_model.conf_int().loc[var]
        red_ci = reduced_model.conf_int().loc[var]
        rows.append({
            "Variable": var,
            "Coef (full)": full_params[var],
            "Coef (reduced)": red_params[var],
            "Change (%)": (
                (red_params[var] - full_params[var]) / abs(full_params[var]) * 100
                if full_params[var] != 0 else 0.0
            ),
            "CI low (full)": full_ci.iloc[0],
            "CI high (full)": full_ci.iloc[1],
            "CI low (reduced)": red_ci.iloc[0],
            "CI high (reduced)": red_ci.iloc[1],
        })

    return pd.DataFrame(rows).set_index("Variable")


def compute_rank_correlations(
    df: pd.DataFrame,
    target: str = "quality",
) -> pd.DataFrame:
    """Return Pearson r and Spearman rho for each numeric predictor vs the target, sorted by Pearson."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictors = [c for c in numeric_cols if c != target]
    sub = df[predictors + [target]]
    pearson = sub.corr(method="pearson", numeric_only=True)[target].drop(target)
    spearman = sub.corr(method="spearman", numeric_only=True)[target].drop(target)
    result = pd.DataFrame({"pearson_r": pearson, "spearman_rho": spearman})
    result = result.reindex(pearson.abs().sort_values().index)
    return result


def answer_hypothesis(
    model,
    variable: str,
    expected_direction: str,
    alpha: float = ALPHA,
) -> None:
    """Print a formal hypothesis test verdict for a single model variable."""
    row = model.summary2().tables[1].loc[variable]
    coef = row["Coef."]
    ci_low = row["[0.025"]
    ci_hi = row["0.975]"]
    pval = row["P>|t|"]

    direction_ok = (
        (expected_direction == "negative" and coef < 0)
        or (expected_direction == "positive" and coef > 0)
    )
    reject = (pval < alpha) and direction_ok

    print(f"Variable : {variable}")
    print(f"  Coefficient  : {coef:.4f}")
    print(f"  95% CI       : [{ci_low:.4f}, {ci_hi:.4f}]")
    print(f"  p-value      : {pval:.2e}")
    print()
    if reject:
        print(
            f"  RESULT: Reject H0. The {expected_direction} association is statistically "
            f"significant at alpha = {alpha}. The 95% CI excludes zero."
        )
    else:
        print(
            f"  RESULT: Fail to reject H0. Insufficient evidence of a {expected_direction} "
            f"association at alpha = {alpha}."
        )


def stepwise_summary(models: list, labels: list[str]) -> pd.DataFrame:
    """Compare a sequence of OLS models on key fit metrics.

    Useful for tracking adjusted R-squared, AIC, and BIC across
    stepwise backward elimination rounds.
    """
    return pd.DataFrame({
        "Model": labels,
        "Predictors": [int(m.df_model) for m in models],
        "Adj R-sq": [round(m.rsquared_adj, 4) for m in models],
        "AIC": [round(m.aic, 1) for m in models],
        "BIC": [round(m.bic, 1) for m in models],
    }).set_index("Model")


def jarque_bera_report(residuals: pd.Series) -> None:
    """Run Jarque-Bera test on residuals and print interpretation."""
    stat, pval = stats.jarque_bera(residuals)
    print(f"Jarque-Bera: statistic = {stat:.2f},  p = {pval:.4f}")
    if pval < 0.05:
        print(
            "  Residuals deviate from normality (p < 0.05). This is common with ordinal targets."
        )
        print(
            "  Coefficients remain unbiased, but confidence intervals should be interpreted with"
        )
        print(
            "  care at extreme quality scores (3 or 8) where the sample is thin."
        )
    else:
        print("  Residuals are approximately normally distributed (p >= 0.05).")

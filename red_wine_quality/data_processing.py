"""
data_processing.py
Data loading, cleaning, feature engineering, and train/test splitting.

Reusable across regression projects — nothing here is specific to a single dataset.
"""

# Standard library
from __future__ import annotations
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Local
from red_wine_quality.config import (
    COLUMN_RENAME,
    RANDOM_SEED,
    SKEW_CANDIDATES,
    TEST_SIZE,
)


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw wine quality CSV and return a DataFrame with safe column names."""
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_RENAME)
    return df


def assess_data(df: pd.DataFrame) -> dict:
    """Return a summary dict and print a formatted report to stdout."""
    report = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "missing_per_column": df.isna().sum().to_dict(),
        "n_duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    print("=" * 55)
    print(f"  Shape     : {report['n_rows']:,} rows x {report['n_cols']} columns")
    print(f"  Duplicates: {report['n_duplicates']}")
    print(f"  Missing   : {sum(v > 0 for v in report['missing_per_column'].values())} columns affected")
    print("=" * 55)

    missing_cols = {k: v for k, v in report["missing_per_column"].items() if v > 0}
    if missing_cols:
        print("\nMissing values:")
        for col, n in missing_cols.items():
            print(f"  {col:<30} {n}")
    else:
        print("\nNo missing values. The dataset is complete.")

    return report


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a transposed describe() augmented with skewness, kurtosis, and CV."""
    desc = df.describe().T
    desc["cv"] = (desc["std"] / desc["mean"]).round(4)
    desc["skew"] = df.skew(numeric_only=True).round(4)
    desc["kurt"] = df.kurt(numeric_only=True).round(4)
    return desc


def outlier_summary(df: pd.DataFrame, target: str = "quality") -> pd.DataFrame:
    """IQR-based outlier count and percentage for each numeric column."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    rows = []
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        rows.append({
            "Feature": col,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "Lower fence": round(lower, 4),
            "Upper fence": round(upper, 4),
            "Outliers": n_out,
            "Outlier %": round(n_out / len(df) * 100, 2),
        })
    result = pd.DataFrame(rows).set_index("Feature").sort_values("Outliers", ascending=False)

    print("IQR-Based Outlier Summary")
    print("=" * 55)
    print(result.to_string())
    print()
    total = result["Outliers"].sum()
    print(f"Total outlier flags: {total} across {len(numeric_cols)} features.")
    print("Note: a single row can be flagged by multiple features.")
    return result


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log1p-transformed columns for skewed predictors listed in SKEW_CANDIDATES."""
    out = df.copy()
    for col in SKEW_CANDIDATES:
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(out[col])
    return out


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    stratify_col: str = "quality",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into training and hold-out sets with stratification on the target."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_col],
    )

    print(f"Training set : {train_df.shape[0]:,} rows ({100 - test_size * 100:.0f}%)")
    print(f"Hold-out set : {test_df.shape[0]:,} rows ({test_size * 100:.0f}%)")

    dist = pd.DataFrame({
        "train": train_df[stratify_col].value_counts(normalize=True).sort_index(),
        "hold-out": test_df[stratify_col].value_counts(normalize=True).sort_index(),
    }).round(3)
    print(f"\n{stratify_col.capitalize()} distribution (proportions):")
    print(dist.to_string())

    return train_df, test_df


def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factor for a list of features, sorted descending."""
    X = sm.add_constant(df[features].dropna())
    vif_values = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    vif_df = (
        pd.DataFrame({"Feature": X.columns, "VIF": vif_values})
        .set_index("Feature")
        .drop("const", errors="ignore")
        .sort_values("VIF", ascending=False)
    )
    return vif_df


def analyze_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Compare feature distributions between unique and duplicate rows using Mann-Whitney U.

    Returns
    -------
    result : pd.DataFrame
        Per-feature comparison (means, medians, U-statistic, p-value).
    safe_to_remove : bool
        True when no feature shows a significant distributional difference.
    """
    dup_mask = df.duplicated(keep=False)
    unique_df = df[~dup_mask]
    dup_df = df[dup_mask]

    rows = []
    for col in df.columns:
        u_stat, p_val = stats.mannwhitneyu(
            unique_df[col], dup_df[col], alternative="two-sided"
        )
        rows.append({
            "Feature": col,
            "Mean (unique)": round(unique_df[col].mean(), 4),
            "Mean (dupes)": round(dup_df[col].mean(), 4),
            "Median (unique)": round(unique_df[col].median(), 4),
            "Median (dupes)": round(dup_df[col].median(), 4),
            "MW U-stat": round(u_stat, 1),
            "p-value": round(p_val, 4),
            "Significant": p_val < 0.05,
        })

    result = pd.DataFrame(rows).set_index("Feature")
    safe_to_remove = not result["Significant"].any()

    n_dup = dup_mask.sum()
    n_unique = (~dup_mask).sum()
    print(f"Duplicate rows : {n_dup:,}  ({n_dup / len(df) * 100:.1f}%)")
    print(f"Unique rows    : {n_unique:,}  ({n_unique / len(df) * 100:.1f}%)")
    print()
    print(result.to_string())
    print()

    if safe_to_remove:
        print("Verdict: REMOVE duplicates.")
        print("No feature shows a significant distributional difference (all p >= 0.05).")
        print("The duplicates are a random sample of the full dataset; removing them")
        print("does not introduce systematic bias.")
    else:
        sig = result[result["Significant"]].index.tolist()
        print(f"Note: {len(sig)} feature(s) differ between groups: {sig}")
        print("Certain profiles are over-represented among duplicates (e.g. common")
        print("quality scores). Removing exact duplicates is still valid — they carry")
        print("no new information — but the marginal distributions of these features")
        print("will shift slightly. This is expected and acceptable.")

    return result, safe_to_remove


def analyze_feature_redundancy(
    df: pd.DataFrame,
    feature: str,
    related_feature: str,
    target: str = "quality",
    controls: list[str] | None = None,
) -> dict:
    """Three-test framework to decide whether a feature is redundant.

    Tests
    -----
    1. Pearson + Spearman correlation between *feature* and *related_feature*.
    2. VIF when both features are included among all numeric predictors.
    3. Regression p-value of *feature* when *related_feature* is controlled.

    Returns a dict with test results and a ``drop`` verdict.
    """
    # Test 1: Correlation
    r_pearson = df[feature].corr(df[related_feature])
    r_spearman = df[feature].corr(df[related_feature], method="spearman")

    # Test 2: VIF
    all_numeric = df.select_dtypes(include="number").columns.tolist()
    predictors = [c for c in all_numeric if c != target]
    vif_df = compute_vif(df, predictors)
    vif_feature = vif_df.loc[feature, "VIF"] if feature in vif_df.index else None

    # Test 3: Regression p-value controlling for related feature
    if controls is None:
        controls = [c for c in predictors if c not in (feature, related_feature)]
    formula = f"{target} ~ {feature} + {related_feature}"
    if controls:
        formula += " + " + " + ".join(controls)
    model = smf.ols(formula, data=df).fit()
    p_feature = model.pvalues[feature]

    # Report
    print(f"Redundancy analysis: {feature}")
    print("-" * 55)
    print(f"  Test 1 — Correlation with {related_feature}:")
    print(f"    Pearson r  = {r_pearson:.3f}")
    print(f"    Spearman ρ = {r_spearman:.3f}")
    if vif_feature is not None:
        print(f"  Test 2 — VIF: {vif_feature:.2f}")
    print(f"  Test 3 — Regression p-value (controlling for {related_feature}): {p_feature:.4f}")
    print()

    drop = p_feature >= 0.05
    if drop:
        print(f"  Verdict: DROP {feature}.")
        print(f"  Non-significant (p={p_feature:.4f}) once {related_feature} is controlled.")
    else:
        print(f"  Verdict: KEEP {feature} — adds significant information (p={p_feature:.4f}).")

    return {
        "pearson_r": r_pearson,
        "spearman_rho": r_spearman,
        "vif": vif_feature,
        "p_value": p_feature,
        "drop": drop,
    }

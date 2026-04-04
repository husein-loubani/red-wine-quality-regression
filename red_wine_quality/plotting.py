"""
plotting.py
All reusable Matplotlib / Seaborn visualisation functions for the
Red Wine Quality Regression project.

Every function follows the same contract:
  - accepts data and optional Axes
  - returns the Figure (so the caller can save it if needed)
  - does NOT call plt.show()
"""

# Standard library
from __future__ import annotations
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats

# Local
from red_wine_quality.config import ACCENT_COLOR, BLUE_COLOR, PALETTE, FIGURE_DPI


def save_figure(fig: plt.Figure, name: str, figures_dir: str | Path) -> None:
    """Save a Figure to the reports/figures directory as a high-resolution PNG."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


def apply_global_style() -> None:
    """Apply the project-wide Seaborn and Matplotlib theme. Call once per notebook."""
    sns.set_theme(
        style="white",
        palette=PALETTE,
        font_scale=1.05,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        },
    )
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def plot_outlier_boxplots(df: pd.DataFrame) -> plt.Figure:
    """Grid of horizontal boxplots for all numeric columns to visualise outliers."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    n_cols = 4
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2.8))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        sns.boxplot(x=df[col], ax=axes[i], color=ACCENT_COLOR if n_out > 0 else BLUE_COLOR,
                    fliersize=3, linewidth=0.8)
        axes[i].set_title(f"{col}\n{n_out} outliers ({n_out / len(df) * 100:.1f}%)", fontsize=8)
        axes[i].set_xlabel("")

    for ax in axes[len(numeric_cols):]:
        ax.set_visible(False)

    fig.suptitle("Outlier Screen (IQR Method)", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_target_distribution(df: pd.DataFrame, target: str = "quality") -> plt.Figure:
    """Bar chart and cumulative percentage for the target variable."""
    counts = df[target].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bars = axes[0].bar(
        counts.index, counts.values,
        color=sns.color_palette("Reds_r", len(counts)),
        edgecolor="white", linewidth=0.8,
    )
    for bar, count in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{count}\n({count / len(df) * 100:.1f}%)",
            ha="center", va="bottom", fontsize=8,
        )
    axes[0].set_xlabel("Quality Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Quality Score Distribution")
    axes[0].set_xticks(counts.index)

    cum_pct = counts.cumsum() / len(df) * 100
    axes[1].plot(cum_pct.index, cum_pct.values, marker="o", color=ACCENT_COLOR, linewidth=2)
    axes[1].axhline(50, linestyle="--", color="grey", linewidth=0.8, label="50%")
    axes[1].axhline(80, linestyle="--", color="steelblue", linewidth=0.8, label="80%")
    axes[1].set_xlabel("Quality Score")
    axes[1].set_ylabel("Cumulative %")
    axes[1].set_title("Cumulative Quality Distribution")
    axes[1].legend(fontsize=8)
    axes[1].set_xticks(counts.index)

    n = len(df)
    fig.suptitle(f"Target Variable: Quality Score  (n = {n:,})", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_univariate_distributions(
    df: pd.DataFrame,
    predictors: list[str],
) -> plt.Figure:
    """Grid of histograms with KDE, mean and median lines, and skewness labels."""
    n_cols = 4
    n_rows = int(np.ceil(len(predictors) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(predictors):
        skew_val = df[col].skew()
        sns.histplot(df[col], ax=axes[i], kde=True,
                     color=ACCENT_COLOR, alpha=0.6, bins=30, linewidth=0)
        axes[i].axvline(df[col].median(), color="black",     linestyle="--", linewidth=1.2, label="median")
        axes[i].axvline(df[col].mean(),   color="steelblue", linestyle="-",  linewidth=1.2, label="mean")
        axes[i].set_title(f"{col}\nskew={skew_val:.2f}", fontsize=9)
        axes[i].set_xlabel("")
        axes[i].legend(fontsize=7, loc="upper right")

    for ax in axes[len(predictors):]:
        ax.set_visible(False)

    fig.suptitle("Univariate Distributions of All Predictor Variables", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_predictors_vs_target(
    df: pd.DataFrame,
    predictors: list[str],
    target: str = "quality",
) -> plt.Figure:
    """Grid of boxplots, each predictor broken down by target score with r annotated."""
    n_cols = 4
    n_rows = int(np.ceil(len(predictors) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(predictors):
        corr_val = df[col].corr(df[target])
        sns.boxplot(
            data=df, x=target, y=col, ax=axes[i],
            hue=target, palette="Reds_r", legend=False,
            linewidth=0.8, fliersize=2,
        )
        axes[i].set_title(f"{col}\nr = {corr_val:.2f}", fontsize=9)
        axes[i].set_xlabel("Quality Score")
        axes[i].set_ylabel("")

    for ax in axes[len(predictors):]:
        ax.set_visible(False)

    fig.suptitle("Predictor Distributions by Quality Score", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_correlation_bar(
    df: pd.DataFrame,
    target: str = "quality",
) -> plt.Figure:
    """Side-by-side Pearson and Spearman correlation bars, ranked by absolute Pearson."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    predictors = [c for c in numeric_cols if c != target]
    sub = df[predictors + [target]]

    pearson = sub.corr(method="pearson", numeric_only=True)[target].drop(target)
    spearman = sub.corr(method="spearman", numeric_only=True)[target].drop(target)
    order = pearson.abs().sort_values().index
    pearson = pearson[order]
    spearman = spearman[order]

    x = np.arange(len(order))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, corr, label, metric in zip(
        axes,
        [pearson, spearman],
        ["Pearson r", "Spearman ρ"],
        ["Pearson r", "Spearman ρ"],
    ):
        colors = [ACCENT_COLOR if v < 0 else BLUE_COLOR for v in corr.values]
        bars = ax.barh(x, corr.values, color=colors, edgecolor="white", height=0.65)
        for bar, val in zip(bars, corr.values):
            offset = 0.005 if val >= 0 else -0.005
            ha = "left" if val >= 0 else "right"
            ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha=ha, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"{metric} with {target}")
        ax.set_title(f"{label} — bivariate correlations with {target}", fontsize=11)
        ax.set_yticks(x)
        ax.set_yticklabels(order, fontsize=9)

    legend_elements = [
        mpatches.Patch(facecolor=BLUE_COLOR,   label="Positive association"),
        mpatches.Patch(facecolor=ACCENT_COLOR, label="Negative association"),
    ]
    axes[1].legend(handles=legend_elements, fontsize=8, loc="lower right")
    fig.suptitle(
        "Bivariate Correlations with Quality: Pearson r vs Spearman ρ",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Lower-triangle Pearson correlation heatmap for all numeric columns."""
    corr_matrix = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.4, linecolor="white",
        annot_kws={"size": 8}, ax=ax,
    )
    ax.set_title("Pairwise Pearson Correlation Matrix", fontsize=14, pad=12)
    fig.tight_layout()
    return fig


def plot_log_transform_comparison(
    df: pd.DataFrame,
    skew_candidates: list[str],
) -> plt.Figure:
    """Before and after log1p transformation side-by-side histograms."""
    fig, axes = plt.subplots(2, len(skew_candidates),
                             figsize=(4 * len(skew_candidates), 7))

    for i, col in enumerate(skew_candidates):
        sns.histplot(df[col], ax=axes[0, i], kde=True,
                     color=ACCENT_COLOR, bins=30, linewidth=0, alpha=0.7)
        axes[0, i].set_title(f"{col}\nskew={df[col].skew():.2f}", fontsize=8)
        axes[0, i].set_xlabel("")

        log_col = np.log1p(df[col])
        sns.histplot(log_col, ax=axes[1, i], kde=True,
                     color=BLUE_COLOR, bins=30, linewidth=0, alpha=0.7)
        axes[1, i].set_title(f"log1p({col})\nskew={log_col.skew():.2f}", fontsize=8)
        axes[1, i].set_xlabel("")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Log-transformed", fontsize=10)

    fig.suptitle("Effect of log1p Transformation on Right-Skewed Variables", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_vif(
    vif_before: pd.DataFrame,
    vif_after: pd.DataFrame,
) -> plt.Figure:
    """Side-by-side VIF bar charts, before and after removing density."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    titles = ["VIF (including density)", "VIF (after removing density)"]

    for ax, vif_df, title in zip(axes, [vif_before, vif_after], titles):
        bar_colors = [
            ACCENT_COLOR if v > 5 else ("orange" if v > 3 else BLUE_COLOR)
            for v in vif_df["VIF"]
        ]
        bars = ax.barh(vif_df.index, vif_df["VIF"],
                       color=bar_colors, edgecolor="white", height=0.6)
        ax.axvline(5,  color=ACCENT_COLOR, linestyle="--", linewidth=1.0, label="VIF = 5")
        ax.axvline(10, color="darkred",    linestyle="--", linewidth=1.0, label="VIF = 10")
        for bar, val in zip(bars, vif_df["VIF"]):
            ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8)
        ax.set_xlabel("VIF")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle("Variance Inflation Factor: Multicollinearity Assessment", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_coefficients(coef_df: pd.DataFrame) -> plt.Figure:
    """Horizontal error-bar plot of OLS coefficients with 95% CIs."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (idx, row) in enumerate(coef_df.iterrows()):
        color = ACCENT_COLOR if row["coef"] < 0 else BLUE_COLOR
        alpha_val = 1.0 if row["significant"] else 0.38
        ax.errorbar(
            row["coef"], i,
            xerr=[[row["coef"] - row["ci_low"]], [row["ci_high"] - row["coef"]]],
            fmt="o", color=color, alpha=alpha_val,
            markersize=7, elinewidth=1.5, capsize=4,
        )
        sig_marker = "  *" if row["significant"] else ""
        ax.text(row["coef"], i + 0.35,
                f"{row['coef']:.3f}{sig_marker}", ha="center", fontsize=7.5)

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_yticks(range(len(coef_df)))
    ax.set_yticklabels(coef_df.index, fontsize=9)
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title("OLS Coefficients (significant predictors marked with *)", fontsize=12)

    legend_elements = [
        mpatches.Patch(facecolor=BLUE_COLOR,   label="Positive effect"),
        mpatches.Patch(facecolor=ACCENT_COLOR, label="Negative effect"),
        plt.Line2D([0], [0], marker="o", color="grey", alpha=0.38,
                   markersize=8, label="Not significant (alpha=0.05)", linewidth=0),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_standardized_coefficients(std_df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of standardized (beta) coefficients for cross-variable comparison."""
    std_df = std_df.sort_values("std_coef")
    colors = [ACCENT_COLOR if v < 0 else BLUE_COLOR for v in std_df["std_coef"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(std_df.index, std_df["std_coef"], color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, std_df["std_coef"]):
        offset = 0.005 if val >= 0 else -0.005
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha=ha, fontsize=8)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized coefficient (beta)")
    ax.set_title("Standardized Coefficients: Relative Importance on a Common Scale", fontsize=12)

    legend_elements = [
        mpatches.Patch(facecolor=BLUE_COLOR,   label="Positive association"),
        mpatches.Patch(facecolor=ACCENT_COLOR, label="Negative association"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_residual_diagnostics(model) -> plt.Figure:
    """2x2 residual diagnostic panel."""
    fitted = model.fittedvalues
    residuals = model.resid
    std_resid = residuals / residuals.std()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted, residuals, alpha=0.3, s=12, color=BLUE_COLOR)
    axes[0, 0].axhline(0, color=ACCENT_COLOR, linewidth=1.2, linestyle="--")
    lw = lowess(residuals, fitted, frac=0.4)
    axes[0, 0].plot(lw[:, 0], lw[:, 1], color="black", linewidth=1.5, label="lowess")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs. Fitted Values")
    axes[0, 0].legend(fontsize=8)

    # 2. Q-Q
    sm.qqplot(residuals, line="s", ax=axes[0, 1], alpha=0.4, markersize=3, color=BLUE_COLOR)
    axes[0, 1].set_title("Normal Q-Q Plot")

    # 3. Scale-Location
    axes[1, 0].scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.3, s=12, color=BLUE_COLOR)
    axes[1, 0].axhline(1, color=ACCENT_COLOR, linewidth=1.0, linestyle="--")
    lw2 = lowess(np.sqrt(np.abs(std_resid)), fitted, frac=0.4)
    axes[1, 0].plot(lw2[:, 0], lw2[:, 1], color="black", linewidth=1.5)
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("sqrt(|Standardised Residuals|)")
    axes[1, 0].set_title("Scale-Location (Homoscedasticity Check)")

    # 4. Residual histogram
    sns.histplot(residuals, kde=True, bins=40, ax=axes[1, 1],
                 color=ACCENT_COLOR, alpha=0.6, linewidth=0)
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    norm_pdf = stats.norm.pdf(x_range, residuals.mean(), residuals.std())
    ax2_twin = axes[1, 1].twinx()
    ax2_twin.plot(x_range, norm_pdf, color="black", linewidth=1.5, label="Normal fit")
    ax2_twin.set_ylabel("Density")
    ax2_twin.legend(fontsize=8)
    axes[1, 1].set_xlabel("Residual")
    axes[1, 1].set_title("Residual Distribution vs. Normal")

    fig.suptitle("Residual Diagnostics", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_holdout_evaluation(actuals: pd.Series, predictions: pd.Series,
                            rmse: float, r2: float) -> plt.Figure:
    """Actual vs Predicted scatter and error distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    lo, hi = actuals.min() - 0.3, actuals.max() + 0.3
    axes[0].scatter(actuals, predictions, alpha=0.4, s=18, color=BLUE_COLOR)
    axes[0].plot([lo, hi], [lo, hi], color=ACCENT_COLOR, linewidth=1.5,
                 linestyle="--", label="Perfect prediction")
    axes[0].set_xlabel("Actual Quality")
    axes[0].set_ylabel("Predicted Quality")
    axes[0].set_title(f"Actual vs Predicted (Hold-out)\nRMSE={rmse:.3f}  R-sq={r2:.3f}")
    axes[0].legend(fontsize=8)

    errors = actuals - predictions
    sns.histplot(errors, kde=True, bins=30, ax=axes[1],
                 color=ACCENT_COLOR, alpha=0.6, linewidth=0)
    axes[1].axvline(0, color="black", linewidth=1.2, linestyle="--")
    axes[1].axvline(errors.mean(), color=BLUE_COLOR, linewidth=1.5,
                    linestyle="-", label=f"Mean error = {errors.mean():.3f}")
    axes[1].set_xlabel("Prediction Error (actual minus predicted)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Hold-out Prediction Error Distribution")
    axes[1].legend(fontsize=8)

    fig.suptitle("Hold-out Set Evaluation", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_violin_pair(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    target: str = "quality",
    palette1: str = "Greens",
    palette2: str = "Reds_r",
) -> plt.Figure:
    """Side-by-side violin plots of two predictors broken down by quality score."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, col, pal in zip(axes, [col1, col2], [palette1, palette2]):
        sns.violinplot(data=df, x=target, y=col,
                       hue=target, palette=pal, legend=False,
                       inner="box", ax=ax, linewidth=0.8)
        means = df.groupby(target)[col].mean()
        ax.plot(range(len(means)), means.values,
                "o--", color="black", markersize=5, linewidth=1.2, label="group mean")
        ax.set_xlabel("Quality Score")
        ax.set_title(f"{col.replace('_', ' ').title()} by Quality Score", fontsize=12)
        ax.legend(fontsize=8)

    fig.suptitle("The Two Strongest Individual Predictors of Wine Quality",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_stratification_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = "quality",
) -> plt.Figure:
    """Side-by-side bar charts comparing train vs hold-out class distributions."""
    train_dist = train_df[target].value_counts(normalize=True).sort_index()
    test_dist = test_df[target].value_counts(normalize=True).sort_index()

    all_scores = sorted(set(train_dist.index) | set(test_dist.index))
    x = np.arange(len(all_scores))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width / 2, [train_dist.get(s, 0) for s in all_scores],
                   width, label="Train", color=BLUE_COLOR, edgecolor="white")
    bars2 = ax.bar(x + width / 2, [test_dist.get(s, 0) for s in all_scores],
                   width, label="Hold-out", color=ACCENT_COLOR, edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.1%}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(all_scores)
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Proportion")
    ax.set_title("Stratified Split: Train vs Hold-out Class Distribution")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def save_all_figures(
    *,
    df_clean: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: list[str],
    skew_candidates: list[str],
    vif_before: pd.DataFrame,
    vif_after: pd.DataFrame,
    coef_df: pd.DataFrame,
    std_coefs: pd.DataFrame,
    reduced_model,
    holdout_actuals: pd.Series,
    holdout_predictions: pd.Series,
    holdout_rmse: float,
    holdout_r2: float,
    figures_dir: str | Path,
) -> None:
    """Generate every project figure and save it to figures_dir as a PNG.

    Call once after the full notebook has been executed. All plot functions are
    invoked internally so the notebook export cell stays to a single line.
    """
    figs = [
        ("01_quality_distribution",            plot_target_distribution,       (df_clean,)),
        ("02_univariate_distributions",        plot_univariate_distributions,  (train_df, predictors)),
        ("03_predictors_vs_quality",           plot_predictors_vs_target,      (train_df, predictors)),
        ("04_correlation_bar_pearson_spearman",plot_correlation_bar,           (train_df,)),
        ("05_correlation_heatmap",             plot_correlation_heatmap,       (train_df,)),
        ("06_violin_alcohol_volatility",       plot_violin_pair,               (train_df, "alcohol", "volatile_acidity")),
        ("07_log_transforms",                  plot_log_transform_comparison,  (train_df, skew_candidates)),
        ("08_vif",                             plot_vif,                       (vif_before, vif_after)),
        ("09_coefficients_full",               plot_coefficients,              (coef_df,)),
        ("10_standardized_coefficients",       plot_standardized_coefficients, (std_coefs,)),
        ("11_residual_diagnostics",            plot_residual_diagnostics,      (reduced_model,)),
        ("12_holdout_evaluation",              plot_holdout_evaluation,        (holdout_actuals, holdout_predictions, holdout_rmse, holdout_r2)),
        ("13_stratification_check",            plot_stratification_check,      (train_df, test_df)),
    ]

    for name, func, args in figs:
        fig = func(*args)
        save_figure(fig, name, figures_dir)
        plt.close(fig)

    print(f"\nAll {len(figs)} figures saved to {Path(figures_dir).resolve()}")


# ── Interactive Plotly Dashboard ─────────────────────────────────────────────

def dashboard_red_wine(
    df: pd.DataFrame,
    coef_df: pd.DataFrame,
    std_coefs: pd.DataFrame,
    reduced_model,
    holdout_actuals,
    holdout_predictions,
    holdout_rmse: float,
    holdout_r2: float,
    vif_df: pd.DataFrame,
    *,
    out_path: str | Path,
) -> "plotly.graph_objects.Figure":
    """Interactive dark-themed executive dashboard for Red Wine Quality OLS."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_samples = len(df)
    n_predictors = len(reduced_model.params) - 1
    train_r2 = reduced_model.rsquared
    train_adj_r2 = reduced_model.rsquared_adj
    f_pvalue = reduced_model.f_pvalue

    fig = make_subplots(
        rows=3, cols=2,
        row_heights=[0.33, 0.33, 0.34],
        column_widths=[0.5, 0.5],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        subplot_titles=[
            "Standardized Coefficients (Relative Importance)",
            "OLS Coefficients with Direction",
            "Actual vs Predicted (Hold-out Set)",
            "Model Summary",
            "Residual Distribution",
            "Variance Inflation Factors (VIF)",
        ],
    )

    # ---- Row 1, Col 1: Standardized coefficients ----
    std_sorted = std_coefs.sort_values("std_coef", key=abs, ascending=True)
    bar_colors_std = ["#f97316" if v > 0 else "#6366f1" for v in std_sorted["std_coef"]]

    fig.add_trace(
        go.Bar(
            y=std_sorted.index,
            x=std_sorted["std_coef"],
            orientation="h",
            marker_color=bar_colors_std,
            text=[f"{v:+.3f}" for v in std_sorted["std_coef"]],
            textposition="outside",
            textfont=dict(size=10),
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Standardized Coefficient", row=1, col=1)

    # ---- Row 1, Col 2: Raw coefficients ----
    coef_sorted = coef_df.sort_values("coef", key=abs, ascending=True)
    bar_colors_raw = ["#f97316" if v > 0 else "#6366f1" for v in coef_sorted["coef"]]
    coef_labels = coef_sorted.index.tolist()

    fig.add_trace(
        go.Bar(
            y=coef_labels,
            x=coef_sorted["coef"],
            orientation="h",
            marker_color=bar_colors_raw,
            text=[f"{v:+.4f}" for v in coef_sorted["coef"]],
            textposition="outside",
            textfont=dict(size=10),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Coefficient (unstandardized)", row=1, col=2)

    # ---- Row 2, Col 1: Actual vs predicted scatter ----
    fig.add_trace(
        go.Scatter(
            x=holdout_actuals,
            y=holdout_predictions,
            mode="markers",
            marker=dict(color="#6366f1", size=6, opacity=0.5),
            showlegend=False,
        ),
        row=2, col=1,
    )
    min_val = min(holdout_actuals.min(), holdout_predictions.min())
    max_val = max(holdout_actuals.max(), holdout_predictions.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.update_xaxes(title_text="Actual Quality", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Quality", row=2, col=1)

    # ---- Row 2, Col 2: Model summary table ----
    summary_labels = [
        "R-squared (train)", "Adj. R-squared (train)",
        "R-squared (hold-out)", "RMSE (hold-out)",
        "F-statistic p-value", "Predictors", "Observations",
    ]
    summary_values = [
        f"{train_r2:.4f}", f"{train_adj_r2:.4f}",
        f"{holdout_r2:.4f}", f"{holdout_rmse:.4f}",
        f"{f_pvalue:.2e}", str(n_predictors), f"{n_samples:,}",
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Value</b>"],
                fill_color="#1e1b4b",
                font=dict(color="white", size=13),
                align="center",
                height=35,
            ),
            cells=dict(
                values=[summary_labels, summary_values],
                fill_color=[["#2d2a5e"] * len(summary_labels)],
                font=dict(color="white", size=12),
                align="center",
                height=30,
            ),
        ),
        row=2, col=2,
    )

    # ---- Row 3, Col 1: Residual histogram ----
    resids = reduced_model.resid
    fig.add_trace(
        go.Histogram(
            x=resids,
            nbinsx=40,
            marker_color="#6366f1",
            opacity=0.8,
            showlegend=False,
        ),
        row=3, col=1,
    )
    fig.update_xaxes(title_text="Residual", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)

    # ---- Row 3, Col 2: VIF bar chart ----
    vif_sorted = vif_df.sort_values("VIF", ascending=True)
    vif_colors = ["#ef4444" if v > 5 else "#22c55e" for v in vif_sorted["VIF"]]

    fig.add_trace(
        go.Bar(
            y=vif_sorted.index.tolist(),
            x=vif_sorted["VIF"],
            orientation="h",
            marker_color=vif_colors,
            text=[f"{v:.1f}" for v in vif_sorted["VIF"]],
            textposition="outside",
            textfont=dict(size=10),
            showlegend=False,
        ),
        row=3, col=2,
    )
    fig.update_xaxes(title_text="VIF", row=3, col=2)

    # ---- Layout ----
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f0e1a",
        plot_bgcolor="#1a1933",
        font=dict(family="Inter, system-ui, sans-serif", color="#e2e8f0", size=13),
        title=dict(
            text=(
                "<b>Red Wine Quality — OLS Regression Dashboard</b>"
                f"<br><span style='font-size:13px; color:#94a3b8'>"
                f"Samples: {n_samples:,} | Predictors: {n_predictors} | "
                f"Train R\u00b2: {train_r2:.4f} | Hold-out R\u00b2: {holdout_r2:.4f} | "
                f"RMSE: {holdout_rmse:.4f}</span>"
            ),
            font=dict(size=18, color="#c084fc"),
            x=0.5,
            xanchor="center",
        ),
        height=1100,
        margin=dict(t=100, b=40, l=60, r=60),
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=14, color="#d8b4fe")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True)
    return fig

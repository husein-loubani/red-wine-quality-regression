"""
Red Wine Quality Regression — source package.

Public API — import directly from red_wine_quality rather than from submodules:

    from red_wine_quality import load_data, fit_ols, plot_coefficients
"""

from red_wine_quality.data_processing import (
    load_data,
    assess_data,
    descriptive_stats,
    add_log_features,
    split_data,
    compute_vif,
    analyze_duplicates,
    analyze_feature_redundancy,
    outlier_summary,
)
from red_wine_quality.statistics import (
    fit_ols,
    print_fit_summary,
    coefficient_table,
    standardized_coefficients,
    evaluate_holdout,
    compare_models,
    compare_shared_coefficients,
    compute_rank_correlations,
    stepwise_summary,
    answer_hypothesis,
    jarque_bera_report,
)
from red_wine_quality.plotting import (
    apply_global_style,
    save_figure,
    plot_outlier_boxplots,
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
    save_all_figures,
)

__all__ = [
    # data_processing
    "load_data",
    "assess_data",
    "descriptive_stats",
    "add_log_features",
    "split_data",
    "compute_vif",
    "analyze_duplicates",
    "analyze_feature_redundancy",
    "outlier_summary",
    # statistics
    "fit_ols",
    "print_fit_summary",
    "coefficient_table",
    "standardized_coefficients",
    "evaluate_holdout",
    "compare_models",
    "compare_shared_coefficients",
    "compute_rank_correlations",
    "stepwise_summary",
    "answer_hypothesis",
    "jarque_bera_report",
    # plotting
    "apply_global_style",
    "save_figure",
    "plot_outlier_boxplots",
    "plot_target_distribution",
    "plot_univariate_distributions",
    "plot_predictors_vs_target",
    "plot_correlation_bar",
    "plot_correlation_heatmap",
    "plot_log_transform_comparison",
    "plot_vif",
    "plot_coefficients",
    "plot_standardized_coefficients",
    "plot_residual_diagnostics",
    "plot_holdout_evaluation",
    "plot_violin_pair",
    "plot_stratification_check",
    "save_all_figures",
]

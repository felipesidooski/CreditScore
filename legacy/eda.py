"""Dataset presentation and exploratory analysis functions."""

from legacy.archive.credit_score_experiments_full import (
    build_missing_report,
    make_eda_frame,
    numeric_summary_by_target,
    plot_numeric_by_target,
    plot_target_rate,
    present_dataset,
    run_multivariate_eda,
    run_univariate_eda,
    target_rate_by_column,
)

__all__ = [
    "build_missing_report",
    "make_eda_frame",
    "numeric_summary_by_target",
    "plot_numeric_by_target",
    "plot_target_rate",
    "present_dataset",
    "run_multivariate_eda",
    "run_univariate_eda",
    "target_rate_by_column",
]


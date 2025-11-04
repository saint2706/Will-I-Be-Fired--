"""Metrics computation with bootstrap confidence intervals.

This module provides utilities for computing classification metrics with
confidence intervals using nonparametric bootstrap resampling.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from ..constants import RANDOM_SEED
except ImportError:  # pragma: no cover - fallback for script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from constants import RANDOM_SEED


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    """Compute metric with bootstrap confidence interval.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray
        Predicted probabilities (for metrics that need them)
    metric_fn : Callable
        Metric function to evaluate (should accept y_true and y_pred/y_prob)
    n_bootstrap : int
        Number of bootstrap resamples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[float, float, float]
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices] if y_prob is not None else None

        try:
            # Try to compute metric on bootstrap sample
            if "roc_auc" in metric_fn.__name__ or "average_precision" in metric_fn.__name__:
                score = metric_fn(y_true_boot, y_prob_boot)
            else:
                score = metric_fn(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except (ValueError, ZeroDivisionError):
            # Skip invalid bootstrap samples (e.g., only one class present)
            continue

    bootstrap_scores = np.array(bootstrap_scores)

    # Compute percentiles for confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    point_estimate = np.mean(bootstrap_scores)
    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)

    return float(point_estimate), float(lower_bound), float(upper_bound)


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """Compute all classification metrics with confidence intervals.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray
        Predicted probabilities (for positive class)
    n_bootstrap : int
        Number of bootstrap resamples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with metrics as keys, each containing 'mean', 'ci_lower', 'ci_upper'
    """
    metrics_config = [
        ("accuracy", accuracy_score, False),
        ("precision", lambda yt, yp: precision_score(yt, yp, zero_division=0), False),
        ("recall", lambda yt, yp: recall_score(yt, yp, zero_division=0), False),
        ("f1", lambda yt, yp: f1_score(yt, yp, zero_division=0), False),
        ("roc_auc", roc_auc_score, True),
        ("pr_auc", average_precision_score, True),
        ("brier", brier_score_loss, True),
    ]

    results = {}

    for metric_name, metric_fn, uses_proba in metrics_config:
        mean_val, lower_val, upper_val = bootstrap_metric(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob if uses_proba else None,
            metric_fn=metric_fn,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        )

        results[metric_name] = {"mean": mean_val, "ci_lower": lower_val, "ci_upper": upper_val}

    return results


def format_metric_with_ci(metric_dict: Dict[str, float]) -> str:
    """Format a metric dictionary as a string with CI.

    Parameters
    ----------
    metric_dict : Dict[str, float]
        Dictionary with 'mean', 'ci_lower', 'ci_upper' keys

    Returns
    -------
    str
        Formatted string like "0.950 (0.920-0.980)"
    """
    return f"{metric_dict['mean']:.3f} ({metric_dict['ci_lower']:.3f}-{metric_dict['ci_upper']:.3f})"

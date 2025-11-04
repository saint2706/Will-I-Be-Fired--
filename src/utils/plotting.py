"""Plotting utilities for model diagnostics and reporting.

This module provides functions to generate diagnostic plots including:
- ROC curves with bootstrap confidence bands
- Precision-Recall curves with bootstrap confidence bands
- Calibration (reliability) curves
- Feature importance plots
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

try:
    from ..constants import RANDOM_SEED
except ImportError:  # pragma: no cover - fallback for script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from constants import RANDOM_SEED


def plot_roc_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    n_bootstrap: int = 1000,
    random_state: int = RANDOM_SEED,
) -> None:
    """Plot ROC curve with bootstrap confidence bands.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities for positive class
    output_path : Path
        Path to save the figure
    n_bootstrap : int
        Number of bootstrap iterations
    random_state : int
        Random seed
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)

    # Compute main ROC curve
    fpr_main, tpr_main, _ = roc_curve(y_true, y_prob)
    roc_auc_main = auc(fpr_main, tpr_main)

    # Bootstrap for confidence bands
    tpr_interp_list = []
    base_fpr = np.linspace(0, 1, 100)

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue

        fpr_boot, tpr_boot, _ = roc_curve(y_true[indices], y_prob[indices])
        tpr_interp = np.interp(base_fpr, fpr_boot, tpr_boot)
        tpr_interp[0] = 0.0
        tpr_interp_list.append(tpr_interp)

    tpr_interp_array = np.array(tpr_interp_list)
    tpr_lower = np.percentile(tpr_interp_array, 2.5, axis=0)
    tpr_upper = np.percentile(tpr_interp_array, 97.5, axis=0)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_main, tpr_main, color="darkblue", lw=2, label=f"ROC curve (AUC = {roc_auc_main:.3f})")
    plt.fill_between(base_fpr, tpr_lower, tpr_upper, color="lightblue", alpha=0.3, label="95% CI")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve with Bootstrap Confidence Interval", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pr_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    n_bootstrap: int = 1000,
    random_state: int = RANDOM_SEED,
) -> None:
    """Plot Precision-Recall curve with bootstrap confidence bands.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities for positive class
    output_path : Path
        Path to save the figure
    n_bootstrap : int
        Number of bootstrap iterations
    random_state : int
        Random seed
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)

    # Compute main PR curve
    precision_main, recall_main, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_main = auc(recall_main, precision_main)

    # Bootstrap for confidence bands
    precision_interp_list = []
    base_recall = np.linspace(0, 1, 100)

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue

        precision_boot, recall_boot, _ = precision_recall_curve(y_true[indices], y_prob[indices])
        # Reverse for interpolation (recall should be increasing)
        precision_boot = precision_boot[::-1]
        recall_boot = recall_boot[::-1]
        precision_interp = np.interp(base_recall, recall_boot, precision_boot)
        precision_interp_list.append(precision_interp)

    precision_interp_array = np.array(precision_interp_list)
    precision_lower = np.percentile(precision_interp_array, 2.5, axis=0)
    precision_upper = np.percentile(precision_interp_array, 97.5, axis=0)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall_main, precision_main, color="darkgreen", lw=2, label=f"PR curve (AUC = {pr_auc_main:.3f})")
    plt.fill_between(base_recall, precision_lower, precision_upper, color="lightgreen", alpha=0.3, label="95% CI")
    baseline = y_true.sum() / len(y_true)
    plt.axhline(y=baseline, color="k", linestyle="--", lw=1, label=f"Baseline (prevalence = {baseline:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve with Bootstrap Confidence Interval", fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Plot calibration (reliability) curve and compute Brier score.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities for positive class
    output_path : Path
        Path to save the figure
    n_bins : int
        Number of bins for calibration curve

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        (brier_score, fraction_of_positives, mean_predicted_value)
    """
    from sklearn.metrics import brier_score_loss

    brier_score = brier_score_loss(y_true, y_prob)
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", color="darkred", label="Model")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives", fontsize=12)
    plt.title(f"Calibration Curve (Brier Score = {brier_score:.4f})", fontsize=14)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return brier_score, fraction_of_positives, mean_predicted_value


def plot_feature_importance(
    feature_names: list, importance_values: np.ndarray, output_path: Path, top_k: int = 20
) -> None:
    """Plot feature importance as a horizontal bar chart.

    Parameters
    ----------
    feature_names : list
        Names of features
    importance_values : np.ndarray
        Importance scores for each feature
    output_path : Path
        Path to save the figure
    top_k : int
        Number of top features to display
    """
    # Sort features by importance
    indices = np.argsort(importance_values)[::-1][:top_k]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance_values[indices]

    # Plot
    plt.figure(figsize=(10, max(6, top_k * 0.3)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    plt.barh(range(len(top_features)), top_importance, color=colors)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("Permutation Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"Top {top_k} Feature Importances", fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

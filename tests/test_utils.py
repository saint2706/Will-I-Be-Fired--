"""Tests for utility modules: repro, metrics, and plotting."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from constants import RANDOM_SEED  # noqa: E402
from utils.metrics import bootstrap_metric, compute_metrics_with_ci, format_metric_with_ci  # noqa: E402
from utils.repro import set_global_seed  # noqa: E402


def test_set_global_seed():
    """Test that set_global_seed sets random state consistently."""
    set_global_seed(42)
    val1 = np.random.rand()

    set_global_seed(42)
    val2 = np.random.rand()

    assert val1 == val2, "Random seed should produce reproducible results"


def test_set_global_seed_with_default():
    """Test that set_global_seed uses default RANDOM_SEED when None."""
    set_global_seed(None)
    val1 = np.random.rand()

    set_global_seed(RANDOM_SEED)
    val2 = np.random.rand()

    assert val1 == val2, "Default seed should match RANDOM_SEED"


def test_bootstrap_metric():
    """Test bootstrap confidence interval computation."""
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1] * 10)
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1] * 10)
    y_prob = np.random.rand(len(y_true))

    from sklearn.metrics import accuracy_score

    mean, lower, upper = bootstrap_metric(y_true, y_pred, y_prob, accuracy_score, n_bootstrap=100, random_state=42)

    # Check that mean is close to actual accuracy
    actual_acc = accuracy_score(y_true, y_pred)
    assert abs(mean - actual_acc) < 0.1, "Bootstrap mean should approximate actual metric"

    # Check that CI is reasonable
    assert lower <= mean <= upper, "Mean should be within CI bounds"
    assert 0 <= lower <= 1, "CI bounds should be valid probabilities"
    assert 0 <= upper <= 1, "CI bounds should be valid probabilities"


def test_compute_metrics_with_ci():
    """Test computation of multiple metrics with CIs."""
    np.random.seed(42)
    n = 100
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    y_prob = np.random.rand(n)

    metrics = compute_metrics_with_ci(y_true, y_pred, y_prob, n_bootstrap=50, random_state=42)

    # Check that all expected metrics are present
    expected_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        assert "mean" in metrics[metric], f"Missing 'mean' for {metric}"
        assert "ci_lower" in metrics[metric], f"Missing 'ci_lower' for {metric}"
        assert "ci_upper" in metrics[metric], f"Missing 'ci_upper' for {metric}"

        # Check validity of CI
        assert (
            metrics[metric]["ci_lower"] <= metrics[metric]["mean"] <= metrics[metric]["ci_upper"]
        ), f"Invalid CI for {metric}"


def test_format_metric_with_ci():
    """Test metric formatting."""
    metric_dict = {"mean": 0.95, "ci_lower": 0.92, "ci_upper": 0.98}
    formatted = format_metric_with_ci(metric_dict)

    assert "0.950" in formatted, "Should contain formatted mean"
    assert "0.920" in formatted, "Should contain formatted CI lower"
    assert "0.980" in formatted, "Should contain formatted CI upper"
    assert "(" in formatted and ")" in formatted, "Should have parentheses around CI"


def test_plotting_functions_run_without_error():
    """Test that plotting functions execute without errors."""
    from utils.plotting import (
        plot_calibration_curve,
        plot_feature_importance,
        plot_pr_with_ci,
        plot_roc_with_ci,
    )

    np.random.seed(42)
    n = 100
    y_true = np.random.randint(0, 2, n)
    y_prob = np.random.rand(n)
    feature_names = [f"feature_{i}" for i in range(10)]
    importance_values = np.random.rand(10)

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Test ROC plot
        plot_roc_with_ci(y_true, y_prob, tmpdir_path / "roc.png", n_bootstrap=10)
        assert (tmpdir_path / "roc.png").exists(), "ROC plot should be created"

        # Test PR plot
        plot_pr_with_ci(y_true, y_prob, tmpdir_path / "pr.png", n_bootstrap=10)
        assert (tmpdir_path / "pr.png").exists(), "PR plot should be created"

        # Test calibration plot
        brier, frac_pos, mean_pred = plot_calibration_curve(y_true, y_prob, tmpdir_path / "calibration.png")
        assert (tmpdir_path / "calibration.png").exists(), "Calibration plot should be created"
        assert 0 <= brier <= 1, "Brier score should be in [0, 1]"

        # Test feature importance plot
        plot_feature_importance(feature_names, importance_values, tmpdir_path / "importance.png")
        assert (tmpdir_path / "importance.png").exists(), "Feature importance plot should be created"

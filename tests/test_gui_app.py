"""Tests for GUI app helper functions."""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gui_app import _best_model_name, _is_model_entry, build_metrics_table  # noqa: E402


def test_is_model_entry():
    """Test the helper function that identifies model entries."""
    # Valid model entry with validation and test
    valid_entry = {"validation": {"roc_auc": 0.95}, "test": {"roc_auc": 0.98}}
    assert _is_model_entry(valid_entry) is True

    # Invalid entries
    assert _is_model_entry({"validation": {"roc_auc": 0.95}}) is False  # Missing test
    assert _is_model_entry({"test": {"roc_auc": 0.98}}) is False  # Missing validation
    assert _is_model_entry({"other": "data"}) is False  # Wrong structure
    assert _is_model_entry(None) is False  # Not a dict
    assert _is_model_entry("string") is False  # Not a dict
    assert _is_model_entry([]) is False  # Not a dict


def test_build_metrics_table_with_baselines():
    """Test that build_metrics_table correctly handles metrics with baselines entry."""
    metrics = {
        "logistic_regression": {
            "cv_best_score": 0.988,
            "validation": {"accuracy": 0.91, "precision": 1.0, "recall": 0.73, "roc_auc": 0.95},
            "test": {"accuracy": 0.98, "precision": 1.0, "recall": 0.94, "roc_auc": 0.996},
        },
        "random_forest": {
            "cv_best_score": 0.994,
            "validation": {"accuracy": 0.98, "precision": 1.0, "recall": 0.93, "roc_auc": 1.0},
            "test": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "roc_auc": 1.0},
        },
        "baselines": {
            "majority_class": {"accuracy": 0.66, "precision": 0.0, "recall": 0.0, "roc_auc": 0.5},
            "stratified_random": {"accuracy": 0.70, "precision": 0.58, "recall": 0.44, "roc_auc": 0.64},
        },
    }

    df = build_metrics_table(metrics)

    assert df is not None, "Should return a DataFrame"
    assert isinstance(df, pd.DataFrame), "Should return a pandas DataFrame"

    # Should only include models with validation/test splits, not baselines
    assert "logistic_regression" in df["model"].values
    assert "random_forest" in df["model"].values
    assert "baselines" not in df["model"].values

    # Should have entries for both validation and test splits
    assert "validation" in df["split"].values
    assert "test" in df["split"].values

    # Check that all expected metrics are present
    for metric in ["accuracy", "precision", "recall", "roc_auc"]:
        assert metric in df["metric"].values


def test_build_metrics_table_empty():
    """Test that build_metrics_table handles empty metrics."""
    assert build_metrics_table(None) is None
    # Empty dict also returns None due to the truthiness check
    assert build_metrics_table({}) is None


def test_best_model_name_with_baselines():
    """Test that _best_model_name correctly ignores baselines entry."""
    metrics = {
        "logistic_regression": {
            "validation": {"accuracy": 0.91, "precision": 1.0, "recall": 0.73, "roc_auc": 0.95},
            "test": {"accuracy": 0.98, "precision": 1.0, "recall": 0.94, "roc_auc": 0.996},
        },
        "random_forest": {
            "validation": {"accuracy": 0.98, "precision": 1.0, "recall": 0.93, "roc_auc": 1.0},
            "test": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "roc_auc": 1.0},
        },
        "gradient_boosting": {
            "validation": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "roc_auc": 0.98},
            "test": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "roc_auc": 1.0},
        },
        "baselines": {
            "majority_class": {"accuracy": 0.66, "precision": 0.0, "recall": 0.0, "roc_auc": 0.5},
            "stratified_random": {"accuracy": 0.70, "precision": 0.58, "recall": 0.44, "roc_auc": 0.64},
        },
    }

    best = _best_model_name(metrics)

    # Should select random_forest which has the highest validation roc_auc (1.0)
    assert best == "random_forest"


def test_best_model_name_empty():
    """Test that _best_model_name handles empty or None metrics."""
    assert _best_model_name(None) is None
    assert _best_model_name({}) is None


def test_best_model_name_only_baselines():
    """Test that _best_model_name returns None when only baselines are present."""
    metrics = {
        "baselines": {
            "majority_class": {"accuracy": 0.66, "precision": 0.0, "recall": 0.0, "roc_auc": 0.5},
        }
    }
    assert _best_model_name(metrics) is None


def test_build_metrics_table_with_incomplete_model():
    """Test that build_metrics_table handles incomplete model entries gracefully."""
    # Scenario 1: Model with only validation (missing test)
    metrics_missing_test = {
        "incomplete_model": {
            "cv_best_score": 0.9,
            "validation": {"accuracy": 0.91, "precision": 1.0},
        },
        "complete_model": {
            "validation": {"accuracy": 0.95, "precision": 0.98},
            "test": {"accuracy": 0.94, "precision": 0.97},
        },
    }
    df = build_metrics_table(metrics_missing_test)
    # Should only include the complete model
    assert "incomplete_model" not in df["model"].values
    assert "complete_model" in df["model"].values

    # Scenario 2: Model with only test (missing validation)
    metrics_missing_validation = {
        "incomplete_model": {
            "cv_best_score": 0.9,
            "test": {"accuracy": 0.91, "precision": 1.0},
        },
    }
    df = build_metrics_table(metrics_missing_validation)
    # Should filter out incomplete model
    assert df is not None
    assert len(df) == 0  # No valid models

    # Scenario 3: Model with only cv_best_score
    metrics_only_cv = {
        "incomplete_model": {
            "cv_best_score": 0.9,
        },
    }
    df = build_metrics_table(metrics_only_cv)
    assert df is not None
    assert len(df) == 0  # No valid models


def test_build_metrics_table_with_invalid_types():
    """Test that build_metrics_table handles invalid metric value types gracefully."""
    # Scenario 1: validation or test is None
    metrics_with_none = {
        "model_with_none": {
            "validation": None,
            "test": {"accuracy": 0.9},
        },
    }
    df = build_metrics_table(metrics_with_none)
    # Should filter out model with None value
    assert df is not None
    assert len(df) == 0

    # Scenario 2: validation or test is a list
    metrics_with_list = {
        "model_with_list": {
            "validation": [0.9, 0.8],
            "test": {"accuracy": 0.9},
        },
    }
    df = build_metrics_table(metrics_with_list)
    # Should filter out model with list value
    assert df is not None
    assert len(df) == 0

    # Scenario 3: Both are present but one is invalid
    metrics_mixed = {
        "valid_model": {
            "validation": {"accuracy": 0.95},
            "test": {"accuracy": 0.94},
        },
        "invalid_model": {
            "validation": {"accuracy": 0.91},
            "test": "invalid",
        },
    }
    df = build_metrics_table(metrics_mixed)
    # Should only include the valid model
    assert "valid_model" in df["model"].values
    assert "invalid_model" not in df["model"].values


def test_is_model_entry_with_invalid_types():
    """Test that _is_model_entry correctly rejects invalid metric structures."""
    # Valid entry
    assert _is_model_entry({"validation": {}, "test": {}}) is True

    # Invalid: validation is None
    assert _is_model_entry({"validation": None, "test": {}}) is False

    # Invalid: test is None
    assert _is_model_entry({"validation": {}, "test": None}) is False

    # Invalid: validation is a list
    assert _is_model_entry({"validation": [], "test": {}}) is False

    # Invalid: test is a string
    assert _is_model_entry({"validation": {}, "test": "invalid"}) is False

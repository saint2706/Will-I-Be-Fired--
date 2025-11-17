"""Bias and fairness diagnostics for the termination prediction models.

This module loads the persisted best model, recreates the canonical
train/validation/test split, and computes group-level performance metrics
for sensitive attributes that exist in the dataset (RaceDesc, Sex,
MaritalDesc, CitizenDesc).  The goal is to provide lightweight fairness
checks—demographic parity and equal opportunity gaps—without introducing
additional heavy dependencies.

The script can be executed directly or invoked through the Makefile via
`make fairness`.  Outputs are written to `reports/`:

* `fairness_summary.csv`: tidy table with per-group metrics and gaps
* `figures/fairness_<attribute>.png`: optional bar charts for quick review
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:  # pragma: no cover - import guard for script vs. module usage
    from .constants import RANDOM_SEED
    from .feature_engineering import prepare_training_data
    from .logging_utils import configure_logging, get_logger
    from .train_model import stratified_splits
except ImportError:  # pragma: no cover - fallback for CLI execution
    from constants import RANDOM_SEED  # type: ignore
    from feature_engineering import prepare_training_data  # type: ignore
    from logging_utils import configure_logging, get_logger  # type: ignore
    from train_model import stratified_splits  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "HRDataset_v14.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "fairness_summary.csv"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

PROTECTED_ATTRIBUTES = ("RaceDesc", "Sex", "MaritalDesc", "CitizenDesc")

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Return CLI arguments for fairness computation."""

    parser = argparse.ArgumentParser(description="Fairness/bias diagnostics for termination model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to HRDataset CSV")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to trained model pipeline")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_SUMMARY_PATH, help="Destination CSV for fairness summary"
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory to store fairness bar charts",
    )
    parser.add_argument(
        "--attributes",
        nargs="+",
        default=list(PROTECTED_ATTRIBUTES),
        help="Protected attributes to audit (must exist in processed feature set)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_SEED,
        help="Random seed so the canonical test split is reproduced",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip generating matplotlib figures (still writes CSV summary)",
    )
    return parser.parse_args()


def load_model(model_path: Path):
    """Return the persisted model pipeline."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run `make train` before the fairness analysis."
        )
    return joblib.load(model_path)


def prepare_test_split(data_path: Path, random_state: int):
    """Load data, engineer features, and return the canonical test set."""

    df = pd.read_csv(data_path)
    X, y = prepare_training_data(df)
    splits = stratified_splits(X, y, random_state=random_state)
    return splits.X_test.reset_index(drop=True), splits.y_test.reset_index(drop=True)


def _safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Compute ROC-AUC when both classes are present; otherwise return NaN."""

    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_brier(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if len(y_prob) == 0:
        return float("nan")
    return float(brier_score_loss(y_true, y_prob))


def compute_group_metrics(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    attributes: Iterable[str],
) -> pd.DataFrame:
    """Return tidy DataFrame with fairness metrics for each attribute/group."""

    records: List[dict] = []

    for attribute in attributes:
        if attribute not in X_test.columns:
            logger.warning("Attribute %s not found in test features; skipping", attribute)
            continue

        attr_series = X_test[attribute].fillna("Unknown").astype(str)
        attr_rows: List[dict] = []
        pred_rates: List[float] = []
        recalls: List[float] = []

        for group_value in sorted(attr_series.unique()):
            mask = attr_series == group_value
            if not mask.any():
                continue

            y_true_group = y_test[mask]
            y_pred_group = y_pred[mask]
            y_prob_group = y_prob[mask]

            accuracy = float(accuracy_score(y_true_group, y_pred_group))
            precision = float(precision_score(y_true_group, y_pred_group, zero_division=0))
            recall = float(recall_score(y_true_group, y_pred_group, zero_division=0))
            roc_auc = _safe_roc_auc(y_true_group, y_prob_group)
            brier = _safe_brier(y_true_group, y_prob_group)
            pred_positive_rate = float((y_pred_group == 1).mean())
            true_positive_rate = float(y_true_group.mean())

            attr_rows.append(
                {
                    "attribute": attribute,
                    "group": group_value,
                    "sample_size": int(mask.sum()),
                    "positive_rate": true_positive_rate,
                    "pred_positive_rate": pred_positive_rate,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "roc_auc": roc_auc,
                    "brier_score": brier,
                }
            )
            pred_rates.append(pred_positive_rate)
            recalls.append(recall)

        if not attr_rows:
            continue

        dp_diff = float(np.nanmax(pred_rates) - np.nanmin(pred_rates)) if len(pred_rates) > 1 else 0.0
        eo_diff = float(np.nanmax(recalls) - np.nanmin(recalls)) if len(recalls) > 1 else 0.0

        for row in attr_rows:
            row["demographic_parity_difference"] = dp_diff
            row["equal_opportunity_difference"] = eo_diff

        logger.info(
            "Attribute %s → demographic parity diff %.3f, equal opportunity diff %.3f",
            attribute,
            dp_diff,
            eo_diff,
        )
        records.extend(attr_rows)

    return pd.DataFrame(records)


def plot_attribute_bars(attribute: str, summary_df: pd.DataFrame, figures_dir: Path) -> None:
    """Persist a simple bar chart for positive prediction rates and recall per group."""

    import matplotlib.pyplot as plt

    attr_df = summary_df[summary_df["attribute"] == attribute]
    if attr_df.empty:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(attr_df["group"], attr_df["pred_positive_rate"], color="#1f77b4")
    axes[0].set_title("Positive prediction rate")
    axes[0].set_ylabel("Rate")
    axes[0].tick_params(axis="x", labelrotation=45)

    axes[1].bar(attr_df["group"], attr_df["recall"], color="#ff7f0e")
    axes[1].set_title("Recall (equal opportunity)")
    axes[1].tick_params(axis="x", labelrotation=45)

    fig.suptitle(f"Fairness diagnostics for {attribute}")
    fig.tight_layout()
    output_path = figures_dir / f"fairness_{attribute.lower()}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fairness figure for %s to %s", attribute, output_path)


def run_fairness_audit(args: argparse.Namespace) -> Path:
    """Execute fairness analysis end-to-end and return summary path."""

    model = load_model(args.model)
    X_test, y_test = prepare_test_split(args.data, random_state=args.random_state)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    summary_df = compute_group_metrics(X_test, y_test, y_pred, y_prob, args.attributes)
    if summary_df.empty:
        raise RuntimeError("No fairness metrics were computed; ensure attributes exist in the dataset.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    logger.info("Wrote fairness summary to %s", args.output)

    if not args.skip_figures:
        for attribute in summary_df["attribute"].unique():
            plot_attribute_bars(attribute, summary_df, args.figures_dir)

    return args.output


def main() -> None:
    configure_logging()
    args = parse_args()
    summary_path = run_fairness_audit(args)
    print(f"Fairness summary saved to {summary_path}")


if __name__ == "__main__":
    main()

"""Failure analysis for misclassified predictions.

This module identifies and analyzes false positives and false negatives
to understand model limitations and potential improvements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from ..logging_utils import get_logger
except ImportError:  # pragma: no cover - fallback for script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FailureCase:
    """Container for a single misclassified example."""

    index: int
    true_label: int
    predicted_label: int
    predicted_probability: float
    error_type: str  # "false_positive" or "false_negative"
    features: dict
    explanation: str


def identify_failure_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    top_n: int = 10,
) -> Tuple[List[FailureCase], List[FailureCase]]:
    """Identify top false positives and false negatives.

    Parameters
    ----------
    X_test : pd.DataFrame
        Test feature matrix
    y_test : pd.Series
        True test labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray
        Predicted probabilities for positive class
    top_n : int
        Number of top cases to return for each error type

    Returns
    -------
    Tuple[List[FailureCase], List[FailureCase]]
        (false_positives, false_negatives)
    """
    # Identify misclassifications
    errors = y_pred != y_test.values

    false_positives = []
    false_negatives = []

    for idx in X_test.index[errors]:
        loc_idx = X_test.index.get_loc(idx)
        true_label = int(y_test.iloc[loc_idx])
        pred_label = int(y_pred[loc_idx])
        pred_prob = float(y_prob[loc_idx])
        features = X_test.loc[idx].to_dict()

        # Generate simple explanation
        explanation = generate_explanation(features, true_label, pred_label, pred_prob)

        case = FailureCase(
            index=idx,
            true_label=true_label,
            predicted_label=pred_label,
            predicted_probability=pred_prob,
            error_type="false_positive" if pred_label == 1 else "false_negative",
            features=features,
            explanation=explanation,
        )

        if pred_label == 1 and true_label == 0:
            false_positives.append(case)
        elif pred_label == 0 and true_label == 1:
            false_negatives.append(case)

    # Sort by confidence (highest probability for FP, lowest for FN)
    false_positives = sorted(false_positives, key=lambda x: x.predicted_probability, reverse=True)[:top_n]
    false_negatives = sorted(false_negatives, key=lambda x: x.predicted_probability)[:top_n]

    logger.info(f"Identified {len(false_positives)} false positives and {len(false_negatives)} false negatives")

    return false_positives, false_negatives


def generate_explanation(features: dict, true_label: int, pred_label: int, pred_prob: float) -> str:
    """Generate a simple textual explanation for a misclassification.

    Parameters
    ----------
    features : dict
        Feature values for the instance
    true_label : int
        True label (0 or 1)
    pred_label : int
        Predicted label (0 or 1)
    pred_prob : float
        Predicted probability

    Returns
    -------
    str
        Human-readable explanation
    """
    explanations = []

    # Check tenure
    if "tenure_years" in features:
        tenure = features["tenure_years"]
        if tenure < 1.0:
            explanations.append("Very short tenure (<1 year)")
        elif tenure > 10.0:
            explanations.append("Long tenure (>10 years)")

    # Check performance
    if "PerformanceScore" in features:
        perf = features["PerformanceScore"]
        if perf in ["Needs Improvement", "PIP"]:
            explanations.append("Poor performance score")
        elif perf == "Exceeds":
            explanations.append("Strong performance score")

    # Check satisfaction
    if "EmpSatisfaction" in features:
        satisfaction = features["EmpSatisfaction"]
        if satisfaction <= 2:
            explanations.append("Low satisfaction score")
        elif satisfaction >= 4:
            explanations.append("High satisfaction score")

    # Check engagement
    if "EngagementSurvey" in features:
        engagement = features["EngagementSurvey"]
        if engagement <= 2.0:
            explanations.append("Low engagement score")
        elif engagement >= 4.0:
            explanations.append("High engagement score")

    # Check absences
    if "Absences" in features:
        absences = features["Absences"]
        if absences > 10:
            explanations.append(f"High absences ({absences})")
        elif absences == 0:
            explanations.append("Perfect attendance")

    # Check days late
    if "DaysLateLast30" in features:
        late_days = features["DaysLateLast30"]
        if late_days > 5:
            explanations.append(f"Frequently late ({late_days} days)")

    # Check age
    if "age_years" in features:
        age = features["age_years"]
        if age < 25:
            explanations.append("Young employee (<25)")
        elif age > 55:
            explanations.append("Senior employee (>55)")

    if not explanations:
        explanations.append("No obvious risk factors")

    error_type = "false positive" if pred_label == 1 and true_label == 0 else "false negative"
    base = f"{error_type.capitalize()} (prob={pred_prob:.2f}): "

    return base + "; ".join(explanations)


def save_failure_cases(
    false_positives: List[FailureCase],
    false_negatives: List[FailureCase],
    output_dir: Path,
) -> None:
    """Save failure cases to CSV and generate markdown report.

    Parameters
    ----------
    false_positives : List[FailureCase]
        List of false positive cases
    false_negatives : List[FailureCase]
        List of false negative cases
    output_dir : Path
        Directory to save outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine all cases for CSV
    all_cases = false_positives + false_negatives

    # Convert to DataFrame
    records = []
    for case in all_cases:
        record = {
            "index": case.index,
            "error_type": case.error_type,
            "true_label": case.true_label,
            "predicted_label": case.predicted_label,
            "predicted_probability": case.predicted_probability,
            "explanation": case.explanation,
        }
        # Add key features
        for key in ["tenure_years", "age_years", "PerformanceScore", "EmpSatisfaction", "EngagementSurvey", "Absences"]:
            if key in case.features:
                record[key] = case.features[key]
        records.append(record)

    df = pd.DataFrame(records)
    csv_path = output_dir / "failure_cases.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved failure cases to {csv_path}")

    # Generate markdown report
    write_failure_markdown(false_positives, false_negatives, output_dir / "FAILURES.md")


def write_failure_markdown(
    false_positives: List[FailureCase],
    false_negatives: List[FailureCase],
    output_path: Path,
) -> None:
    """Write failure analysis as markdown report.

    Parameters
    ----------
    false_positives : List[FailureCase]
        List of false positive cases
    false_negatives : List[FailureCase]
        List of false negative cases
    output_path : Path
        Path to save markdown file
    """
    with open(output_path, "w") as f:
        f.write("# Failure Analysis Report\n\n")
        f.write(
            "This report analyzes misclassified predictions to identify patterns "
            "and potential model improvements.\n\n"
        )

        f.write("## False Positives (Predicted Termination, Actually Retained)\n\n")
        f.write(
            f"We identified {len(false_positives)} false positives on the test set. "
            "These are employees the model incorrectly predicted would be terminated.\n\n"
        )

        if false_positives:
            f.write("### Representative Cases\n\n")
            for i, case in enumerate(false_positives[:3], 1):
                f.write(f"**Case {i}** (Probability: {case.predicted_probability:.2%})\n")
                f.write(f"- {case.explanation}\n")

                # Extract key features
                if "tenure_years" in case.features:
                    f.write(f"- Tenure: {case.features['tenure_years']:.1f} years\n")
                if "PerformanceScore" in case.features:
                    f.write(f"- Performance: {case.features['PerformanceScore']}\n")
                if "EmpSatisfaction" in case.features:
                    f.write(f"- Satisfaction: {case.features['EmpSatisfaction']}\n")
                if "EngagementSurvey" in case.features:
                    f.write(f"- Engagement: {case.features['EngagementSurvey']:.1f}\n")
                f.write("\n")

            f.write("**Pattern:** False positives often occur when employees have some risk factors ")
            f.write("(e.g., low satisfaction, short tenure) but other protective factors ")
            f.write("(e.g., strong performance, high engagement) keep them retained.\n\n")

        f.write("## False Negatives (Predicted Retention, Actually Terminated)\n\n")
        f.write(
            f"We identified {len(false_negatives)} false negatives on the test set. "
            "These are employees the model incorrectly predicted would be retained.\n\n"
        )

        if false_negatives:
            f.write("### Representative Cases\n\n")
            for i, case in enumerate(false_negatives[:3], 1):
                f.write(f"**Case {i}** (Probability: {case.predicted_probability:.2%})\n")
                f.write(f"- {case.explanation}\n")

                # Extract key features
                if "tenure_years" in case.features:
                    f.write(f"- Tenure: {case.features['tenure_years']:.1f} years\n")
                if "PerformanceScore" in case.features:
                    f.write(f"- Performance: {case.features['PerformanceScore']}\n")
                if "EmpSatisfaction" in case.features:
                    f.write(f"- Satisfaction: {case.features['EmpSatisfaction']}\n")
                if "EngagementSurvey" in case.features:
                    f.write(f"- Engagement: {case.features['EngagementSurvey']:.1f}\n")
                f.write("\n")

            f.write("**Pattern:** False negatives often occur when employees appear stable on observable metrics ")
            f.write("but have unobserved factors (e.g., external job offers, family circumstances, ")
            f.write("manager conflicts) that lead to termination.\n\n")

        f.write("## Recommendations for Model Improvement\n\n")
        f.write("1. **Additional Features**: Collect data on:\n")
        f.write("   - Manager tenure and management style\n")
        f.write("   - Peer network strength (e.g., number of close colleagues)\n")
        f.write("   - Career progression (promotions, lateral moves)\n")
        f.write("   - External market conditions (competitive job offers)\n\n")

        f.write("2. **Ensemble Methods**: Consider stacking models that specialize in:\n")
        f.write("   - Early terminations (<1 year tenure)\n")
        f.write("   - Performance-driven terminations\n")
        f.write("   - Voluntary quits vs. involuntary terminations\n\n")

        f.write("3. **Temporal Modeling**: Use time-series features:\n")
        f.write("   - Trend in satisfaction scores over time\n")
        f.write("   - Changes in absence patterns\n")
        f.write("   - Frequency of performance review score changes\n\n")

        f.write("4. **Calibration**: The model may need better calibration for edge cases:\n")
        f.write("   - Use CalibratedClassifierCV with isotonic regression\n")
        f.write("   - Consider cost-sensitive learning (weight false negatives more heavily)\n\n")

    logger.info(f"Wrote failure analysis markdown to {output_path}")


def analyze_failures_from_test(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model: Pipeline,
    output_dir: Optional[Path] = None,
    top_n: int = 10,
) -> None:
    """Run complete failure analysis pipeline.

    Parameters
    ----------
    X_test : pd.DataFrame
        Test feature matrix
    y_test : pd.Series
        True test labels
    model : Pipeline
        Trained model pipeline
    output_dir : Path, optional
        Output directory. If None, uses reports/
    top_n : int
        Number of top cases per error type
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "reports"

    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Identify failure cases
    false_positives, false_negatives = identify_failure_cases(X_test, y_test, y_pred, y_prob, top_n=top_n)

    # Save results
    save_failure_cases(false_positives, false_negatives, output_dir)

    logger.info("Failure analysis complete")

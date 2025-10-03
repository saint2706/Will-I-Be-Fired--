"""Command-line utility for predicting termination risk.

This script provides a command-line interface (CLI) for generating termination
risk predictions for a single employee. It supports two main modes of operation:

1.  **Interactive Mode:** If run without the `--employee-json` argument, the
    script will interactively prompt the user to enter the employee's details,
    field by field. Default values are provided for convenience.

2.  **File-based Mode:** The `--employee-json` argument allows specifying a
    path to a JSON file containing the employee's record. This is useful for
    programmatic use or for re-running predictions.

In both modes, the script uses the `predict_tenure_risk` function from the
`inference` module to calculate risk at specified tenure horizons and prints
a formatted table of the results to the console.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

try:
    from .inference import DEFAULT_MODEL_PATH, TenureRisk, predict_tenure_risk
except ImportError:  # pragma: no cover - fallback for script execution
    from inference import DEFAULT_MODEL_PATH, TenureRisk, predict_tenure_risk

try:
    from .logging_utils import configure_logging, get_logger
except ImportError:  # pragma: no cover - fallback for script execution
    from logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

# Default values for the interactive user prompts.
DEFAULT_PROMPTS = {
    "Department": "IT",
    "PerformanceScore": "Fully Meets",
    "RecruitmentSource": "Indeed",
    "Position": "IT Support",
    "State": "MA",
    "Sex": "Male",
    "MaritalDesc": "Single",
    "CitizenDesc": "US Citizen",
    "RaceDesc": "White",
    "HispanicLatino": "No",
    "Salary": "65000",
    "EngagementSurvey": "4.0",
    "EmpSatisfaction": "4",
    "SpecialProjectsCount": "3",
    "DaysLateLast30": "0",
    "Absences": "5",
    "DateofHire": "2018-07-01",
    "DOB": "1990-05-18",
    "LastPerformanceReview_Date": "2023-10-01",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the prediction script."""
    parser = argparse.ArgumentParser(description="Predict termination risk for an employee")
    parser.add_argument(
        "--employee-json",
        type=Path,
        help="Path to a JSON file with employee data. If omitted, prompts for interactive input.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the trained model pipeline")
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="*",
        default=None,
        help="Optional tenure horizons in years (default: 1, 2, 5)",
    )
    return parser.parse_args()


def _prompt_user(prompts: Dict[str, str]) -> Dict[str, str]:
    """Interactively prompt the user for employee details.

    Parameters
    ----------
    prompts:
        A dictionary mapping field names to their default string values.

    Returns
    -------
    A dictionary of the user's responses.
    """
    print("Enter employee information. Press Enter to accept the suggested default.")
    responses: Dict[str, str] = {}
    for field, default in prompts.items():
        raw_input = input(f"{field} [{default}]: ").strip()
        responses[field] = raw_input or default
        logger.debug("Captured input for '%s': '%s'", field, responses[field])
    return responses


def _load_employee_json(path: Path) -> Dict:
    """Load an employee record from a JSON file.

    If the JSON file contains a list of records, only the first one is used.

    Parameters
    ----------
    path:
        The `Path` object pointing to the JSON file.

    Returns
    -------
    A dictionary representing the employee record.

    Raises
    ------
    ValueError:
        If the JSON file contains an empty list.
    """
    logger.info("Loading employee data from %s", path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        if not data:
            raise ValueError(f"Employee JSON file is empty: {path}")
        if len(data) > 1:
            logger.warning("Multiple records found in %s; only the first will be used.", path)
        return data[0]
    return data


def _parse_numeric(value: str) -> float:
    """Safely convert a string to a float, returning NaN on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _normalise_inputs(payload: Dict[str, str]) -> Dict[str, Any]:
    """Convert raw string inputs into types suitable for the model.

    - Numeric fields are converted to floats.
    - Date fields are passed as strings (or None if empty).
    - Other fields are passed through as-is.

    Parameters
    ----------
    payload:
        A dictionary of raw string inputs from the user or JSON file.

    Returns
    -------
    A dictionary with values converted to appropriate types.
    """
    numeric_fields = {
        "Salary",
        "EngagementSurvey",
        "EmpSatisfaction",
        "SpecialProjectsCount",
        "DaysLateLast30",
        "Absences",
    }
    date_fields = {"DateofHire", "DOB", "LastPerformanceReview_Date"}
    normalised: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in numeric_fields:
            normalised[key] = _parse_numeric(value)
        elif key in date_fields:
            # Pass empty strings as None so they are treated as missing dates.
            normalised[key] = value or None
        else:
            normalised[key] = value
    return normalised


def _build_record(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct the employee record from either JSON or user prompts.

    Parameters
    ----------
    args:
        The parsed command-line arguments.

    Returns
    -------
    A dictionary of the normalised employee record.
    """
    if args.employee_json:
        payload = _load_employee_json(args.employee_json)
    else:
        payload = _prompt_user(DEFAULT_PROMPTS)
    return _normalise_inputs(payload)


def _format_probability(probability: float) -> str:
    """Format a float probability as a percentage string."""
    return f"{probability:.2%}"


def display_results(risks: Iterable[TenureRisk]) -> None:
    """Print a formatted table of tenure risk predictions.

    Parameters
    ----------
    risks:
        An iterable of `TenureRisk` objects.
    """
    header = f"{'Tenure':<10}{'Probability':<15}{'Confidence':<15}"
    print("\n--- Predicted Termination Risk ---")
    print(header)
    print("-" * len(header))

    if not risks:
        print("No valid predictions could be generated.")
        return

    for risk in risks:
        tenure_str = f"{risk.tenure_years:>6.1f} yrs"
        prob_str = _format_probability(risk.termination_probability)
        conf_str = _format_probability(risk.confidence)
        print(f"{tenure_str:<10}{prob_str:<15}{conf_str:<15}")


def main() -> None:
    """Main entry point for the CLI application."""
    configure_logging()
    args = parse_args()

    try:
        record = _build_record(args)
        horizons: Sequence[float] = tuple(args.horizons) if args.horizons else (1.0, 2.0, 5.0)

        logger.info("Running inference for horizons: %s", ", ".join(map(str, horizons)))
        risks = predict_tenure_risk(record, horizons=horizons, model_path=args.model)
        display_results(risks)
    except Exception as e:
        logger.exception("An error occurred during prediction.")
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()

"""Command line utility for predicting termination risk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .inference import DEFAULT_MODEL_PATH, predict_tenure_risk
except ImportError:  # pragma: no cover - fallback for script execution
    from inference import DEFAULT_MODEL_PATH, predict_tenure_risk

try:
    from .logging_utils import configure_logging, get_logger
except ImportError:  # pragma: no cover - fallback for script execution
    from logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

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
    parser = argparse.ArgumentParser(description="Predict termination risk for an employee")
    parser.add_argument(
        "--employee-json",
        type=Path,
        help="Path to a JSON file describing the employee (keys must match training columns)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the persisted model pipeline",
    )
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="*",
        default=None,
        help="Optional tenure horizons in years (default: 1, 2, 5)",
    )
    return parser.parse_args()


def _prompt_user(prompts: dict[str, str]) -> dict[str, str]:
    print("Enter employee information. Press Enter to accept the suggested default.")
    responses: dict[str, str] = {}
    for field, default in prompts.items():
        raw = input(f"{field} [{default}]: ").strip()
        responses[field] = raw or default
        logger.debug("Captured input for %s", field)
    return responses


def _load_employee_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        if not data:
            raise ValueError("Employee JSON file is empty")
        if len(data) > 1:
            print("Multiple records found; only the first will be used for prediction.")
            logger.warning("Multiple records provided; truncating to first entry")
        return data[0]
    return data


def _parse_numeric(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _normalise_inputs(payload: dict[str, str]) -> dict[str, object]:
    numeric_fields = {
        "Salary",
        "EngagementSurvey",
        "EmpSatisfaction",
        "SpecialProjectsCount",
        "DaysLateLast30",
        "Absences",
    }
    date_fields = {"DateofHire", "DOB", "LastPerformanceReview_Date"}
    normalised: dict[str, object] = {}
    for key, value in payload.items():
        if key in numeric_fields:
            normalised[key] = _parse_numeric(value)
        elif key in date_fields:
            normalised[key] = value or None
        else:
            normalised[key] = value
    return normalised


def _build_record(args: argparse.Namespace) -> dict[str, object]:
    if args.employee_json:
        payload = _load_employee_json(args.employee_json)
    else:
        payload = _prompt_user(DEFAULT_PROMPTS)
    return _normalise_inputs(payload)


def _format_probability(probability: float) -> str:
    return f"{probability:.2%}"


def display_results(risks: Iterable) -> None:
    header = f"{'Tenure':<10}{'Probability':<15}{'Confidence':<15}"
    print("\nPredicted termination risk")
    print(header)
    print("-" * len(header))
    for risk in risks:
        print(
            f"{risk.tenure_years:>6.1f} yrs    {_format_probability(risk.termination_probability):<15}"
            f"{_format_probability(risk.confidence):<15}"
        )


def main() -> None:
    configure_logging()
    args = parse_args()
    record = _build_record(args)
    horizons: Sequence[float] = (
        tuple(args.horizons) if args.horizons else (1.0, 2.0, 5.0)
    )
    logger.info("Running inference for horizons: %s", ", ".join(map(str, horizons)))
    risks = predict_tenure_risk(record, horizons=horizons, model_path=args.model)
    display_results(risks)


if __name__ == "__main__":
    main()

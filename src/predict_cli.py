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
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml

try:
    from .constants import RANDOM_SEED
    from .inference import DEFAULT_MODEL_PATH, TenureRisk, predict_tenure_risk
    from .schemas import EmployeeRecord, ValidationError
    from .utils.repro import set_global_seed
except ImportError:  # pragma: no cover - fallback for script execution
    from constants import RANDOM_SEED
    from inference import DEFAULT_MODEL_PATH, TenureRisk, predict_tenure_risk
    from schemas import EmployeeRecord, ValidationError
    from utils.repro import set_global_seed

try:
    from .logging_utils import configure_logging, get_logger
except ImportError:  # pragma: no cover - fallback for script execution
    from logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_CONFIG_PATH = PROJECT_ROOT / "configs" / "policy.yaml"

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
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable calibration display and show recommended actions based on policy.yaml",
    )
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=POLICY_CONFIG_PATH,
        help="Path to policy.yaml configuration file",
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


def _load_employee_json(path: Path) -> List[Dict[str, Any]]:
    """Load employee records from a JSON file.

    If the JSON file contains a single record, it is wrapped in a list so the
    caller can treat the result uniformly. When a list of records is provided,
    all entries are preserved.

    Parameters
    ----------
    path:
        The `Path` object pointing to the JSON file.

    Returns
    -------
    A list of dictionaries representing the employee records.

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
        return data
    return [data]


def _build_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Construct one or more employee records from JSON or user prompts.

    Parameters
    ----------
    args:
        The parsed command-line arguments.

    Returns
    -------
    A list of dictionaries representing the normalised employee records.
    """
    if args.employee_json:
        payloads = _load_employee_json(args.employee_json)
    else:
        payloads = [_prompt_user(DEFAULT_PROMPTS)]

    validated_records: List[Dict[str, Any]] = []
    for payload in payloads:
        record = EmployeeRecord.model_validate(payload)
        validated_records.append(record.normalized_payload())
    return validated_records


def _format_probability(probability: float) -> str:
    """Format a float probability as a percentage string."""
    return f"{probability:.2%}"


def load_policy_config(config_path: Path) -> Dict[str, Any]:
    """Load policy configuration from YAML file.

    Parameters
    ----------
    config_path : Path
        Path to policy.yaml

    Returns
    -------
    Dict[str, Any]
        Policy configuration dictionary
    """
    if not config_path.exists():
        logger.warning(f"Policy config not found at {config_path}, using defaults")
        return {
            "risk_thresholds": {"low": 0.10, "low_moderate": 0.30, "moderate": 0.60, "high": 1.00},
            "action_mappings": {},
        }

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_risk_band_and_actions(probability: float, policy_config: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Determine risk band and recommended actions for a given probability.

    Parameters
    ----------
    probability : float
        Predicted termination probability
    policy_config : Dict[str, Any]
        Policy configuration

    Returns
    -------
    Tuple[str, List[str]]
        (risk_band, recommended_actions)
    """
    thresholds = policy_config.get("risk_thresholds", {})
    mappings = policy_config.get("action_mappings", {})

    if probability < thresholds.get("low", 0.10):
        band = "low"
    elif probability < thresholds.get("low_moderate", 0.30):
        band = "low_moderate"
    elif probability < thresholds.get("moderate", 0.60):
        band = "moderate"
    else:
        band = "high"

    band_info = mappings.get(band, {})
    label = band_info.get("label", band.replace("_", " ").title())
    actions = band_info.get("actions", [])

    return label, actions


def display_results(
    risks: Iterable[TenureRisk], show_actions: bool = False, policy_config: Dict[str, Any] = None
) -> None:
    """Print a formatted table of tenure risk predictions.

    Parameters
    ----------
    risks:
        An iterable of `TenureRisk` objects.
    show_actions : bool
        Whether to display recommended actions
    policy_config : Dict[str, Any]
        Policy configuration for action recommendations
    """
    header = f"{'Tenure':<10}{'Probability':<15}{'Confidence':<15}"
    if show_actions:
        header += f"{'Risk Band':<20}"
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
        line = f"{tenure_str:<10}{prob_str:<15}{conf_str:<15}"

        if show_actions and policy_config:
            band_label, actions = get_risk_band_and_actions(risk.termination_probability, policy_config)
            line += f"{band_label:<20}"

        print(line)

    # Display recommended actions for the first risk (typically current tenure)
    if show_actions and policy_config and risks:
        first_risk = list(risks)[0]
        band_label, actions = get_risk_band_and_actions(first_risk.termination_probability, policy_config)

        print(f"\n--- Recommended Actions ({band_label}) ---")
        if actions:
            for i, action in enumerate(actions, 1):
                print(f"{i}. {action}")
        else:
            print("No specific actions configured for this risk band.")


def _display_validation_errors(error: ValidationError) -> None:
    print("\nInput validation failed:")
    for issue in error.errors():
        loc = " -> ".join(str(part) for part in issue.get("loc", [])) or "record"
        print(f" - {loc}: {issue.get('msg')}")


def main() -> None:
    """Main entry point for the CLI application."""
    configure_logging()
    set_global_seed(RANDOM_SEED)
    args = parse_args()

    try:
        # Load policy config if calibration is requested
        policy_config = None
        if args.calibrate:
            policy_config = load_policy_config(args.policy_config)
            logger.info(f"Loaded policy configuration from {args.policy_config}")

        records = _build_records(args)
        horizons: Sequence[float] = tuple(args.horizons) if args.horizons else (1.0, 2.0, 5.0)

        logger.info("Running inference for horizons: %s", ", ".join(map(str, horizons)))
        for index, record in enumerate(records, start=1):
            logger.info("Processing record %d", index)
            print(f"\n=== Employee record {index} ===")
            risks = predict_tenure_risk(record, horizons=horizons, model_path=args.model)
            display_results(risks, show_actions=args.calibrate, policy_config=policy_config)
    except ValidationError as err:
        _display_validation_errors(err)
        sys.exit(1)
    except Exception as e:
        logger.exception("An error occurred during prediction.")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

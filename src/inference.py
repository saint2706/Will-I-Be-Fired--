"""Inference helpers for the employee termination model.

This module provides a collection of functions to facilitate predictions using
the trained model pipeline. It supports:
- Loading the persisted model from disk.
- Preparing raw input records for inference, ensuring they match the feature
  schema expected by the model.
- Generating both probabilistic and binary predictions for termination.
- A specialized function (`predict_tenure_risk`) to estimate termination
  risk at various future tenure horizons (e.g., at 1, 2, and 5 years post-hire).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from .feature_engineering import parse_dates, prepare_inference_frame
    from .logging_utils import get_logger
except ImportError:  # pragma: no cover - fallback for script execution
    from feature_engineering import parse_dates, prepare_inference_frame
    from logging_utils import get_logger

# Define project structure and default paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_TENURE_HORIZONS: Sequence[float] = (1.0, 2.0, 5.0)

logger = get_logger(__name__)

# Type alias for records that can be processed.
Records = Union[pd.DataFrame, dict, pd.Series]


@dataclass
class TenureRisk:
    """Represents predicted risk for a specific tenure horizon.

    Attributes
    ----------
    tenure_years:
        The tenure horizon (in years) for which the prediction was made.
    termination_probability:
        The model's predicted probability of termination at this tenure.
    confidence:
        The model's confidence in its prediction, derived from the probability.
        A value of 0.5 means maximum uncertainty, while 1.0 means certainty.
    """

    tenure_years: float
    termination_probability: float
    confidence: float


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load the persisted preprocessing and estimator pipeline.

    Parameters
    ----------
    model_path:
        The path to the `.joblib` file containing the scikit-learn pipeline.

    Returns
    -------
    The loaded scikit-learn pipeline object.
    """
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)
    logger.debug("Model loaded successfully: %s", type(model))
    return model


def _ensure_model(model: Optional[Pipeline], model_path: Path) -> Pipeline:
    """Helper to load the model only if it hasn't been loaded already.

    Parameters
    ----------
    model:
        An optional, already-loaded model object.
    model_path:
        Path to load the model from if `model` is None.

    Returns
    -------
    A loaded model pipeline.
    """
    if model is None:
        return load_model(model_path)
    return model


def prepare_features_for_inference(
    records: Records, *, reference_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Return a DataFrame formatted for the trained pipeline.

    This is a thin wrapper around `prepare_inference_frame` from the feature
    engineering module.

    Parameters
    ----------
    records:
        The raw input data (dict, Series, or DataFrame).
    reference_date:
        An optional reference date for temporal calculations.

    Returns
    -------
    A DataFrame ready for the model's `predict` or `predict_proba` methods.
    """
    logger.debug("Preparing features for inference")
    features = prepare_inference_frame(records, reference_date=reference_date)
    logger.debug("Prepared feature columns: %s", ", ".join(features.columns))
    return features


def predict_termination_probability(
    records: Records,
    *,
    model: Optional[Pipeline] = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    reference_date: Optional[pd.Timestamp] = None,
) -> Iterable[float]:
    """Return termination probabilities for the supplied employee records.

    Parameters
    ----------
    records:
        The employee data to predict on. Can be a single record (dict, Series)
        or multiple records (DataFrame).
    model:
        An optional pre-loaded model pipeline. If not provided, it will be
        loaded from `model_path`.
    model_path:
        The path to the model file if `model` is not provided.
    reference_date:
        An optional reference date for temporal feature engineering.

    Returns
    -------
    An iterable of float values representing the termination probabilities.
    """
    estimator = _ensure_model(model, model_path)
    features = prepare_features_for_inference(records, reference_date=reference_date)
    probabilities = estimator.predict_proba(features)[:, 1]
    logger.debug("Predicted probabilities for %d records", len(probabilities))
    return probabilities


def predict_termination(
    records: Records,
    threshold: float = 0.5,
    *,
    model: Optional[Pipeline] = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    reference_date: Optional[pd.Timestamp] = None,
) -> Iterable[int]:
    """Return binary termination predictions for the supplied employee records.

    Parameters
    ----------
    records:
        The employee data to predict on.
    threshold:
        The probability threshold to classify a prediction as "terminated" (1).
    model:
        An optional pre-loaded model pipeline.
    model_path:
        The path to the model file if `model` is not provided.
    reference_date:
        An optional reference date for temporal features.

    Returns
    -------
    An iterable of binary integers (0 or 1) for the predictions.
    """
    probabilities = predict_termination_probability(
        records, model=model, model_path=model_path, reference_date=reference_date
    )
    predictions = (probabilities >= threshold).astype(int)
    logger.debug("Generated binary predictions using threshold %.2f", threshold)
    return predictions


def confidence_from_probability(probability: float) -> float:
    """Convert a prediction probability into a confidence score.

    The confidence is highest (1.0) when the probability is 0.0 or 1.0.
    It is lowest (0.5) when the probability is 0.5 (maximum uncertainty).

    Parameters
    ----------
    probability:
        The raw prediction probability, between 0.0 and 1.0.

    Returns
    -------
    The corresponding confidence score.
    """
    return max(probability, 1 - probability)


def _reference_date_for_horizon(record_frame: pd.DataFrame, horizon_years: float) -> Optional[pd.Timestamp]:
    """Derive a future reference date for a requested tenure horizon.

    Calculates the target date by adding `horizon_years` to the hire date.

    Parameters
    ----------
    record_frame:
        A DataFrame containing the parsed employee record.
    horizon_years:
        The number of years of tenure to project forward to.

    Returns
    -------
    The calculated `pd.Timestamp` or `None` if hire date is missing.
    """
    if "DateofHire" in record_frame:
        hire_date = record_frame["DateofHire"].iloc[0]
        if pd.notna(hire_date):
            # Convert the fractional-year horizon into a timedelta so pandas can
            # handle non-integer values (e.g., 1.5 years).
            horizon_in_days = horizon_years * 365.25
            horizon_delta = pd.to_timedelta(horizon_in_days, unit="D")
            return hire_date + horizon_delta
    logger.warning("Could not calculate horizon reference date due to missing 'DateofHire'")
    return None


def predict_tenure_risk(
    record: Records,
    horizons: Sequence[float] = DEFAULT_TENURE_HORIZONS,
    *,
    model: Optional[Pipeline] = None,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> list[TenureRisk]:
    """Predict termination risk at multiple future tenure horizons.

    For each specified tenure horizon (e.g., 1 year, 5 years), this function
    simulates the state of the employee at that future point in time and
    predicts their termination risk. It does this by adjusting the reference
    date used for temporal feature engineering.

    Parameters
    ----------
    record:
        A single employee record (dict, Series, or single-row DataFrame).
    horizons:
        A sequence of tenure values (in years) at which to estimate risk.
    model:
        An optional pre-loaded estimator pipeline.
    model_path:
        Path to the model if not already loaded.

    Returns
    -------
    A list of `TenureRisk` objects, one for each requested horizon.
    """
    # Standardize input and ensure 'DateofTermination' is not present.
    raw_frame = record.iloc[[0]].copy() if isinstance(record, pd.DataFrame) else pd.DataFrame([record])
    base_frame = parse_dates(raw_frame)
    if "DateofTermination" in base_frame.columns:
        base_frame["DateofTermination"] = pd.NaT

    risks: list[TenureRisk] = []
    estimator = _ensure_model(model, model_path)

    # Iterate through each horizon, calculate features, and predict.
    for horizon in horizons:
        # Determine the reference date for this specific "what-if" scenario.
        ref_date = _reference_date_for_horizon(base_frame, horizon)
        if ref_date is None:
            continue  # Skip if hire date is missing

        features = prepare_features_for_inference(base_frame, reference_date=ref_date)
        probability = estimator.predict_proba(features)[:, 1][0]
        logger.debug(
            "Predicted risk %.4f at %.1f years tenure (ref_date: %s)",
            probability,
            horizon,
            ref_date.date(),
        )

        risks.append(
            TenureRisk(
                tenure_years=float(horizon),
                termination_probability=float(probability),
                confidence=float(confidence_from_probability(probability)),
            )
        )

    return risks

"""Inference helpers for the employee termination model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import pandas as pd

try:
    from .logging_utils import get_logger
except ImportError:  # pragma: no cover - fallback for script execution
    from logging_utils import get_logger

try:
    from .feature_engineering import prepare_inference_frame, parse_dates
except ImportError:  # pragma: no cover - fallback for script execution
    from feature_engineering import prepare_inference_frame, parse_dates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_TENURE_HORIZONS: Sequence[float] = (1.0, 2.0, 5.0)

logger = get_logger(__name__)


@dataclass
class TenureRisk:
    """Represents predicted risk for a specific tenure horizon."""

    tenure_years: float
    termination_probability: float
    confidence: float


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    """Load the persisted preprocessing + estimator pipeline."""

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)
    logger.debug("Model loaded: %s", type(model))
    return model


def _ensure_model(model, model_path: Path):
    if model is None:
        model = load_model(model_path)
    return model


def prepare_features_for_inference(records: pd.DataFrame | dict | pd.Series, *, reference_date=None) -> pd.DataFrame:
    """Return a dataframe formatted for the trained pipeline."""

    logger.debug("Preparing features for inference")
    features = prepare_inference_frame(records, reference_date=reference_date)
    logger.debug("Prepared feature columns: %s", ", ".join(features.columns))
    return features


def predict_termination_probability(
    records: pd.DataFrame | dict | pd.Series,
    *,
    model=None,
    model_path: Path = DEFAULT_MODEL_PATH,
    reference_date=None,
) -> Iterable[float]:
    """Return termination probabilities for the supplied employee records."""

    model = _ensure_model(model, model_path)
    features = prepare_features_for_inference(records, reference_date=reference_date)
    probabilities = model.predict_proba(features)[:, 1]
    logger.debug("Predicted probabilities for %d records", len(probabilities))
    return probabilities


def predict_termination(
    records: pd.DataFrame | dict | pd.Series,
    threshold: float = 0.5,
    *,
    model=None,
    model_path: Path = DEFAULT_MODEL_PATH,
    reference_date=None,
) -> Iterable[int]:
    """Return binary termination predictions for the supplied employee records."""

    probabilities = predict_termination_probability(
        records,
        model=model,
        model_path=model_path,
        reference_date=reference_date,
    )
    predictions = (probabilities >= threshold).astype(int)
    logger.debug("Generated binary predictions using threshold %.2f", threshold)
    return predictions


def confidence_from_probability(probability: float) -> float:
    """Convert a probability into a confidence score between 0 and 1."""

    return max(probability, 1 - probability)


def _reference_date_for_horizon(record: pd.DataFrame, horizon_years: float) -> pd.Timestamp | None:
    """Derive a reference date for the requested tenure horizon."""

    parsed = parse_dates(record)
    if "DateofHire" in parsed:
        hire = parsed["DateofHire"].iloc[0]
        if pd.notna(hire):
            return hire + pd.DateOffset(years=horizon_years)
    return None


def predict_tenure_risk(
    record: pd.DataFrame | dict | pd.Series,
    horizons: Sequence[float] = DEFAULT_TENURE_HORIZONS,
    *,
    model=None,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> list[TenureRisk]:
    """Predict termination risk at multiple tenure horizons.

    Parameters
    ----------
    record:
        Single employee record. The function accepts dictionaries, pandas
        Series, or single-row DataFrames using the same column names as the
        training data. Missing fields are imputed with neutral defaults.
    horizons:
        Tenure values (in years) at which risk will be estimated.
    model / model_path:
        Optionally supply an already loaded estimator; otherwise the function
        loads ``models/best_model.joblib``.
    """

    raw_frame = record.iloc[[0]].copy() if isinstance(record, pd.DataFrame) else pd.DataFrame([record])
    base_frame = parse_dates(raw_frame)
    if "DateofTermination" in base_frame.columns:
        base_frame["DateofTermination"] = pd.NaT

    risks: list[TenureRisk] = []
    estimator = _ensure_model(model, model_path)

    for horizon in horizons:
        ref_date = _reference_date_for_horizon(base_frame, horizon)
        features = prepare_features_for_inference(base_frame, reference_date=ref_date)
        probability = estimator.predict_proba(features)[:, 1][0]
        logger.debug(
            "Predicted risk %.4f at %.1f years using reference date %s",
            probability,
            horizon,
            ref_date,
        )
        risks.append(
            TenureRisk(
                tenure_years=float(horizon),
                termination_probability=float(probability),
                confidence=float(confidence_from_probability(probability)),
            )
        )

    return risks

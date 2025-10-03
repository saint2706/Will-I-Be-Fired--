"""Reusable feature engineering utilities for termination modeling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd

try:
    from .logging_utils import get_logger
except ImportError:  # pragma: no cover - fallback for script execution
    from logging_utils import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "Termd"

DATE_COLUMNS: Sequence[str] = (
    "DateofHire",
    "DateofTermination",
    "DOB",
    "LastPerformanceReview_Date",
)

NUMERIC_FEATURES: Sequence[str] = (
    "Salary",
    "EngagementSurvey",
    "EmpSatisfaction",
    "SpecialProjectsCount",
    "DaysLateLast30",
    "Absences",
    "tenure_years",
    "age_years",
    "years_since_last_review",
)

CATEGORICAL_FEATURES: Sequence[str] = (
    "Department",
    "PerformanceScore",
    "RecruitmentSource",
    "Position",
    "State",
    "Sex",
    "MaritalDesc",
    "CitizenDesc",
    "RaceDesc",
    "HispanicLatino",
)

DROP_COLUMNS: Sequence[str] = (
    "Employee_Name",
    "EmpID",
    "MarriedID",
    "MaritalStatusID",
    "GenderID",
    "EmpStatusID",
    "DeptID",
    "PerfScoreID",
    "FromDiversityJobFairID",
    "ManagerName",
    "ManagerID",
    "EmploymentStatus",
    "TermReason",
    "DateofTermination",
    "LastPerformanceReview_Date",
    "DOB",
    "DateofHire",
)


@dataclass
class FeatureSpec:
    """Defines the schema expected by the downstream model."""

    numeric: Sequence[str]
    categorical: Sequence[str]


FEATURE_SPEC = FeatureSpec(numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with known date columns parsed."""

    result = df.copy()
    for column in DATE_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce", format="mixed")
            logger.debug("Parsed date column %s", column)
    return result


def _resolve_reference_date(df: pd.DataFrame, reference_date: Optional[pd.Timestamp]) -> pd.Timestamp:
    """Return the date used for tenure/age calculations."""

    if reference_date is not None:
        return reference_date

    for column in ("LastPerformanceReview_Date", "DateofTermination", "DateofHire"):
        if column in df.columns:
            candidate = pd.to_datetime(df[column], errors="coerce").max()
            if pd.notna(candidate):
                return candidate
    return pd.Timestamp.today().normalize()


def engineer_temporal_features(
    df: pd.DataFrame, *, reference_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Derive tenure, age, and recency features.

    Parameters
    ----------
    df:
        Raw employee records.
    reference_date:
        Optional date representing "today". When omitted the function infers
        a sensible default using the latest performance review, termination
        date, or hire date found in *df*.
    """

    frame = parse_dates(df)
    ref_date = _resolve_reference_date(frame, reference_date)
    logger.debug("Using reference date %s for temporal features", ref_date)

    hire_date = frame.get("DateofHire")
    termination_date = frame.get("DateofTermination")
    dob = frame.get("DOB")
    last_review = frame.get("LastPerformanceReview_Date")

    if hire_date is not None:
        tenure_end = termination_date.fillna(ref_date) if termination_date is not None else ref_date
        tenure_days = (tenure_end - hire_date).dt.days.clip(lower=0)
        frame["tenure_years"] = tenure_days / 365.25
        logger.debug("Computed tenure years for %d records", len(frame))
    else:
        frame["tenure_years"] = 0.0

    if dob is not None:
        frame["age_years"] = ((ref_date - dob).dt.days.clip(lower=0)) / 365.25
        logger.debug("Computed age for %d records", len(frame))
    else:
        frame["age_years"] = 0.0

    if last_review is not None:
        review_delta = (ref_date - last_review).dt.days / 365.25
        frame["years_since_last_review"] = review_delta.fillna(review_delta.median())
        logger.debug("Computed years since last review for %d records", len(frame))
    else:
        frame["years_since_last_review"] = 0.0

    return frame


def drop_unused_columns(df: pd.DataFrame, *, drop_target: bool = True) -> pd.DataFrame:
    """Remove identifier and leakage-prone columns."""

    columns_to_drop: Iterable[str] = [column for column in DROP_COLUMNS if column in df.columns]
    result = df.drop(columns=list(columns_to_drop), errors="ignore")
    if columns_to_drop:
        logger.debug("Dropped columns: %s", ", ".join(columns_to_drop))
    if drop_target and TARGET_COLUMN in result.columns:
        result = result.drop(columns=[TARGET_COLUMN])
    return result


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features, target) ready for model training."""

    engineered = engineer_temporal_features(df)
    features = drop_unused_columns(engineered, drop_target=True)
    target = engineered[TARGET_COLUMN].astype(int)
    return features, target


def prepare_inference_frame(
    record: pd.DataFrame | dict | pd.Series,
    *,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Prepare a single record for inference.

    The function accepts a mapping, series, or single-row dataframe and
    returns a dataframe containing the engineered features expected by the
    trained model.
    """

    if isinstance(record, dict):
        frame = pd.DataFrame([record])
    elif isinstance(record, pd.Series):
        frame = record.to_frame().T
    else:
        frame = record.copy()

    engineered = engineer_temporal_features(frame, reference_date=reference_date)
    features = drop_unused_columns(engineered, drop_target=True)

    for column in FEATURE_SPEC.numeric:
        if column not in features:
            features[column] = 0.0
    for column in FEATURE_SPEC.categorical:
        if column not in features:
            features[column] = "Unknown"

    return features[sorted(features.columns)]

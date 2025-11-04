"""Reusable feature engineering utilities for termination modeling.

This module provides a suite of functions for transforming raw employee data
from the HRDataset_v14.csv file into a feature matrix suitable for machine
learning. Key responsibilities include:

- **Date parsing:** Robustly converting date strings into datetime objects.
- **Temporal feature engineering:** Deriving features like employee tenure,
  age, and time since the last performance review.
- **Column cleanup:** Removing identifiers, redundant fields, and columns
  that would leak information from the future (e.g., termination reason).
- **Data preparation:** Offering dedicated functions to prepare data for
  either model training or inference, ensuring consistency between phases.

The feature lists (numeric, categorical, date columns) are centralized here
to simplify maintenance and ensure that all downstream components (training,
inference, GUI) use a consistent feature set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd

try:
    from .logging_utils import get_logger
except ImportError:  # pragma: no cover - fallback for script execution
    from logging_utils import get_logger

logger = get_logger(__name__)

# The target variable for the classification model.
TARGET_COLUMN = "Termd"

# Columns containing date information that require parsing.
DATE_COLUMNS: Sequence[str] = (
    "DateofHire",
    "DateofTermination",
    "DOB",
    "LastPerformanceReview_Date",
)

# Raw numeric inputs that must be provided by the user or source dataset.
RAW_NUMERIC_INPUTS: Sequence[str] = (
    "Salary",
    "EngagementSurvey",
    "EmpSatisfaction",
    "SpecialProjectsCount",
    "DaysLateLast30",
    "Absences",
)

# Numeric features to be used in the model.
# These will typically be scaled and imputed.
NUMERIC_FEATURES: Sequence[str] = RAW_NUMERIC_INPUTS + (
    "tenure_years",  # Engineered feature
    "age_years",  # Engineered feature
    "years_since_last_review",  # Engineered feature
)

# Categorical features to be used in the model.
# These will be one-hot encoded after imputation.
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

# Columns to be dropped from the dataset before modeling.
# This list includes identifiers, redundant fields, and data leakage sources.
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
    # Date columns are dropped after feature engineering.
    "DateofTermination",
    "LastPerformanceReview_Date",
    "DOB",
    "DateofHire",
)


@dataclass
class FeatureSpec:
    """Defines the schema expected by the downstream model.

    This dataclass provides a structured way to access the lists of numeric
    and categorical features that the model pipeline is trained on.

    Attributes
    ----------
    numeric:
        A sequence of strings naming the numeric features.
    categorical:
        A sequence of strings naming the categorical features.
    """

    numeric: Sequence[str]
    categorical: Sequence[str]


# A global instance of FeatureSpec for easy access to feature lists.
FEATURE_SPEC = FeatureSpec(numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with known date columns parsed.

    Iterates through `DATE_COLUMNS` and converts each one to a pandas
    datetime object. It uses `errors='coerce'` to turn unparseable dates
    into `NaT` (Not a Time).

    Parameters
    ----------
    df:
        The input DataFrame with potentially unparsed date strings.

    Returns
    -------
    A new DataFrame with the date columns converted to datetime objects.
    """
    result = df.copy()
    for column in DATE_COLUMNS:
        if column in result.columns:
            # `format="mixed"` allows pandas to handle multiple date formats.
            result[column] = pd.to_datetime(result[column], errors="coerce", format="mixed")
            logger.debug("Parsed date column %s", column)
    return result


def _resolve_reference_date(df: pd.DataFrame, reference_date: Optional[pd.Timestamp]) -> pd.Timestamp:
    """Return the date used for tenure/age calculations.

    This helper determines the "current" date for calculating time-based
    features. The priority is:
    1. An explicitly provided `reference_date`.
    2. The latest non-null date from key event columns in the dataset.
    3. Today's date as a final fallback.

    Parameters
    ----------
    df:
        The DataFrame from which to infer a date if needed.
    reference_date:
        An optional, explicit reference date.

    Returns
    -------
    The resolved `pd.Timestamp` to use as the reference point.
    """
    if reference_date is not None:
        return reference_date

    # Infer the reference date from the latest event in the data.
    for column in ("LastPerformanceReview_Date", "DateofTermination", "DateofHire"):
        if column in df.columns:
            candidate = pd.to_datetime(df[column], errors="coerce").max()
            if pd.notna(candidate):
                logger.debug("Inferred reference date %s from column %s", candidate, column)
                return candidate

    # Fallback to the current date if no other reference is available.
    today = pd.Timestamp.today().normalize()
    logger.debug("Using today's date %s as the reference", today)
    return today


def engineer_temporal_features(df: pd.DataFrame, *, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Derive tenure, age, and recency features from date columns.

    This function calculates key temporal features:
    - `tenure_years`: Time from hire date to termination or reference date.
    - `age_years`: Employee's age at the reference date.
    - `years_since_last_review`: Time since the last performance review.

    Parameters
    ----------
    df:
        A DataFrame containing raw employee records with date columns.
    reference_date:
        Optional date representing "today". If omitted, the function infers
        a sensible default using `_resolve_reference_date`.

    Returns
    -------
    A new DataFrame with the added temporal feature columns.
    """
    frame = parse_dates(df)
    ref_date = _resolve_reference_date(frame, reference_date)
    logger.debug("Using reference date %s for temporal features", ref_date)

    # Safely get date columns, which may not always be present.
    hire_date = frame.get("DateofHire")
    termination_date = frame.get("DateofTermination")
    dob = frame.get("DOB")
    last_review = frame.get("LastPerformanceReview_Date")

    # Calculate tenure in years.
    if hire_date is not None:
        # Use termination date if available, otherwise use the reference date.
        tenure_end = termination_date.fillna(ref_date) if termination_date is not None else ref_date
        tenure_days = (tenure_end - hire_date).dt.days.clip(lower=0)
        frame["tenure_years"] = tenure_days / 365.25
        logger.debug("Computed tenure years for %d records", len(frame))
    else:
        frame["tenure_years"] = 0.0

    # Calculate age in years.
    if dob is not None:
        age_days = (ref_date - dob).dt.days.clip(lower=0)
        frame["age_years"] = age_days / 365.25
        logger.debug("Computed age for %d records", len(frame))
    else:
        frame["age_years"] = 0.0

    # Calculate years since the last performance review.
    if last_review is not None:
        review_delta = (ref_date - last_review).dt.days / 365.25
        # Impute missing review dates with the median for the column.
        frame["years_since_last_review"] = review_delta.fillna(review_delta.median())
        logger.debug("Computed years since last review for %d records", len(frame))
    else:
        frame["years_since_last_review"] = 0.0

    return frame


def drop_unused_columns(df: pd.DataFrame, *, drop_target: bool = True) -> pd.DataFrame:
    """Remove identifier, leakage-prone, and redundant columns.

    This function cleans the DataFrame by dropping columns listed in the
    global `DROP_COLUMNS` list. This is crucial for preventing data leakage
    and removing features that are not useful for modeling.

    Parameters
    ----------
    df:
        The DataFrame to clean.
    drop_target:
        If `True`, the target column (`Termd`) will also be dropped. This is
        the default for preparing a feature matrix `X`.

    Returns
    -------
    A DataFrame with the specified columns removed.
    """
    # Identify which of the columns to drop are actually in the DataFrame.
    columns_to_drop: Iterable[str] = [column for column in DROP_COLUMNS if column in df.columns]
    result = df.drop(columns=list(columns_to_drop), errors="ignore")
    if columns_to_drop:
        logger.debug("Dropped columns: %s", ", ".join(columns_to_drop))

    # Optionally drop the target column.
    if drop_target and TARGET_COLUMN in result.columns:
        result = result.drop(columns=[TARGET_COLUMN])
        logger.debug("Dropped target column: %s", TARGET_COLUMN)
    return result


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features, target) ready for model training.

    This function orchestrates the full feature engineering pipeline for a
    training dataset. It generates temporal features, drops unused columns,
    and separates the feature matrix `X` from the target vector `y`.

    Parameters
    ----------
    df:
        The raw DataFrame loaded from the source CSV.

    Returns
    -------
    A tuple containing:
    - A DataFrame `X` with the final features for training.
    - A Series `y` with the integer-coded target variable.
    """
    engineered = engineer_temporal_features(df)
    features = drop_unused_columns(engineered, drop_target=True)
    target = engineered[TARGET_COLUMN].astype(int)
    logger.info("Prepared training data with %d features and %d target values", len(features), len(target))
    return features, target


def prepare_inference_frame(
    record: pd.DataFrame | dict | pd.Series,
    *,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Prepare a single record or a small batch for inference.

    This function handles the transformation of raw input (like a dictionary
    from a web form or a row from a file) into a DataFrame that matches the
    schema expected by the trained model pipeline.

    Parameters
    ----------
    record:
        A single record, which can be a dictionary, a pandas Series, or a
        single-row DataFrame.
    reference_date:
        An optional reference date for temporal feature calculation. This is
        particularly useful for "what-if" scenarios (e.g., estimating risk
        at a future date).

    Returns
    -------
    A DataFrame formatted with the correct columns and dtypes for the model.
    """
    # Standardize the input record into a DataFrame.
    if isinstance(record, dict):
        frame = pd.DataFrame([record])
    elif isinstance(record, pd.Series):
        frame = record.to_frame().T
    else:
        frame = record.copy()

    # Apply the same feature engineering and cleanup steps as in training.
    engineered = engineer_temporal_features(frame, reference_date=reference_date)
    features = drop_unused_columns(engineered, drop_target=True)

    # Ensure all expected feature columns are present, filling with defaults.
    # This prevents errors if the input record is missing some fields.
    for column in FEATURE_SPEC.numeric:
        if column not in features:
            features[column] = 0.0
    for column in FEATURE_SPEC.categorical:
        if column not in features:
            features[column] = "Unknown"

    # Return columns in a consistent, sorted order.
    return features[sorted(features.columns)]

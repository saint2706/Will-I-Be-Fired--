from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import feature_engineering as fe  # noqa: E402


def test_raw_numeric_inputs_exclude_engineered_features():
    engineered = {"tenure_years", "age_years", "years_since_last_review"}

    assert not engineered.intersection(fe.RAW_NUMERIC_INPUTS)
    for column in fe.RAW_NUMERIC_INPUTS:
        assert column in fe.NUMERIC_FEATURES


def test_prepare_inference_frame_recomputes_temporal_features():
    record = {
        "Salary": 75000,
        "EngagementSurvey": 4.0,
        "EmpSatisfaction": 3.0,
        "SpecialProjectsCount": 2,
        "DaysLateLast30": 1,
        "Absences": 0,
        "tenure_years": 25.0,  # Should be ignored in favour of date-derived value
        "age_years": 80.0,  # Should be ignored in favour of date-derived value
        "DateofHire": "2020-01-01",
        "DateofTermination": "2021-01-01",
        "DOB": "1990-01-01",
        "LastPerformanceReview_Date": "2020-06-01",
    }

    prepared = fe.prepare_inference_frame(record)

    tenure_years = prepared.loc[0, "tenure_years"]
    age_years = prepared.loc[0, "age_years"]

    expected_tenure = (
        pd.Timestamp("2021-01-01") - pd.Timestamp("2020-01-01")
    ).days / 365.25
    # Reference date resolves from the first available event column (last review here).
    resolved_reference = pd.Timestamp("2020-06-01")
    expected_age = (
        resolved_reference - pd.Timestamp("1990-01-01")
    ).days / 365.25

    assert abs(tenure_years - expected_tenure) < 1e-6
    assert abs(age_years - expected_age) < 1e-6

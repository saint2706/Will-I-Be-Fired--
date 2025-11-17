import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import inference  # noqa: E402


def _valid_record(**overrides):
    payload = {
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
        "Salary": 60000,
        "EngagementSurvey": 4.0,
        "EmpSatisfaction": 4,
        "SpecialProjectsCount": 3,
        "DaysLateLast30": 0,
        "Absences": 5,
        "DateofHire": pd.Timestamp("2020-01-01"),
        "DOB": pd.Timestamp("1990-01-01"),
        "LastPerformanceReview_Date": pd.Timestamp("2023-01-01"),
    }
    payload.update(overrides)
    return payload


class DummyEstimator:
    def predict_proba(self, X):
        # Return deterministic probabilities for testing.
        return np.array([[0.2, 0.8]])


def test_predict_tenure_risk_accepts_fractional_horizon(monkeypatch):
    record = _valid_record(DateofHire=pd.Timestamp("2020-01-01"))

    def fake_prepare_features(records, *, reference_date=None):
        # Ensure the fractional horizon was translated to a timestamp.
        assert isinstance(reference_date, pd.Timestamp)
        return pd.DataFrame({"dummy": [1]})

    monkeypatch.setattr(inference, "prepare_features_for_inference", fake_prepare_features)

    risks = inference.predict_tenure_risk(record, horizons=[1.5], model=DummyEstimator())

    assert len(risks) == 1
    assert isinstance(risks[0], inference.TenureRisk)
    assert risks[0].tenure_years == 1.5

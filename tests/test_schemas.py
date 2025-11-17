import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from schemas import EmployeeRecord, ValidationError  # noqa: E402


def _valid_payload(**overrides):
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
        "Salary": 65000,
        "EngagementSurvey": 4.0,
        "EmpSatisfaction": 4,
        "SpecialProjectsCount": 3,
        "DaysLateLast30": 0,
        "Absences": 5,
        "DateofHire": date(2020, 1, 1),
        "DOB": date(1990, 5, 18),
        "LastPerformanceReview_Date": date(2023, 10, 1),
    }
    payload.update(overrides)
    return payload


def test_employee_record_accepts_valid_payload():
    record = EmployeeRecord.model_validate(_valid_payload())
    assert record.Salary == 65000
    normalized = record.normalized_payload()
    assert normalized["DateofHire"] == "2020-01-01"


def test_employee_record_requires_all_fields():
    payload = _valid_payload()
    payload.pop("Department")
    with pytest.raises(ValidationError):
        EmployeeRecord.model_validate(payload)


def test_employee_record_rejects_bad_numeric_type():
    payload = _valid_payload(Salary="not-a-number")
    with pytest.raises(ValidationError):
        EmployeeRecord.model_validate(payload)


def test_employee_record_rejects_bad_dates():
    payload = _valid_payload(DOB=date(2025, 1, 1))
    with pytest.raises(ValidationError):
        EmployeeRecord.model_validate(payload)

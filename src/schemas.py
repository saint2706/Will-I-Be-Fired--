"""Pydantic models describing user-facing input payloads."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DATE_BOUNDS = (date(1900, 1, 1), date(2100, 12, 31))


class EmployeeRecord(BaseModel):
    """Schema describing the raw employee data collected from HR partners."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, populate_by_name=True)

    Department: str = Field(..., min_length=1, max_length=64)
    PerformanceScore: str = Field(..., min_length=1, max_length=64)
    RecruitmentSource: str = Field(..., min_length=1, max_length=64)
    Position: str = Field(..., min_length=1, max_length=64)
    State: str = Field(..., min_length=1, max_length=64)
    Sex: str = Field(..., min_length=1, max_length=32)
    MaritalDesc: str = Field(..., min_length=1, max_length=32)
    CitizenDesc: str = Field(..., min_length=1, max_length=64)
    RaceDesc: str = Field(..., min_length=1, max_length=64)
    HispanicLatino: str = Field(..., min_length=1, max_length=16)

    Salary: float = Field(65000.0, ge=0, le=1_000_000, description="Annual salary in USD")
    EngagementSurvey: float = Field(4.0, ge=0.0, le=5.0)
    EmpSatisfaction: int = Field(4, ge=1, le=5)
    SpecialProjectsCount: int = Field(3, ge=0, le=50)
    DaysLateLast30: int = Field(0, ge=0, le=30)
    Absences: int = Field(5, ge=0, le=365)

    DateofHire: date = Field(...)
    DOB: date = Field(...)
    LastPerformanceReview_Date: date = Field(...)

    @field_validator("DateofHire", "DOB", "LastPerformanceReview_Date")
    @classmethod
    def _validate_date_range(cls, value: date) -> date:
        lower, upper = DATE_BOUNDS
        if value < lower or value > upper:
            raise ValueError(f"Date must be between {lower} and {upper}")
        return value

    @field_validator("EngagementSurvey")
    @classmethod
    def _round_engagement(cls, value: float) -> float:
        return round(float(value), 2)

    @model_validator(mode="after")
    def _check_temporal_consistency(self) -> "EmployeeRecord":
        if self.DOB >= self.DateofHire:
            raise ValueError("DOB must be earlier than DateofHire")
        if self.LastPerformanceReview_Date < self.DateofHire:
            raise ValueError("Last performance review must occur after DateofHire")
        return self

    def normalized_payload(self) -> Dict[str, Any]:
        data = self.model_dump()
        for field in ("DateofHire", "DOB", "LastPerformanceReview_Date"):
            data[field] = data[field].isoformat()
        return data


__all__ = ["EmployeeRecord", "ValidationError"]

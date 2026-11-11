"""
================================================================================
Pydantic Schemas for FastAPI Inference Engine
================================================================================
Comprehensive validation, documentation, and serialization schemas for requests
and responses across all event intelligence tasks.
================================================================================
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class EventCategory(str, Enum):
    ACADEMIC = "academic"
    CULTURAL = "cultural"
    FUN_CLUB = "fun_club"
    SPORTS = "sports"
    TECHNICAL = "technical"


class EventType(str, Enum):
    NORMAL = "Normal"
    IMPORTANT = "Important"
    VERY_IMPORTANT = "Very important"


class FacultyDesignation(str, Enum):
    ASSISTANT_PROFESSOR = "Assistant Professor"
    ASSOCIATE_PROFESSOR = "Associate Professor"
    PROFESSOR = "Professor"


# ------------------------------------------------------------------------------
# High-Level Event Proposal Input (Translates raw booking request into features)
# ------------------------------------------------------------------------------
class EventProposalRequest(BaseModel):
    event_name: Optional[str] = Field(default="RVCE Tech Symposium", description="Name of the festival event")
    request_date: str = Field(..., description="Date of event (YYYY-MM-DD or MM/DD/YYYY)", example="2026-11-20")
    submitted_date: Optional[str] = Field(default="2026-11-10", description="Date request was submitted", example="2026-11-10")
    requested_time: str = Field(..., description="Time of event (e.g. '14:30' or '02:30PM')", example="14:30")
    duration_min: int = Field(..., ge=15, le=720, description="Duration of event in minutes", example=90)
    room_no: int = Field(..., ge=1, le=999, description="Campus Room or Venue identifier", example=101)
    room_capacity: int = Field(..., ge=10, le=5000, description="Seating capacity of room", example=350)
    category: EventCategory = Field(..., description="Genre of event", example=EventCategory.TECHNICAL)
    event_type: EventType = Field(default=EventType.IMPORTANT, description="Priority level", example=EventType.IMPORTANT)
    faculty_designation: FacultyDesignation = Field(
        default=FacultyDesignation.ASSOCIATE_PROFESSOR,
        description="Designation of organizing faculty lead"
    )
    faculty_event_count: Optional[int] = Field(default=35, ge=0, description="Historical events by faculty")
    room_usage_count: Optional[int] = Field(default=75, ge=0, description="Historical bookings for room")
    earlier_clash_count: Optional[int] = Field(
        default=0, ge=0,
        description="Number of detected spatio-temporal overlapping bookings for the venue"
    )


# ------------------------------------------------------------------------------
# Direct Numerical Feature Vector Inputs
# ------------------------------------------------------------------------------
class AttendanceFeatureInput(BaseModel):
    month: int = Field(..., ge=1, le=12, example=11)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Mon, 6=Sun", example=4)
    is_saturday: int = Field(..., ge=0, le=1, example=0)
    week_of_year: int = Field(..., ge=1, le=53, example=46)
    is_exam_season: int = Field(..., ge=0, le=1, example=1)
    hour: int = Field(..., ge=0, le=23, example=14)
    is_afternoon: int = Field(..., ge=0, le=1, example=1)
    duration_min: int = Field(..., ge=15, le=720, example=90)
    lead_days: int = Field(..., ge=0, le=365, example=10)
    type_level: int = Field(..., ge=1, le=3, description="1=Normal, 2=Important, 3=Very important", example=2)
    designation_level: int = Field(..., ge=1, le=3, description="1=Asst, 2=Assoc, 3=Prof", example=2)
    room_no: int = Field(..., ge=1, le=999, example=101)
    room_capacity: int = Field(..., ge=10, le=5000, example=350)
    faculty_event_count: int = Field(..., ge=0, example=40)
    room_usage_count: int = Field(..., ge=0, example=65)
    earlier_clash_count: int = Field(..., ge=0, example=0)
    cat_academic: int = Field(..., ge=0, le=1, example=0)
    cat_cultural: int = Field(..., ge=0, le=1, example=0)
    cat_fun_club: int = Field(..., ge=0, le=1, example=0)
    cat_sports: int = Field(..., ge=0, le=1, example=0)
    cat_technical: int = Field(..., ge=0, le=1, example=1)


class ApprovalFeatureInput(AttendanceFeatureInput):
    fill_ratio: float = Field(..., ge=0.0, le=10.0, description="Expected registrations / room capacity", example=0.68)


class BudgetFeatureInput(AttendanceFeatureInput):
    TARGET_registers: float = Field(..., ge=0.0, description="Expected attendance turnout", example=240.0)


# ------------------------------------------------------------------------------
# Response Schemas
# ------------------------------------------------------------------------------
class AttendancePredictionResponse(BaseModel):
    predicted_attendance: float = Field(..., description="Forecasted student registrations")
    model_name: str
    model_version: str
    feature_count: int
    turnout_bracket: str = Field(..., description="E.g. Low (<50), Medium (50-150), High (>150)")


class ApprovalPredictionResponse(BaseModel):
    decision: str = Field(..., description="'ACCEPT' or 'REJECT'")
    decision_binary: int = Field(..., description="1 for Accept, 0 for Reject")
    approval_probability: float = Field(..., description="Model confidence in approval (0.0 - 1.0)")
    rejection_probability: float = Field(..., description="Model confidence in rejection (0.0 - 1.0)")
    is_capacity_exceeded: bool
    is_clash_detected: bool
    risk_score: float = Field(..., description="Risk score from 0.0 (Safe) to 1.0 (Critical Risk)")
    explanation: str
    model_name: str
    model_version: str


class BudgetPredictionResponse(BaseModel):
    estimated_budget_inr: float = Field(..., description="Forecasted operational event budget in INR")
    budget_range_lower: float = Field(..., description="Lower 95% bound")
    budget_range_upper: float = Field(..., description="Upper 95% bound")
    model_name: str
    model_version: str
    category_note: str


class ComprehensiveEventDecisionResponse(BaseModel):
    event_summary: Dict[str, Any]
    attendance_forecast: AttendancePredictionResponse
    approval_evaluation: ApprovalPredictionResponse
    budget_allocation: BudgetPredictionResponse
    capacity_analysis: Dict[str, Any]
    recommendations: List[str]
    audit_trace: Dict[str, Any]


class BatchPredictionRequest(BaseModel):
    proposals: List[EventProposalRequest] = Field(..., max_length=100)


class BatchPredictionResponse(BaseModel):
    total_processed: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str
    models_loaded: Dict[str, bool]
    timestamp: str


class ReadinessResponse(BaseModel):
    status: str
    models_ready: bool
    registered_models: List[str]


class ModelInfoResponse(BaseModel):
    registry: Dict[str, Any]


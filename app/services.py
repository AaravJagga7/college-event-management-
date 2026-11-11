"""
================================================================================
Event Intelligence Inference & Feature Engineering Services
================================================================================
Coordinates feature extraction, data transformations, model inference,
risk assessment, and decision explanations.
================================================================================
"""

import datetime
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

from app.registry import registry
from app.schemas import (
    EventProposalRequest,
    AttendanceFeatureInput,
    ApprovalFeatureInput,
    BudgetFeatureInput,
    AttendancePredictionResponse,
    ApprovalPredictionResponse,
    BudgetPredictionResponse,
    ComprehensiveEventDecisionResponse,
    EventType,
    FacultyDesignation,
    EventCategory
)

logger = logging.getLogger("uvicorn.error")


class EventInferenceService:
    """Service handling ML inference and end-to-end event evaluation."""

    def __init__(self):
        self.registry = registry

    def parse_datetime(self, date_str: str) -> datetime.date:
        """Parses dates across ISO (YYYY-MM-DD) and US (MM/DD/YYYY) formats."""
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"):
            try:
                return datetime.datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                pass
        raise ValueError(f"Unable to parse date string: {date_str}. Expected YYYY-MM-DD or MM/DD/YYYY.")

    def parse_time(self, time_str: str) -> datetime.time:
        """Parses time across 24h (HH:MM) and 12h (hh:mmAM/PM) formats."""
        cleaned = time_str.strip().upper()
        for fmt in ("%H:%M", "%I:%M%p", "%I:%M %p", "%H:%M:%S"):
            try:
                return datetime.datetime.strptime(cleaned, fmt).time()
            except ValueError:
                pass
        raise ValueError(f"Unable to parse time string: {time_str}. Expected 24h '14:30' or 12h '02:30PM'.")

    def proposal_to_feature_dict(self, proposal: EventProposalRequest) -> Dict[str, Any]:
        """Transforms a human-readable event booking request into raw features."""
        event_date = self.parse_datetime(proposal.request_date)
        submit_date = self.parse_datetime(proposal.submitted_date) if proposal.submitted_date else event_date
        event_time = self.parse_time(proposal.requested_time)

        month = event_date.month
        day_of_week = event_date.weekday()  # 0=Monday, 6=Sunday
        is_saturday = 1 if day_of_week == 5 else 0
        week_of_year = event_date.isocalendar()[1]
        is_exam_season = 1 if month in [4, 5, 11, 12] else 0

        hour = event_time.hour
        is_afternoon = 1 if hour >= 14 else 0
        lead_days = max(0, (event_date - submit_date).days)

        type_level_map = {
            EventType.NORMAL: 1,
            EventType.IMPORTANT: 2,
            EventType.VERY_IMPORTANT: 3
        }
        type_level = type_level_map.get(proposal.event_type, 2)

        desig_level_map = {
            FacultyDesignation.ASSISTANT_PROFESSOR: 1,
            FacultyDesignation.ASSOCIATE_PROFESSOR: 2,
            FacultyDesignation.PROFESSOR: 3
        }
        designation_level = desig_level_map.get(proposal.faculty_designation, 2)

        features = {
            "month": month,
            "day_of_week": day_of_week,
            "is_saturday": is_saturday,
            "week_of_year": week_of_year,
            "is_exam_season": is_exam_season,
            "hour": hour,
            "is_afternoon": is_afternoon,
            "duration_min": proposal.duration_min,
            "lead_days": lead_days,
            "type_level": type_level,
            "designation_level": designation_level,
            "room_no": proposal.room_no,
            "room_capacity": proposal.room_capacity,
            "faculty_event_count": proposal.faculty_event_count or 35,
            "room_usage_count": proposal.room_usage_count or 75,
            "earlier_clash_count": proposal.earlier_clash_count or 0,
            "cat_academic": 1 if proposal.category == EventCategory.ACADEMIC else 0,
            "cat_cultural": 1 if proposal.category == EventCategory.CULTURAL else 0,
            "cat_fun_club": 1 if proposal.category == EventCategory.FUN_CLUB else 0,
            "cat_sports": 1 if proposal.category == EventCategory.SPORTS else 0,
            "cat_technical": 1 if proposal.category == EventCategory.TECHNICAL else 0,
        }
        return features

    def predict_attendance(self, features_dict: Dict[str, Any]) -> AttendancePredictionResponse:
        """Predicts expected attendee registrations using Champion Regressor."""
        meta = self.registry.get_metadata("attendance")
        model = self.registry.load_model("attendance")
        feature_cols = meta["features"]

        df_in = pd.DataFrame([{col: features_dict[col] for col in feature_cols}])
        pred = float(model.predict(df_in)[0])
        pred = max(5.0, round(pred, 1))

        if pred < 50:
            bracket = "Low Turnout (<50)"
        elif pred <= 150:
            bracket = "Medium Turnout (50-150)"
        else:
            bracket = "High Turnout (>150)"

        return AttendancePredictionResponse(
            predicted_attendance=pred,
            model_name=meta["model_name"],
            model_version=meta["version"],
            feature_count=len(feature_cols),
            turnout_bracket=bracket
        )

    def predict_approval(self, features_dict: Dict[str, Any]) -> ApprovalPredictionResponse:
        """Evaluates booking approval decision and clash/capacity risk factors."""
        meta = self.registry.get_metadata("approval")
        model = self.registry.load_model("approval")
        feature_cols = meta["features"]

        df_in = pd.DataFrame([{col: features_dict[col] for col in feature_cols}])
        pred_cls = int(model.predict(df_in)[0])

        proba = model.predict_proba(df_in)[0] if hasattr(model, "predict_proba") else [0.0, 1.0]
        prob_accept = float(proba[1])
        prob_reject = float(proba[0])

        fill_ratio = features_dict.get("fill_ratio", 0.0)
        earlier_clash = features_dict.get("earlier_clash_count", 0)

        is_capacity_exceeded = fill_ratio > 1.0
        is_clash_detected = earlier_clash > 0

        # Compute risk score (0.0=lowest risk, 1.0=critical risk)
        risk_score = min(1.0, (1.0 - prob_accept) * 0.7 + (0.3 if is_clash_detected else 0.0) + (0.2 if is_capacity_exceeded else 0.0))
        risk_score = round(risk_score, 3)

        # Contextual explanation
        reasons = []
        if is_clash_detected:
            reasons.append(f"Detected {earlier_clash} overlapping booking(s) for this room/time slot.")
        if is_capacity_exceeded:
            reasons.append(f"Expected attendance exceeds venue seating capacity by {round((fill_ratio - 1.0) * 100, 1)}%.")
        if features_dict.get("is_exam_season", 0) == 1:
            reasons.append("Scheduled during college examination season.")

        if pred_cls == 1:
            decision = "ACCEPT"
            explanation = "Event booking meets venue capacity guidelines with no unresolved room scheduling clashes."
            if reasons:
                explanation += " Note: " + " ".join(reasons)
        else:
            decision = "REJECT"
            explanation = "Event booking rejected by automated policy. Primary causes: " + (
                "; ".join(reasons) if reasons else "High rejection probability based on scheduling patterns."
            )

        return ApprovalPredictionResponse(
            decision=decision,
            decision_binary=pred_cls,
            approval_probability=round(prob_accept, 4),
            rejection_probability=round(prob_reject, 4),
            is_capacity_exceeded=is_capacity_exceeded,
            is_clash_detected=is_clash_detected,
            risk_score=risk_score,
            explanation=explanation,
            model_name=meta["model_name"],
            model_version=meta["version"]
        )

    def predict_budget(self, features_dict: Dict[str, Any]) -> BudgetPredictionResponse:
        """Forecasts required event operational budget in INR (₹)."""
        meta = self.registry.get_metadata("budget")
        model = self.registry.load_model("budget")
        feature_cols = meta["features"]

        df_in = pd.DataFrame([{col: features_dict[col] for col in feature_cols}])
        pred = float(model.predict(df_in)[0])
        pred = max(5000.0, round(pred, 2))

        # 95% Confidence Interval based on model test RMSE (~ ₹4,488)
        rmse = meta["metrics"].get("test_rmse", 4488.22)
        lower_bound = max(5000.0, round(pred - 1.96 * rmse, 2))
        upper_bound = round(pred + 1.96 * rmse, 2)

        return BudgetPredictionResponse(
            estimated_budget_inr=pred,
            budget_range_lower=lower_bound,
            budget_range_upper=upper_bound,
            model_name=meta["model_name"],
            model_version=meta["version"],
            category_note="Budget estimation incorporates sound, stage, AV equipment, and participant hospitality."
        )

    def evaluate_full_proposal(self, proposal: EventProposalRequest) -> ComprehensiveEventDecisionResponse:
        """
        Executes unified triple-task decision flow:
          1. Predicts attendance without leakage
          2. Computes fill ratio
          3. Evaluates room booking approval
          4. Predicts required budget
          5. Synthesizes proactive recommendations
        """
        raw_features = self.proposal_to_feature_dict(proposal)

        # 1. Attendance Forecast
        attendance_resp = self.predict_attendance(raw_features)
        predicted_turnout = attendance_resp.predicted_attendance

        # 2. Fill Ratio Computation
        fill_ratio = round(predicted_turnout / float(proposal.room_capacity), 3)
        raw_features["fill_ratio"] = fill_ratio

        # 3. Approval Classification
        approval_resp = self.predict_approval(raw_features)

        # 4. Budget Forecast (uses predicted attendance)
        raw_features["TARGET_registers"] = predicted_turnout
        budget_resp = self.predict_budget(raw_features)

        # 5. Capacity Analysis
        capacity_status = "Optimal"
        if fill_ratio > 1.0:
            capacity_status = "Overcrowded (Violation)"
        elif fill_ratio < 0.25:
            capacity_status = "Underutilized Venue"
        elif fill_ratio >= 0.70:
            capacity_status = "Near Full Capacity"

        capacity_analysis = {
            "room_capacity": proposal.room_capacity,
            "forecasted_attendance": predicted_turnout,
            "fill_ratio": fill_ratio,
            "capacity_status": capacity_status,
            "available_seats": max(0, proposal.room_capacity - int(predicted_turnout)),
            "excess_students": max(0, int(predicted_turnout) - proposal.room_capacity)
        }

        # 6. Prescriptive Recommendations
        recs = []
        if approval_resp.is_clash_detected:
            recs.append("CRITICAL: Reschedule this event to an alternate time slot or relocate to an unbooked venue.")
        if fill_ratio > 1.0:
            recs.append(f"ACTION REQUIRED: Projected attendance ({int(predicted_turnout)}) exceeds room capacity ({proposal.room_capacity}). Upgrade to a larger auditorium (e.g. IEM Auditorium or Main Ground).")
        elif fill_ratio < 0.25:
            recs.append(f"OPTIMIZATION: Venue is heavily underutilized ({round(fill_ratio * 100, 1)}% fill). Consider a smaller seminar hall to conserve energy and AV budget.")
        if raw_features.get("is_exam_season", 0) == 1:
            recs.append("ADVISORY: Event falls during examination season; student turnout may experience downward variance.")
        if not recs:
            recs.append("All operational checks passed. Safe to confirm reservation and disburse budget.")

        return ComprehensiveEventDecisionResponse(
            event_summary={
                "event_name": proposal.event_name,
                "request_date": proposal.request_date,
                "requested_time": proposal.requested_time,
                "duration_min": proposal.duration_min,
                "room_no": proposal.room_no,
                "category": proposal.category.value,
                "type": proposal.event_type.value
            },
            attendance_forecast=attendance_resp,
            approval_evaluation=approval_resp,
            budget_allocation=budget_resp,
            capacity_analysis=capacity_analysis,
            recommendations=recs,
            audit_trace={
                "pipeline_version": "1.0.0",
                "lakefs_commit": self.registry.get_metadata("attendance")["lineage"].get("commit_id", "c_74c9896903"),
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
        )


inference_service = EventInferenceService()


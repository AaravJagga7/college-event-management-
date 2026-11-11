"""
================================================================================
FastAPI Inference Service Integration Tests
================================================================================
Validates REST API routes, request/response serialization, input validation,
error handling (422), latency headers, and decision recommendations.
================================================================================
"""

import pytest


def test_root_endpoint(client):
    """Verifies service banner and status."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "version" in data


def test_ui_dashboard_endpoint(client):
    """Verifies that the interactive HTML UI dashboard is served properly."""
    res = client.get("/ui")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "RVCE 8th Mile" in res.text
    assert "Scenario Presets" in res.text

    # Also check /dashboard alias
    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "RVCE 8th Mile" in res_dash.text


def test_health_check_endpoint(client):
    """Liveness probe test."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "models_loaded" in data
    assert data["models_loaded"]["attendance"] is True
    assert data["models_loaded"]["approval"] is True
    assert data["models_loaded"]["budget"] is True


def test_readiness_probe_endpoint(client):
    """Readiness probe test."""
    res = client.get("/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "READY"
    assert data["models_ready"] is True
    assert "attendance" in data["registered_models"]


def test_model_registry_metadata_endpoint(client):
    """Tests model registry metadata and lakeFS lineage."""
    res = client.get("/registry/models")
    assert res.status_code == 200
    data = res.json()
    reg = data["registry"]
    assert "attendance" in reg
    assert "approval" in reg
    assert "budget" in reg
    assert reg["budget"]["model_name"] == "college-budget-champion"
    assert "lineage" in reg["budget"]
    assert reg["budget"]["lineage"]["branch"] == "main"


def test_model_promotion_endpoint(client):
    """Tests lifecycle promotion endpoint."""
    res = client.post("/registry/promote?task=budget&stage=Staging")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["detail"]["current_stage"] == "Staging"

    # Promote back to Production
    res2 = client.post("/registry/promote?task=budget&stage=Production")
    assert res2.status_code == 200
    assert res2.json()["detail"]["current_stage"] == "Production"


def test_kubeflow_model_registry_endpoint(client):
    """Tests Kubeflow Model Registry export endpoint adhering to KFMR v1alpha3 spec."""
    res = client.get("/registry/kubeflow")
    assert res.status_code == 200
    data = res.json()
    assert data["backend"] == "kubeflow-model-registry"
    assert "registered_models" in data
    assert len(data["registered_models"]) == 3
    assert "model_versions" in data
    assert "model_artifacts" in data
    registered_names = [rm["name"] for rm in data["registered_models"]]
    assert "college-attendance-champion" in registered_names
    assert "college-approval-champion" in registered_names
    assert "college-budget-champion" in registered_names


def test_predict_attendance_endpoint(client):
    """Tests direct attendance inference route."""
    payload = {
        "month": 11,
        "day_of_week": 4,
        "is_saturday": 0,
        "week_of_year": 46,
        "is_exam_season": 1,
        "hour": 15,
        "is_afternoon": 1,
        "duration_min": 90,
        "lead_days": 7,
        "type_level": 2,
        "designation_level": 2,
        "room_no": 101,
        "room_capacity": 350,
        "faculty_event_count": 40,
        "room_usage_count": 70,
        "earlier_clash_count": 0,
        "cat_academic": 0,
        "cat_cultural": 1,
        "cat_fun_club": 0,
        "cat_sports": 0,
        "cat_technical": 0
    }
    res = client.post("/predict/attendance", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_attendance" in data
    assert data["predicted_attendance"] > 0
    assert "turnout_bracket" in data
    assert "X-Process-Time-Ms" in res.headers


def test_predict_approval_endpoint(client):
    """Tests approval decision route."""
    payload = {
        "month": 11,
        "day_of_week": 4,
        "is_saturday": 0,
        "week_of_year": 46,
        "is_exam_season": 1,
        "hour": 15,
        "is_afternoon": 1,
        "duration_min": 90,
        "lead_days": 7,
        "type_level": 2,
        "designation_level": 2,
        "room_no": 101,
        "room_capacity": 350,
        "faculty_event_count": 40,
        "room_usage_count": 70,
        "earlier_clash_count": 0,
        "cat_academic": 0,
        "cat_cultural": 1,
        "cat_fun_club": 0,
        "cat_sports": 0,
        "cat_technical": 0,
        "fill_ratio": 0.65
    }
    res = client.post("/predict/approval", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["decision"] in ["ACCEPT", "REJECT"]
    assert 0.0 <= data["approval_probability"] <= 1.0
    assert "risk_score" in data


def test_predict_budget_endpoint(client):
    """Tests financial budget allocation route."""
    payload = {
        "month": 11,
        "day_of_week": 4,
        "is_saturday": 0,
        "week_of_year": 46,
        "is_exam_season": 1,
        "hour": 15,
        "is_afternoon": 1,
        "duration_min": 90,
        "lead_days": 7,
        "type_level": 2,
        "designation_level": 2,
        "room_no": 101,
        "room_capacity": 350,
        "faculty_event_count": 40,
        "room_usage_count": 70,
        "earlier_clash_count": 0,
        "cat_academic": 0,
        "cat_cultural": 1,
        "cat_fun_club": 0,
        "cat_sports": 0,
        "cat_technical": 0,
        "TARGET_registers": 220.0
    }
    res = client.post("/predict/budget", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["estimated_budget_inr"] > 5000.0
    assert data["budget_range_lower"] < data["budget_range_upper"]


def test_predict_full_evaluation_endpoint(client, sample_proposal_payload):
    """Tests end-to-end decision engine with recommendations."""
    res = client.post("/predict/all", json=sample_proposal_payload)
    assert res.status_code == 200
    data = res.json()
    assert "attendance_forecast" in data
    assert "approval_evaluation" in data
    assert "budget_allocation" in data
    assert "capacity_analysis" in data
    assert len(data["recommendations"]) > 0
    assert "audit_trace" in data


def test_predict_batch_endpoint(client, sample_proposal_payload):
    """Tests batch inference endpoint."""
    batch_req = {
        "proposals": [
            sample_proposal_payload,
            {**sample_proposal_payload, "event_name": "Hackathon 2026", "room_capacity": 500, "category": "technical"}
        ]
    }
    res = client.post("/predict/batch", json=batch_req)
    assert res.status_code == 200
    data = res.json()
    assert data["total_processed"] == 2
    assert data["successful"] == 2
    assert data["failed"] == 0


def test_input_validation_error(client):
    """Ensures Pydantic 422 Unprocessable Entity on illegal inputs."""
    # Invalid month (month=13) and negative duration
    invalid_payload = {
        "month": 13,
        "day_of_week": 4,
        "is_saturday": 0,
        "week_of_year": 46,
        "is_exam_season": 1,
        "hour": 15,
        "is_afternoon": 1,
        "duration_min": -30,
        "lead_days": 7,
        "type_level": 2,
        "designation_level": 2,
        "room_no": 101,
        "room_capacity": 350,
        "faculty_event_count": 40,
        "room_usage_count": 70,
        "earlier_clash_count": 0,
        "cat_academic": 0,
        "cat_cultural": 1,
        "cat_fun_club": 0,
        "cat_sports": 0,
        "cat_technical": 0
    }
    res = client.post("/predict/attendance", json=invalid_payload)
    assert res.status_code == 422


def test_monitoring_stats_endpoint(client):
    """Verifies that the /monitoring/stats endpoint returns telemetry metrics and SLA status."""
    res = client.get("/monitoring/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OPERATIONAL"
    assert "uptime_seconds" in data
    assert "total_requests" in data
    assert "latency_ms" in data
    assert "sla_threshold_ms" in data["latency_ms"]
    assert data["latency_ms"]["sla_threshold_ms"] == 25.0
    assert "system_resources" in data
    assert "models_status" in data
    assert data["models_status"]["attendance"] is True


def test_prometheus_metrics_endpoint(client):
    """Verifies that the /metrics endpoint returns Prometheus exposition format."""
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]
    text = res.text
    assert "rvce_uptime_seconds" in text
    assert "rvce_requests_total" in text
    assert "rvce_inference_latency_ms" in text


def test_artifacts_catalog_and_serving(client):
    """Verifies ML artifacts listing and static artifact file serving."""
    res = client.get("/api/artifacts")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 10
    assert len(data["artifacts"]) == 10

    # Test serving a valid artifact file
    art_res = client.get("/artifacts/ml_architecture.png")
    assert art_res.status_code == 200
    assert art_res.headers["content-type"].startswith("image/")

    # Test 404 on nonexistent artifact
    nonexistent = client.get("/artifacts/non_existent_plot.png")
    assert nonexistent.status_code == 404



"""
Pytest Test Fixtures and Setup
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from fastapi.testclient import TestClient

from app.main import app
from app.registry import registry
from app.services import inference_service


@pytest.fixture(scope="session")
def client():
    """FastAPI Test Client with lifespan initialization."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def test_dataset():
    """Loads features_ready.csv for testing."""
    df = pd.read_csv("features_ready.csv")
    return df


@pytest.fixture(scope="session")
def champion_models():
    """Pre-loaded champion models."""
    return {
        "attendance": registry.load_model("attendance"),
        "approval": registry.load_model("approval"),
        "budget": registry.load_model("budget"),
    }


@pytest.fixture
def sample_proposal_payload():
    """Valid event booking proposal payload."""
    return {
        "event_name": "Ashtrang Cultural Night",
        "request_date": "2026-11-20",
        "submitted_date": "2026-11-10",
        "requested_time": "15:00",
        "duration_min": 120,
        "room_no": 101,
        "room_capacity": 350,
        "category": "cultural",
        "event_type": "Important",
        "faculty_designation": "Associate Professor",
        "faculty_event_count": 45,
        "room_usage_count": 80,
        "earlier_clash_count": 0
    }


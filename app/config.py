"""
Application Configuration
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    app_name: str = "RVCE 8th Mile Event Intelligence API"
    app_version: str = "1.0.0"
    app_description: str = (
        "Production ML Inference Service for Attendance Forecasting, "
        "Room Booking Approval, and Budget Allocation for RVCE Events."
    )
    environment: str = os.getenv("APP_ENV", "production")
    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))
    
    # Model Artifact Paths
    models_dir: str = os.getenv("MODELS_DIR", "reports/models")
    champion_metrics_path: str = os.getenv("CHAMPION_METRICS_PATH", "reports/champion_metrics.json")
    kubeflow_registry_uri: str = os.getenv("KUBEFLOW_REGISTRY_URI", "http://localhost:8080")
    kubeflow_registry_file: str = os.getenv("KUBEFLOW_REGISTRY_FILE", "reports/kubeflow_model_registry.json")
    
    # Feature store / baseline data path for drift & testing
    feature_store_path: str = os.getenv("FEATURE_STORE_PATH", "features_ready.csv")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()


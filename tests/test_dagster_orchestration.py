"""
================================================================================
Unit & Integration Tests for Dagster Model Orchestration & lakeFS Integration
================================================================================
Verifies:
1. Dagster Definitions compilation & Asset graph topology
2. LakeFSResource interface & URI generation
3. Software-Defined Asset (SDA) materialization
4. Atomic lakeFS commit creation & cryptographic hash lineage
================================================================================
"""

import os
import pytest
from dagster import materialize
from dagster_orchestration.definitions import defs, all_mlops_assets
from dagster_orchestration.resources import LakeFSResource


class TestDagsterLakeFSOrchestration:
    """Test suite for Dagster pipeline execution with lakeFS data lake."""

    def test_dagster_definitions_structure(self):
        """Verifies that all 8 assets, jobs, and lakeFS resources are registered in Dagster defs."""
        asset_keys = {a.key.to_user_string() for a in defs.assets}
        expected_keys = {
            "raw_event_data",
            "features_ready",
            "attendance_model",
            "approval_model",
            "budget_model",
            "model_quality_gate",
            "kubeflow_model_registry",
            "lakefs_pipeline_commit"
        }
        assert expected_keys.issubset(asset_keys), f"Missing assets in Dagster graph: {expected_keys - asset_keys}"

        job = defs.get_job_def("rvce_event_intelligence_dagster_pipeline")
        assert job is not None
        assert job.name == "rvce_event_intelligence_dagster_pipeline"

    def test_lakefs_resource_and_uris(self):
        """Verifies LakeFSResource configuration and canonical lakefs:// URI formatting."""
        lakefs = LakeFSResource()
        uri_raw = lakefs.get_lakefs_uri("raw/rvce_8th_mile_data.csv")
        assert uri_raw == "lakefs://college-event-data/main/raw/rvce_8th_mile_data.csv"

        uri_model = lakefs.get_lakefs_uri("/models/champion_attendance_model.joblib")
        assert uri_model == "lakefs://college-event-data/main/models/champion_attendance_model.joblib"

        latest = lakefs.get_latest_commit()
        assert "id" in latest
        assert "objects" in latest

    def test_dagster_asset_materialization(self):
        """Executes in-process materialization of the full asset graph."""
        lakefs_res = LakeFSResource()
        result = materialize(
            all_mlops_assets,
            resources={"lakefs": lakefs_res}
        )
        assert result.success, "Dagster asset materialization failed"

        # Verify lakeFS commit was produced
        latest_commit = lakefs_res.get_latest_commit()
        assert latest_commit.get("id", "").startswith("c_")
        assert "raw/rvce_8th_mile_data.csv" in latest_commit["objects"]
        assert "processed/features_ready.csv" in latest_commit["objects"]

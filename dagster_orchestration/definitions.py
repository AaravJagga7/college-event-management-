"""
================================================================================
Dagster Definitions: RVCE 8th Mile MLOps Platform
================================================================================
Binds Software-Defined Assets, LakeFS resource, and pipeline orchestration jobs.
================================================================================
"""

from dagster import (
    Definitions,
    define_asset_job,
    AssetSelection,
    ScheduleDefinition
)

from dagster_orchestration.resources import LakeFSResource
from dagster_orchestration.assets import (
    raw_event_data,
    features_ready,
    attendance_model,
    approval_model,
    budget_model,
    model_quality_gate,
    kubeflow_model_registry,
    lakefs_pipeline_commit
)

# Define asset list
all_mlops_assets = [
    raw_event_data,
    features_ready,
    attendance_model,
    approval_model,
    budget_model,
    model_quality_gate,
    kubeflow_model_registry,
    lakefs_pipeline_commit
]

# Pipeline Job executing the full data-to-model-to-registry-to-lakefs lifecycle
rvce_mlops_job = define_asset_job(
    name="rvce_event_intelligence_dagster_pipeline",
    selection=AssetSelection.all(),
    description="End-to-end Dagster orchestration with lakeFS data lake versioning and Kubeflow registry promotion."
)

# Daily Schedule Definition (matches festival daily operational cycle)
daily_mlops_schedule = ScheduleDefinition(
    job=rvce_mlops_job,
    cron_schedule="0 2 * * *",  # Daily at 02:00 UTC
    execution_timezone="UTC"
)

# Export Dagster Definitions object (standard entrypoint for dagster-webserver and dagster CLI)
defs = Definitions(
    assets=all_mlops_assets,
    jobs=[rvce_mlops_job],
    schedules=[daily_mlops_schedule],
    resources={
        "lakefs": LakeFSResource()
    }
)

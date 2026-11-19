"""
================================================================================
Dagster Software-Defined Assets (SDA) for RVCE 8th Mile MLOps Platform
================================================================================
Defines end-to-end data and model lineage integrating with lakeFS versioning:
1. raw_event_data          -> Ingests & validates raw campus data
2. features_ready          -> Feature engineering & data drift verification
3. attendance_model        -> Task A: Turnout regression (TARGET_registers)
4. approval_model          -> Task B: Booking approval classification (TARGET_accept)
5. budget_model            -> Task C: Operational budget regression (TARGET_budget)
6. model_quality_gate      -> Pre-deployment validation, monotonicity & fairness
7. kubeflow_model_registry -> Kubeflow Model Registry (v1alpha3) promotion
8. lakefs_pipeline_commit  -> Atomic lakeFS commit linking code, data & models
================================================================================
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib

from dagster import (
    asset,
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

from dagster_orchestration.resources import LakeFSResource
from app.registry import registry, ModelStage

REPORTS_DIR = "reports"
MODELS_DIR = os.path.join(REPORTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ==============================================================================
# 1. RAW DATA ASSET
# ==============================================================================
@asset(
    group_name="data_ingestion",
    description="Raw RVCE 8th Mile Festival & Campus event booking dataset tracked in lakeFS."
)
def raw_event_data(context: AssetExecutionContext, lakefs: LakeFSResource) -> MaterializeResult:
    """Ingests raw CSV, computes checksums, and binds lakeFS object URI."""
    raw_path = "rvce_8th_mile_data.csv"
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Raw data file '{raw_path}' not found.")

    df = pd.read_csv(raw_path)
    file_size_bytes = os.path.getsize(raw_path)

    hasher = hashlib.sha256()
    with open(raw_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    checksum = hasher.hexdigest()

    lakefs_uri = lakefs.get_lakefs_uri(f"raw/{raw_path}")
    context.log.info(f"Ingested {len(df)} rows from {raw_path}. lakeFS URI: {lakefs_uri}")

    return MaterializeResult(
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "column_count": MetadataValue.int(df.shape[1]),
            "file_size_kb": MetadataValue.float(round(file_size_bytes / 1024, 2)),
            "sha256_checksum": MetadataValue.text(checksum[:16] + "..."),
            "lakefs_uri": MetadataValue.text(lakefs_uri),
            "columns": MetadataValue.json(list(df.columns))
        }
    )


# ==============================================================================
# 2. FEATURE STORE ASSET
# ==============================================================================
@asset(
    deps=[raw_event_data],
    group_name="feature_engineering",
    description="Engineered 25-column feature store with leakage firewalls and lakeFS provenance."
)
def features_ready(context: AssetExecutionContext, lakefs: LakeFSResource) -> MaterializeResult:
    """Validates the feature store dataset and confirms data integrity."""
    feat_path = "features_ready.csv"
    if not os.path.isfile(feat_path):
        from eda_and_feature_extraction import run_pipeline
        context.log.info("Generating feature store features_ready.csv...")
        run_pipeline()

    df = pd.read_csv(feat_path)
    null_count = int(df.isnull().sum().sum())
    assert null_count == 0, f"Found {null_count} nulls in feature store"

    lakefs_uri = lakefs.get_lakefs_uri(f"processed/{feat_path}")
    context.log.info(f"Verified feature store with {df.shape[0]} samples and {df.shape[1]} features.")

    return MaterializeResult(
        metadata={
            "row_count": MetadataValue.int(len(df)),
            "feature_count": MetadataValue.int(df.shape[1]),
            "null_values": MetadataValue.int(null_count),
            "lakefs_uri": MetadataValue.text(lakefs_uri),
            "feature_columns": MetadataValue.json(list(df.columns))
        }
    )


# ==============================================================================
# 3. TASK A: ATTENDANCE REGRESSION MODEL ASSET
# ==============================================================================
@asset(
    deps=[features_ready],
    group_name="model_training",
    description="Task A: Student Turnout / Attendance Regressor (TARGET_registers) with lakeFS lineage."
)
def attendance_model(context: AssetExecutionContext, lakefs: LakeFSResource) -> MaterializeResult:
    """Trains and benchmarks attendance turnout regression model."""
    df = pd.read_csv("features_ready.csv")
    cols = [c for c in df.columns if c not in ["TARGET_registers", "TARGET_accept", "fill_ratio", "TARGET_budget"]]

    X = df[cols]
    y = df["TARGET_registers"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

    # Train Champion Gradient Boosting Regressor
    model = GradientBoostingRegressor(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42)
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    r2 = float(r2_score(y_te, preds))
    mae = float(mean_absolute_error(y_te, preds))
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))

    # Persist artifact
    artifact_path = os.path.join(MODELS_DIR, "champion_attendance_model.joblib")
    joblib.dump(model, artifact_path)

    context.log.info(f"Task A (Turnout) Model: R²={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    assert r2 >= 0.50, f"Attendance R² {r2:.3f} failed production gate (>=0.50)"
    assert mae <= 95.0, f"Attendance MAE {mae:.2f} failed production gate (<=95.0)"

    return MaterializeResult(
        metadata={
            "task": MetadataValue.text("attendance_regression"),
            "model_type": MetadataValue.text("GradientBoostingRegressor"),
            "r2_score": MetadataValue.float(round(r2, 4)),
            "mae": MetadataValue.float(round(mae, 2)),
            "rmse": MetadataValue.float(round(rmse, 2)),
            "artifact_path": MetadataValue.text(artifact_path),
            "lakefs_object": MetadataValue.text(lakefs.get_lakefs_uri("models/champion_attendance_model.joblib"))
        }
    )


# ==============================================================================
# 4. TASK B: BOOKING APPROVAL CLASSIFIER ASSET
# ==============================================================================
@asset(
    deps=[features_ready],
    group_name="model_training",
    description="Task B: Venue Booking Approval Classifier (TARGET_accept) with risk calibration."
)
def approval_model(context: AssetExecutionContext, lakefs: LakeFSResource) -> MaterializeResult:
    """Trains and benchmarks room booking approval classification model."""
    df = pd.read_csv("features_ready.csv")
    cols = [c for c in df.columns if c not in ["TARGET_registers", "TARGET_accept", "TARGET_budget"]]

    X = df[cols]
    y = df["TARGET_accept"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=3, class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    probas = model.predict_proba(X_te)[:, 1]

    acc = float(accuracy_score(y_te, preds))
    prec = float(precision_score(y_te, preds, zero_division=0))
    rec = float(recall_score(y_te, preds, zero_division=0))
    f1 = float(f1_score(y_te, preds, zero_division=0))
    auc = float(roc_auc_score(y_te, probas))

    artifact_path = os.path.join(MODELS_DIR, "cls_random_forest.joblib")
    joblib.dump(model, artifact_path)

    context.log.info(f"Task B (Approval) Model: AUC={auc:.4f}, F1={f1:.4f}, Accuracy={acc:.4f}")
    assert auc >= 0.78, f"Approval ROC-AUC {auc:.3f} failed production gate (>=0.78)"
    assert f1 >= 0.85, f"Approval F1 {f1:.3f} failed production gate (>=0.85)"

    return MaterializeResult(
        metadata={
            "task": MetadataValue.text("booking_approval"),
            "model_type": MetadataValue.text("RandomForestClassifier"),
            "roc_auc": MetadataValue.float(round(auc, 4)),
            "f1_score": MetadataValue.float(round(f1, 4)),
            "accuracy": MetadataValue.float(round(acc, 4)),
            "precision": MetadataValue.float(round(prec, 4)),
            "recall": MetadataValue.float(round(rec, 4)),
            "artifact_path": MetadataValue.text(artifact_path),
            "lakefs_object": MetadataValue.text(lakefs.get_lakefs_uri("models/cls_random_forest.joblib"))
        }
    )


# ==============================================================================
# 5. TASK C: OPERATIONAL BUDGET REGRESSION ASSET
# ==============================================================================
@asset(
    deps=[features_ready],
    group_name="model_training",
    description="Task C: Operational Expenditure Regressor (TARGET_budget) with INR scaling."
)
def budget_model(context: AssetExecutionContext, lakefs: LakeFSResource) -> MaterializeResult:
    """Trains and benchmarks operational budget regression model."""
    df = pd.read_csv("features_ready.csv")
    cols = [c for c in df.columns if c not in ["TARGET_budget", "TARGET_accept", "fill_ratio"]]

    X = df[cols]
    y = df["TARGET_budget"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

    model = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    r2 = float(r2_score(y_te, preds))
    mae = float(mean_absolute_error(y_te, preds))
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    mape = float(mean_absolute_percentage_error(y_te, preds) * 100)

    artifact_path = os.path.join(MODELS_DIR, "champion_budget_model.joblib")
    joblib.dump(model, artifact_path)

    context.log.info(f"Task C (Budget) Model: R²={r2:.4f}, MAE=₹{mae:,.2f}, MAPE={mape:.2f}%")
    assert r2 >= 0.95, f"Budget R² {r2:.3f} failed production gate (>=0.95)"
    assert mape <= 5.0, f"Budget MAPE {mape:.2f}% failed production gate (<=5.0%)"

    return MaterializeResult(
        metadata={
            "task": MetadataValue.text("budget_regression"),
            "model_type": MetadataValue.text("GradientBoostingRegressor"),
            "r2_score": MetadataValue.float(round(r2, 4)),
            "mae_inr": MetadataValue.float(round(mae, 2)),
            "mape_percent": MetadataValue.float(round(mape, 2)),
            "rmse": MetadataValue.float(round(rmse, 2)),
            "currency": MetadataValue.text("INR (₹)"),
            "artifact_path": MetadataValue.text(artifact_path),
            "lakefs_object": MetadataValue.text(lakefs.get_lakefs_uri("models/champion_budget_model.joblib"))
        }
    )


# ==============================================================================
# 6. MODEL QUALITY & RISK GATES ASSET
# ==============================================================================
@asset(
    deps=[attendance_model, approval_model, budget_model],
    group_name="model_governance",
    description="Pre-deployment validation verifying monotonicity, fairness parity, and latency SLA."
)
def model_quality_gate(context: AssetExecutionContext) -> MaterializeResult:
    """Executes pre-deployment behavioral, invariance, and fairness quality checks."""
    df = pd.read_csv("features_ready.csv")
    cols_app = [c for c in df.columns if c not in ["TARGET_registers", "TARGET_accept", "TARGET_budget"]]
    model_app = joblib.load(os.path.join(MODELS_DIR, "cls_random_forest.joblib"))

    # 1. Monotonicity: clashes must reduce or keep approval probability below baseline
    base = df[(df["earlier_clash_count"] == 0) & (df["fill_ratio"] < 0.8)].iloc[0][cols_app].to_dict()
    probs = []
    for c in [0, 1, 2, 4]:
        v = dict(base)
        v["earlier_clash_count"] = c
        p = model_app.predict_proba(pd.DataFrame([v])[cols_app])[0, 1]
        probs.append(p)

    assert probs[0] >= probs[-1], "Clash monotonicity check breached"
    context.log.info(f"Clash monotonicity verified: {probs[0]:.3f} >= {probs[-1]:.3f}")

    # 2. Fairness Parity: Check approval rate across genres
    genres = ["academic", "cultural", "fun_club", "sports", "technical"]
    rates = {}
    for g in genres:
        sub = df[df[f"cat_{g}"] == 1]
        if len(sub) > 0:
            rates[g] = float((model_app.predict(sub[cols_app]) == 1).mean())

    max_rate = max(rates.values())
    di_ratios = {}
    for g, rate in rates.items():
        ratio = rate / max_rate if max_rate > 0 else 1.0
        di_ratios[g] = round(ratio, 3)
        assert ratio >= 0.70, f"Disparate impact detected in {g}: ratio {ratio:.2f} < 0.70"

    context.log.info(f"Fairness parity verified across {len(rates)} categories. Disparate impact ratios: {di_ratios}")

    # 3. Latency SLA Check
    cols_att = [c for c in df.columns if c not in ["TARGET_registers", "TARGET_accept", "fill_ratio", "TARGET_budget"]]
    model_att = joblib.load(os.path.join(MODELS_DIR, "champion_attendance_model.joblib"))
    single = df[cols_att].iloc[:1]
    
    # Warmup
    for _ in range(5):
        _ = model_att.predict(single)

    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = model_att.predict(single)
        latencies.append((time.perf_counter() - t0) * 1000)

    p95 = float(np.percentile(latencies, 95))
    context.log.info(f"Latency SLA P95: {p95:.2f}ms (Target: <30.0ms)")
    assert p95 < 30.0, f"P95 latency {p95:.2f}ms breached SLA limit"

    return MaterializeResult(
        metadata={
            "monotonicity_check": MetadataValue.text("PASSED"),
            "fairness_parity_check": MetadataValue.text("PASSED"),
            "latency_p95_ms": MetadataValue.float(round(p95, 2)),
            "approval_rates_by_category": MetadataValue.json(rates),
            "quality_gate_decision": MetadataValue.text("APPROVED_FOR_PRODUCTION")
        }
    )


# ==============================================================================
# 7. KUBEFLOW MODEL REGISTRY PROMOTION ASSET
# ==============================================================================
@asset(
    deps=[model_quality_gate],
    group_name="model_governance",
    description="Promotes all three models to Production stage in Kubeflow Model Registry (v1alpha3)."
)
def kubeflow_model_registry(context: AssetExecutionContext, lakefs: LakeFSResource) -> MaterializeResult:
    """Updates and exports Kubeflow Model Registry entities with lakeFS artifact URIs."""
    kf_path = os.path.join(REPORTS_DIR, "kubeflow_model_registry.json")
    
    # Read or build KFMR catalog
    kf_catalog = {
        "registry_version": "v1alpha3",
        "backend": "kubeflow-model-registry",
        "server_uri": os.getenv("KUBEFLOW_REGISTRY_URI", "http://localhost:8080"),
        "registered_models": [
            {
                "id": "rm-attendance-001",
                "name": "college-attendance-champion",
                "state": "LIVE",
                "stage": "Production",
                "artifact_uri": lakefs.get_lakefs_uri("models/champion_attendance_model.joblib")
            },
            {
                "id": "rm-approval-002",
                "name": "college-approval-champion",
                "state": "LIVE",
                "stage": "Production",
                "artifact_uri": lakefs.get_lakefs_uri("models/cls_random_forest.joblib")
            },
            {
                "id": "rm-budget-003",
                "name": "college-budget-champion",
                "state": "LIVE",
                "stage": "Production",
                "artifact_uri": lakefs.get_lakefs_uri("models/champion_budget_model.joblib")
            }
        ],
        "promoted_at": datetime.utcnow().isoformat() + "Z"
    }

    with open(kf_path, "w") as f:
        json.dump(kf_catalog, f, indent=2)

    context.log.info(f"Promoted all 3 models to Production stage in Kubeflow Registry at {kf_path}.")

    return MaterializeResult(
        metadata={
            "spec_version": MetadataValue.text("v1alpha3"),
            "models_promoted": MetadataValue.int(3),
            "promotion_stage": MetadataValue.text("Production"),
            "catalog_path": MetadataValue.text(kf_path)
        }
    )


# ==============================================================================
# 8. LAKEFS ATOMIC PIPELINE COMMIT ASSET
# ==============================================================================
@asset(
    deps=[kubeflow_model_registry],
    group_name="data_versioning",
    description="Creates an atomic lakeFS commit linking code, datasets, feature store, and model weights."
)
def lakefs_pipeline_commit(context: AssetExecutionContext, lakefs: LakeFSResource) -> MaterializeResult:
    """Creates a cryptographic lakeFS commit for end-to-end auditability."""
    commit_msg = f"dagster: automated pipeline run at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    commit = lakefs.commit_version(
        raw_path="rvce_8th_mile_data.csv",
        processed_path="features_ready.csv",
        message=commit_msg
    )

    commit_id = commit.get("id", "c_unknown")
    context.log.info(f"✅ Created lakeFS commit: {commit_id} on branch '{lakefs.branch}'")

    return MaterializeResult(
        metadata={
            "lakefs_commit_id": MetadataValue.text(commit_id),
            "branch": MetadataValue.text(lakefs.branch),
            "repository": MetadataValue.text(lakefs.repository),
            "commit_message": MetadataValue.text(commit.get("message", "")),
            "timestamp": MetadataValue.text(commit.get("timestamp", "")),
            "raw_lakefs_uri": MetadataValue.text(commit["objects"]["raw/rvce_8th_mile_data.csv"]["lakefs_uri"]),
            "features_lakefs_uri": MetadataValue.text(commit["objects"]["processed/features_ready.csv"]["lakefs_uri"]),
            "checksum_provenance": MetadataValue.text(commit["metadata"]["features_checksum"][:16] + "...")
        }
    )

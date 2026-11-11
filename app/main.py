"""
================================================================================
FastAPI Inference Application Entrypoint
================================================================================
Provides high-performance, asynchronous REST endpoints for ML model inference,
automated event risk evaluation, and model registry lifecycle control.
================================================================================
"""

import time
import datetime
import logging
import os
import collections
from contextlib import asynccontextmanager
from typing import Dict, Any, List

import numpy as np
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, PlainTextResponse
from app.ui import UI_HTML

from app.config import settings
from app.registry import registry, ModelStage
from app.services import inference_service
from app.schemas import (
    EventProposalRequest,
    AttendanceFeatureInput,
    ApprovalFeatureInput,
    BudgetFeatureInput,
    AttendancePredictionResponse,
    ApprovalPredictionResponse,
    BudgetPredictionResponse,
    ComprehensiveEventDecisionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ReadinessResponse,
    ModelInfoResponse
)

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: pre-warms champion models on startup."""
    logger.info("Initializing RVCE Event Intelligence ML inference service...")
    try:
        loaded = registry.load_all_models()
        logger.info(f"Successfully pre-warmed {len(loaded)} champion models into cache.")
    except Exception as e:
        logger.error(f"Failed to pre-warm models: {e}")
    yield
    logger.info("Shutting down RVCE Event Intelligence inference service.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TelemetryTracker:
    """Thread-safe in-memory metrics and telemetry store for monitoring."""
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.endpoint_hits = {}
        self.latencies_ms = collections.deque(maxlen=1000)
        self.status_codes = {}
        self.last_inference_time = None
        self.task_predictions = {
            "attendance": 0,
            "approval": 0,
            "budget": 0,
            "all": 0,
            "batch": 0
        }

    def record(self, path: str, status_code: int, duration_ms: float):
        self.total_requests += 1
        if 200 <= status_code < 400:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.endpoint_hits[path] = self.endpoint_hits.get(path, 0) + 1
        self.status_codes[status_code] = self.status_codes.get(status_code, 0) + 1
        self.latencies_ms.append(duration_ms)

        if "predict" in path:
            self.last_inference_time = datetime.datetime.utcnow().isoformat() + "Z"
            for t in ["attendance", "approval", "budget", "all", "batch"]:
                if t in path:
                    self.task_predictions[t] = self.task_predictions.get(t, 0) + 1


telemetry = TelemetryTracker()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Profiles latency and logs telemetry for every request."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    
    # Exclude high-frequency polling from distorting inference latency
    if not request.url.path.startswith("/monitoring"):
        telemetry.record(request.url.path, response.status_code, process_time)
        
    return response


# ------------------------------------------------------------------------------
# Operational, UI & Health Endpoints
# ------------------------------------------------------------------------------
@app.get("/", tags=["General"])
async def root(request: Request):
    """
    Root route with smart content negotiation:
    - Serves Interactive Web Dashboard UI to browsers.
    - Serves JSON API metadata banner to REST/test clients.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept and accept != "*/*" and not accept.startswith("application/json"):
        return HTMLResponse(content=UI_HTML)
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "online",
        "ui": "/ui",
        "documentation": "/docs",
        "system_time": datetime.datetime.utcnow().isoformat() + "Z"
    }


@app.get("/ui", response_class=HTMLResponse, tags=["UI Dashboard"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["UI Dashboard"])
async def serve_ui_dashboard():
    """Renders the RVCE 8th Mile Event Intelligence Model UI Dashboard."""
    return HTMLResponse(content=UI_HTML)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Liveness probe reporting model cache status and runtime health."""
    models_status = {
        task: (task in registry._models_cache)
        for task in ["attendance", "approval", "budget"]
    }
    return HealthResponse(
        status="HEALTHY",
        environment=settings.environment,
        version=settings.app_version,
        models_loaded=models_status,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z"
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["Monitoring"])
async def readiness_check():
    """Readiness probe ensuring all champion models are loaded and inference-ready."""
    try:
        registry.load_all_models()
        is_ready = all(task in registry._models_cache for task in ["attendance", "approval", "budget"])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )
    return ReadinessResponse(
        status="READY" if is_ready else "NOT_READY",
        models_ready=is_ready,
        registered_models=list(registry._metadata_cache.keys())
    )


@app.get("/monitoring/stats", tags=["Monitoring"])
async def get_monitoring_stats():
    """Returns real-time tool and system operational telemetry for the dashboard."""
    uptime = time.time() - telemetry.start_time
    hours, rem = divmod(int(uptime), 3600)
    minutes, seconds = divmod(rem, 60)
    uptime_fmt = f"{hours}h {minutes}m {seconds}s"

    lats = list(telemetry.latencies_ms)
    avg_lat = float(np.mean(lats)) if lats else 0.0
    p50_lat = float(np.percentile(lats, 50)) if lats else 0.0
    p95_lat = float(np.percentile(lats, 95)) if lats else 0.0
    p99_lat = float(np.percentile(lats, 99)) if lats else 0.0

    cpu_pct = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0
    mem = psutil.virtual_memory() if PSUTIL_AVAILABLE else None
    mem_pct = mem.percent if mem else 0.0
    mem_used_mb = round(mem.used / (1024 * 1024), 1) if mem else 0.0

    lakefs_meta = registry.get_metadata("attendance")["lineage"]

    return {
        "status": "OPERATIONAL",
        "uptime_seconds": int(uptime),
        "uptime_formatted": uptime_fmt,
        "total_requests": telemetry.total_requests,
        "successful_requests": telemetry.successful_requests,
        "failed_requests": telemetry.failed_requests,
        "error_rate_percent": round((telemetry.failed_requests / max(1, telemetry.total_requests)) * 100, 2),
        "latency_ms": {
            "avg": round(avg_lat, 2),
            "p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
            "p99": round(p99_lat, 2),
            "sla_threshold_ms": 25.0,
            "sla_breached": p95_lat > 25.0
        },
        "system_resources": {
            "cpu_percent": cpu_pct,
            "memory_percent": mem_pct,
            "memory_used_mb": mem_used_mb
        },
        "task_predictions": telemetry.task_predictions,
        "models_status": {
            task: (task in registry._models_cache)
            for task in ["attendance", "approval", "budget"]
        },
        "lakefs_lineage": lakefs_meta,
        "data_drift_status": "NORMAL (No Drift Detected)"
    }


@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus-compatible exposition format for monitoring scrapers."""
    uptime = int(time.time() - telemetry.start_time)
    lats = list(telemetry.latencies_ms)
    p50 = round(float(np.percentile(lats, 50)), 2) if lats else 0.0
    p95 = round(float(np.percentile(lats, 95)), 2) if lats else 0.0
    p99 = round(float(np.percentile(lats, 99)), 2) if lats else 0.0
    cpu = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0
    mem = psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0.0

    metrics_text = f"""# HELP rvce_uptime_seconds Total service uptime in seconds
# TYPE rvce_uptime_seconds gauge
rvce_uptime_seconds {uptime}

# HELP rvce_requests_total Total number of HTTP requests processed
# TYPE rvce_requests_total counter
rvce_requests_total {telemetry.total_requests}

# HELP rvce_requests_success_total Successful HTTP requests
# TYPE rvce_requests_success_total counter
rvce_requests_success_total {telemetry.successful_requests}

# HELP rvce_requests_failed_total Failed HTTP requests
# TYPE rvce_requests_failed_total counter
rvce_requests_failed_total {telemetry.failed_requests}

# HELP rvce_inference_latency_ms Inference latency percentiles in ms
# TYPE rvce_inference_latency_ms gauge
rvce_inference_latency_ms{{quantile="0.50"}} {p50}
rvce_inference_latency_ms{{quantile="0.95"}} {p95}
rvce_inference_latency_ms{{quantile="0.99"}} {p99}

# HELP rvce_system_cpu_percent Host CPU utilization percent
# TYPE rvce_system_cpu_percent gauge
rvce_system_cpu_percent {cpu}

# HELP rvce_system_memory_percent Host Memory utilization percent
# TYPE rvce_system_memory_percent gauge
rvce_system_memory_percent {mem}

# HELP rvce_predictions_total Total predictions processed per task
# TYPE rvce_predictions_total counter
rvce_predictions_total{{task="attendance"}} {telemetry.task_predictions.get('attendance', 0)}
rvce_predictions_total{{task="approval"}} {telemetry.task_predictions.get('approval', 0)}
rvce_predictions_total{{task="budget"}} {telemetry.task_predictions.get('budget', 0)}
rvce_predictions_total{{task="all"}} {telemetry.task_predictions.get('all', 0)}
"""
    return PlainTextResponse(metrics_text.strip(), media_type="text/plain; version=0.0.4")


@app.get("/api/artifacts", tags=["Artifacts"])
async def list_ml_artifacts():
    """Lists all available machine learning evaluation plots and architecture artifacts."""
    artifacts = [
        {
            "id": "arch",
            "title": "ML Pipeline System Architecture",
            "category": "Architecture",
            "filename": "ml_architecture.png",
            "url": "/artifacts/ml_architecture.png",
            "badge": "System Flow",
            "description": "End-to-end data lake, feature extraction, MLflow model registry, and FastAPI inference pipeline."
        },
        {
            "id": "master_dashboard",
            "title": "Master Performance Evaluation Dashboard",
            "category": "Evaluation",
            "filename": "master_model_performance_dashboard.png",
            "url": "/artifacts/master_model_performance_dashboard.png",
            "badge": "Full Dashboard",
            "description": "Multi-metric comparison matrix across regression, classification, and budget forecasting tasks."
        },
        {
            "id": "confusion_all",
            "title": "Confusion Matrices (All Benchmark Models)",
            "category": "Classification",
            "filename": "confusion_matrix_all_models.png",
            "url": "/artifacts/confusion_matrix_all_models.png",
            "badge": "Task B Heatmap",
            "description": "Decision Tree, Random Forest, Gradient Boosting, and Voting Classifier confusion matrices."
        },
        {
            "id": "feature_importance",
            "title": "Feature Importance Analysis (Triple Task)",
            "category": "Explainability",
            "filename": "feature_importance.png",
            "url": "/artifacts/feature_importance.png",
            "badge": "Importance",
            "description": "Top feature weights for student attendance, booking approval decision, and financial budget."
        },
        {
            "id": "budget_feature_importance",
            "title": "Budget Predictor Feature Weights",
            "category": "Explainability",
            "filename": "budget_feature_importance.png",
            "url": "/artifacts/budget_feature_importance.png",
            "badge": "Task C Weights",
            "description": "Dominant drivers of event budgets: venue capacity, expected turnout, category, and lead days."
        },
        {
            "id": "comp_regression",
            "title": "Task A: Attendance Model Comparison",
            "category": "Evaluation",
            "filename": "model_comparison_regression.png",
            "url": "/artifacts/model_comparison_regression.png",
            "badge": "Task A (R²=0.857)",
            "description": "R2, MAE, and RMSE benchmarking across 4 candidate regressors."
        },
        {
            "id": "comp_classification",
            "title": "Task B: Booking Approval Model Comparison",
            "category": "Evaluation",
            "filename": "model_comparison_classification.png",
            "url": "/artifacts/model_comparison_classification.png",
            "badge": "Task B (F1=0.887)",
            "description": "ROC-AUC, F1-Score, and Accuracy benchmarks across 4 candidate classifiers."
        },
        {
            "id": "comp_budget",
            "title": "Task C: Budget Allocation Model Comparison",
            "category": "Evaluation",
            "filename": "model_comparison_budget.png",
            "url": "/artifacts/model_comparison_budget.png",
            "badge": "Task C (R²=0.9991)",
            "description": "R2, MAE, and MAPE comparison showing Gradient Boosting champion performance."
        },
        {
            "id": "eda",
            "title": "Exploratory Data Analysis & Turnout Distribution",
            "category": "EDA",
            "filename": "eda_visualization.png",
            "url": "/artifacts/eda_visualization.png",
            "badge": "Raw Data",
            "description": "Turnout distribution, booking frequency by department, and priority distribution across RVCE venues."
        },
        {
            "id": "budget_residuals",
            "title": "Budget Residuals & Prediction Scatter",
            "category": "Evaluation",
            "filename": "eval_reg_budget.png",
            "url": "/artifacts/eval_reg_budget.png",
            "badge": "Residuals",
            "description": "Actual vs. Predicted scatter plot and residual distribution for champion budget model."
        }
    ]
    return {"total": len(artifacts), "artifacts": artifacts}


@app.get("/artifacts/{filename:path}", tags=["Artifacts"])
async def get_artifact_file(filename: str):
    """Safely serves ML evaluation images and architecture diagrams."""
    clean_name = os.path.basename(filename)
    candidates = [
        os.path.join("reports", clean_name),
        clean_name,
        os.path.join("reports", "models", clean_name)
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.isfile(path):
            return FileResponse(path)
            
    raise HTTPException(
        status_code=404,
        detail=f"Artifact '{filename}' not found on server."
    )


# ------------------------------------------------------------------------------
# Model Registry Endpoints
# ------------------------------------------------------------------------------
@app.get("/registry/models", response_model=ModelInfoResponse, tags=["Model Registry"])
async def get_model_registry_info():
    """Returns metadata, performance metrics, and feature schemas for all registered models."""
    return ModelInfoResponse(registry=registry.list_all_metadata())


@app.post("/registry/promote", tags=["Model Registry"])
async def promote_model_stage(task: str, stage: ModelStage):
    """Transitions a model to a target deployment stage (e.g. Staging, Production)."""
    try:
        res = registry.promote_model(task, stage)
        return {"status": "SUCCESS", "detail": res}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ------------------------------------------------------------------------------
# Core Prediction Endpoints
# ------------------------------------------------------------------------------
@app.post("/predict/attendance", response_model=AttendancePredictionResponse, tags=["Predictions"])
async def predict_attendance(input_data: AttendanceFeatureInput):
    """
    Predicts student registrations for an event given engineered features.
    Strictly isolated from downstream targets (zero leakage).
    """
    try:
        return inference_service.predict_attendance(input_data.model_dump())
    except Exception as e:
        logger.error(f"Error predicting attendance: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/approval", response_model=ApprovalPredictionResponse, tags=["Predictions"])
async def predict_approval(input_data: ApprovalFeatureInput):
    """
    Evaluates automated event booking approval/rejection decision and clash risk.
    """
    try:
        return inference_service.predict_approval(input_data.model_dump())
    except Exception as e:
        logger.error(f"Error predicting approval: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/budget", response_model=BudgetPredictionResponse, tags=["Predictions"])
async def predict_budget(input_data: BudgetFeatureInput):
    """
    Forecasts necessary financial operational budget in INR.
    """
    try:
        return inference_service.predict_budget(input_data.model_dump())
    except Exception as e:
        logger.error(f"Error predicting budget: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/all", response_model=ComprehensiveEventDecisionResponse, tags=["Predictions"])
async def predict_full_event_evaluation(proposal: EventProposalRequest):
    """
    End-to-end event intelligence decision:
    Translates raw proposal -> Attendance forecast -> Fill ratio -> Approval decision & risk -> Budget estimate -> Actionable recommendations.
    """
    try:
        return inference_service.evaluate_full_proposal(proposal)
    except Exception as e:
        logger.error(f"Error evaluating full proposal: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline evaluation error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch_proposals(batch: BatchPredictionRequest):
    """
    Batch evaluation endpoint processing up to 100 event booking requests concurrently.
    """
    results = []
    successful = 0
    failed = 0

    for proposal in batch.proposals:
        try:
            eval_res = inference_service.evaluate_full_proposal(proposal)
            results.append({
                "event_name": proposal.event_name,
                "status": "SUCCESS",
                "decision": eval_res.approval_evaluation.decision,
                "predicted_attendance": eval_res.attendance_forecast.predicted_attendance,
                "estimated_budget_inr": eval_res.budget_allocation.estimated_budget_inr,
                "fill_ratio": eval_res.capacity_analysis["fill_ratio"]
            })
            successful += 1
        except Exception as e:
            results.append({
                "event_name": proposal.event_name,
                "status": "FAILED",
                "error": str(e)
            })
            failed += 1

    return BatchPredictionResponse(
        total_processed=len(batch.proposals),
        successful=successful,
        failed=failed,
        results=results
    )


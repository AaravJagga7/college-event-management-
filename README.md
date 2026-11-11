# 🏛️ RVCE 8th Mile Event Intelligence — MLOps Platform

An end-to-end, reproducible MLOps system for intelligent campus event scheduling, venue allocation, capacity optimisation, and budget forecasting for **RV College of Engineering (RVCE)**'s flagship annual festival: **8th Mile (Ashtrang / Kannada Habba)**.

> **Production API:** https://rvce-event-intelligence.vercel.app  
> **DagsHub:** https://dagshub.com/AaravJagga7/college-event-management-  
> **GitHub:** https://github.com/AaravJagga7/college-event-management-

---

## 🧠 Three Prediction Tasks

| Task | Type | Target | Champion Model |
|---|---|---|---|
| **A — Student Turnout** | Regression | `TARGET_registers` | GradientBoostingRegressor (R²=0.858) |
| **B — Booking Approval** | Classification | `TARGET_accept` | RandomForestClassifier (ROC-AUC=0.766) |
| **C — Budget Allocation** | Regression | `TARGET_budget` | GradientBoostingRegressor (R²=0.963) |

---

## 🗂️ Project Structure

```
college-event-management/
├── app/
│   ├── main.py                  # FastAPI inference service (Vercel)
│   └── ui.py                    # Gradient UI dashboard
├── airflow/
│   └── dags/
│       └── rvce_event_intelligence_mlops_dag.py  # 10-task MLOps DAG
├── tests/
│   ├── test_api.py              # 15 API endpoint tests
│   ├── test_model_performance.py
│   └── test_model_risk_quality.py
├── reports/
│   ├── models/                  # Trained .joblib model artifacts
│   ├── champion_metrics.json    # Benchmark metrics store
│   └── *.png                    # Evaluation plots & dashboards
├── train_and_track_wandb.py     # W&B + DagsHub experiment tracking
├── start_airflow.sh             # One-command Airflow launcher
├── requirements.txt
└── rvce_8th_mile_data.csv       # 1,500-row RVCE festival dataset
```

---

## 🚀 Stack

| Layer | Tool |
|---|---|
| **Inference API** | FastAPI + Uvicorn |
| **Frontend UI** | Gradient HTML dashboard (Vercel) |
| **Model Registry** | Kubeflow Model Registry (KFMR v1alpha3) |
| **Pipeline Orchestration** | Apache Airflow 2.8 (`@daily` DAG) |
| **Experiment Tracking** | Weights & Biases (W&B) |
| **Data / Model Versioning** | DagsHub + lakeFS |
| **Testing** | pytest (26 tests, 100% pass) |
| **Deployment** | Vercel (serverless) |

---

## ⚡ Quick Start

### 1 — Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Run the FastAPI service locally
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
# → http://127.0.0.1:8000/ui
```

### 3 — Launch Airflow (pipeline orchestration)
```bash
./start_airflow.sh
# → http://127.0.0.1:8081  (admin / admin)
```

### 4 — Run all tests
```bash
pytest -v
# → 26 passed
```

---

## 🔄 Airflow MLOps Pipeline (`rvce_event_intelligence_mlops_pipeline`)

```
validate_ingested_data
        │
check_features_and_drift
        │
   ┌────┴────────────────────┐
train_attendance   train_approval   train_budget
   └────┬────────────────────┘
evaluate_quality_gates  ← ShortCircuit on failure
        │
promote_model_registry
        │
   ┌────┴──────────────────┐
sync_wandb_and_dagshub   verify_production_sla
   └────┬──────────────────┘
pipeline_completion_summary
```

Run a one-off test:
```bash
export AIRFLOW_HOME="$(pwd)/airflow"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export WANDB_MODE=offline
.venv/bin/airflow dags test rvce_event_intelligence_mlops_pipeline $(date +%Y-%m-%d)
```

---

## 🌐 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health + models loaded |
| `GET` | `/ui` | Gradient dashboard |
| `GET` | `/registry/models` | Registered model metadata & lineage |
| `GET` | `/registry/kubeflow` | Full Kubeflow Model Registry (KFMR) specification |
| `POST` | `/registry/promote` | Promote model stage (Dev/Staging/Prod/Archived) |
| `POST` | `/predict/attendance` | Turnout regression |
| `POST` | `/predict/approval` | Booking approval classification |
| `POST` | `/predict/budget` | Budget regression (INR) |
| `POST` | `/predict/all` | All 3 predictions in one call |

---

## 📈 Model Quality Gates

| Task | Metric | Threshold |
|---|---|---|
| Attendance Regressor | R² | ≥ 0.70 |
| Attendance Regressor | MAE | ≤ 95 registrations |
| Approval Classifier | ROC-AUC | ≥ 0.75 |
| Approval Classifier | F1 | ≥ 0.80 |
| Budget Regressor | R² | ≥ 0.90 |
| Budget Regressor | MAPE | ≤ 15 % |

---

## 🧪 Test Coverage

```
tests/test_api.py               16 tests  — health, predict/all, Kubeflow registry, edge cases
tests/test_model_performance.py  3 tests  — R², MAE, ROC-AUC thresholds
tests/test_model_risk_quality.py 8 tests  — quality gates, risk constraints
─────────────────────────────────────────
                                27 passed
```

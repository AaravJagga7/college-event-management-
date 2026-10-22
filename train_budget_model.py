"""
================================================================================
Task C: Event Budget Prediction Pipeline (MLflow + Neptune.ai + lakeFS)
================================================================================
Benchmarks 4 competitive regression models for forecasting event budgets:
  1. Decision Tree Regressor (Interpretable baseline)
  2. Random Forest Regressor (Bagging ensemble)
  3. Gradient Boosting Regressor (Boosting ensemble)
  4. Voting Regressor (Weighted metamodel combining DT, RF, and GB)

Outputs:
  - reports/budget_comparison.csv
  - reports/model_comparison_budget.png
  - reports/eval_reg_budget.png
  - reports/budget_feature_importance.png
  - reports/models/champion_budget_model.joblib
  - Updates reports/champion_metrics.json
================================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
)

# Optional MLflow
try:
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Optional Neptune.ai
try:
    import neptune
    from neptune.types import File
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False

# Optional lakeFS
try:
    from lakefs_versioning import LakeFSManager
    LAKEFS_AVAILABLE = True
except ImportError:
    LAKEFS_AVAILABLE = False

# ------------------------------------------------------------------------------
# 0. CONFIGURATION & ENVIRONMENT
# ------------------------------------------------------------------------------
DATA_PATH = "features_ready.csv"
REPORTS_DIR = "reports"
MODELS_DIR = os.path.join(REPORTS_DIR, "models")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# lakeFS metadata
lakefs_commit_id = "uncommitted"
lakefs_branch = "main"
if LAKEFS_AVAILABLE:
    try:
        mgr = LakeFSManager()
        commit = mgr.get_latest_commit()
        lakefs_commit_id = commit.get("id", "uncommitted")
        lakefs_branch = mgr.branch
    except Exception as e:
        print(f"⚠️ lakeFS context lookup fallback: {e}")

print("=" * 75)
print("💰 TASK C: COLLEGE EVENT BUDGET FORECASTING PIPELINE")
print("=" * 75)
print(f"  • lakeFS Commit: {lakefs_commit_id} (Branch: {lakefs_branch})")

# MLflow Setup
DB_PATH = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "college-event-models"
if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri(DB_PATH)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow_client = MlflowClient()
    print(f"  • MLflow Tracking URI: {DB_PATH}")

# Neptune Setup
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT", "workspace/college-event-ml")
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN", "")
NEPTUNE_MODE = "async" if NEPTUNE_API_TOKEN else "offline"
print(f"  • Neptune.ai Mode:     [{NEPTUNE_MODE.upper()}]")
print("=" * 75 + "\n")

# ------------------------------------------------------------------------------
# 1. DATA PREPARATION & LEAKAGE FIREWALL
# ------------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} records from {DATA_PATH}.")

if "TARGET_budget" not in df.columns:
    raise ValueError("TARGET_budget column missing in features_ready.csv. Run eda_and_feature_extraction.py first.")

# Strict leakage prevention:
# Forbidden for budget forecasting:
# - TARGET_budget: Target variable itself
# - TARGET_accept: Approval decision (post-evaluation administrative outcome)
# - fill_ratio: Computed from attendance / room_capacity
# Note: Expected room capacity, duration, category, importance level, and attendance
# represent the pre-event requisition parameters.
LEAKY_COLS = ["TARGET_budget", "TARGET_accept", "fill_ratio"]
feature_cols = [c for c in df.columns if c not in LEAKY_COLS]

print(f"Predictor Features ({len(feature_cols)} total, zero target leakage):")
print(f"  {', '.join(feature_cols)}\n")

X = df[feature_cols]
y = df["TARGET_budget"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"Target Budget Summary -> Mean: ₹{y.mean():,.2f}, Median: ₹{y.median():,.2f}, Min: ₹{y.min():,.2f}, Max: ₹{y.max():,.2f}\n")

# ------------------------------------------------------------------------------
# 2. DEFINE ESTIMATOR ARCHITECTURES
# ------------------------------------------------------------------------------
dt_model = DecisionTreeRegressor(
    max_depth=6,
    min_samples_leaf=5,
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=8,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

gb_model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

voting_model = VotingRegressor(
    estimators=[
        ("dt", dt_model),
        ("rf", rf_model),
        ("gb", gb_model)
    ],
    weights=[1, 2, 2]
)

models = {
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Voting Regressor": voting_model
}

# ------------------------------------------------------------------------------
# 3. MODEL TRAINING, 5-FOLD CV & BENCHMARKING
# ------------------------------------------------------------------------------
results = []
trained_models = {}
predictions = {}

print("-" * 75)
print(f"{'Model':20s} | {'Test R²':8s} | {'Test MAE':12s} | {'Test RMSE':12s} | {'Test MAPE':9s} | {'CV R² (Mean±Std)':16s}")
print("-" * 75)

for name, model in models.items():
    # 5-fold cross validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    
    # Fit model on training set
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    tr_r2 = r2_score(y_train, y_pred_train)
    te_r2 = r2_score(y_test, y_pred_test)
    te_mae = mean_absolute_error(y_test, y_pred_test)
    te_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    te_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100.0
    
    trained_models[name] = model
    predictions[name] = y_pred_test
    
    print(f"{name:20s} | {te_r2:8.4f} | ₹{te_mae:10,.2f} | ₹{te_rmse:10,.2f} | {te_mape:8.2f}% | {cv_mean:.3f} ± {cv_std:.3f}")
    
    # Log to MLflow if available
    run_id = None
    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name=f"Budget_{name.replace(' ', '_')}") as run:
                run_id = run.info.run_id
                mlflow.set_tag("task", "budget_regression")
                mlflow.set_tag("target", "TARGET_budget")
                mlflow.set_tag("model_type", name)
                mlflow.set_tag("lakefs_commit", lakefs_commit_id)
                
                if hasattr(model, "get_params"):
                    clean_params = {k: str(v) for k, v in model.get_params().items() if not k.startswith("estimators") and len(str(v)) < 250}
                    mlflow.log_params(clean_params)
                    
                mlflow.log_metrics({
                    "train_r2": tr_r2,
                    "test_r2": te_r2,
                    "test_mae": te_mae,
                    "test_rmse": te_rmse,
                    "test_mape": te_mape,
                    "cv_r2_mean": cv_mean,
                    "cv_r2_std": cv_std
                })
                
                mlflow.sklearn.log_model(model, "budget_model")
        except Exception as e:
            print(f"⚠️ MLflow logging notice: {e}")
            
    # Log to Neptune if available
    if NEPTUNE_AVAILABLE:
        try:
            nep_run = neptune.init_run(
                project=NEPTUNE_PROJECT,
                api_token=NEPTUNE_API_TOKEN if NEPTUNE_API_TOKEN else None,
                mode=NEPTUNE_MODE,
                name=f"Budget_{name.replace(' ', '_')}",
                tags=["budget_regression", name, f"lakefs:{lakefs_commit_id}"]
            )
            nep_run["metrics/test_r2"] = te_r2
            nep_run["metrics/test_mae"] = te_mae
            nep_run["metrics/test_rmse"] = te_rmse
            nep_run["metrics/test_mape"] = te_mape
            nep_run["metrics/cv_r2_mean"] = cv_mean
            nep_run["metrics/cv_r2_std"] = cv_std
            nep_run.stop()
        except Exception as e:
            pass

    results.append({
        "Model": name,
        "Test R²": round(te_r2, 4),
        "Test MAE": round(te_mae, 2),
        "Test RMSE": round(te_rmse, 2),
        "Test MAPE (%)": round(te_mape, 2),
        "Train R²": round(tr_r2, 4),
        "CV R² (Mean)": round(cv_mean, 4),
        "CV R² (Std)": round(cv_std, 4),
        "MLflow Run ID": run_id or "local"
    })

print("-" * 75 + "\n")
res_df = pd.DataFrame(results)
res_df.to_csv(f"{REPORTS_DIR}/budget_comparison.csv", index=False)
print(f"✅ Saved comparison table: {REPORTS_DIR}/budget_comparison.csv")

# ------------------------------------------------------------------------------
# 4. CHAMPION SELECTION & SERIALIZATION
# ------------------------------------------------------------------------------
# Designate champion based on lowest Test MAE
champion_row = res_df.sort_values(by="Test MAE").iloc[0]
champion_name = champion_row["Model"]
champion_model = trained_models[champion_name]

print(f"🏆 CHAMPION MODEL SELECTED: {champion_name}")
print(f"   Test R²:   {champion_row['Test R²']}")
print(f"   Test MAE:  ₹{champion_row['Test MAE']:,.2f}")
print(f"   Test RMSE: ₹{champion_row['Test RMSE']:,.2f}")
print(f"   CV R²:     {champion_row['CV R² (Mean)']}")

# Save Champion Model Artifact (.joblib)
model_artifact_path = os.path.join(MODELS_DIR, "champion_budget_model.joblib")
joblib.dump(champion_model, model_artifact_path)
print(f"💾 Saved serialized champion artifact: {model_artifact_path}")

# Register in MLflow Model Registry if available
if MLFLOW_AVAILABLE and champion_row["MLflow Run ID"] != "local":
    try:
        model_uri = f"runs:/{champion_row['MLflow Run ID']}/budget_model"
        reg_name = "college-budget-champion"
        mv = mlflow.register_model(model_uri=model_uri, name=reg_name)
        mlflow_client.set_registered_model_alias(name=reg_name, alias="champion", version=mv.version)
        print(f"🎖️ Registered '{reg_name}' version {mv.version} with alias '@champion' in MLflow Registry.")
    except Exception as e:
        print(f"⚠️ MLflow registry notice: {e}")

# ------------------------------------------------------------------------------
# 5. VISUALIZATIONS & REPORT GENERATION
# ------------------------------------------------------------------------------
print("\n📊 Generating production evaluation charts...")

# 1. Model Comparison Chart (3-panel)
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
palette = ["#4e79a7", "#f28e2c", "#2ca02c", "#e15759"]

# A: Test R2
sns.barplot(x="Model", y="Test R²", data=res_df, ax=axes[0], palette=palette)
axes[0].set_title("A. Test R² (Higher is Better)", fontweight="bold")
axes[0].set_ylim(0, 1.1)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")

# B: Test MAE
sns.barplot(x="Model", y="Test MAE", data=res_df, ax=axes[1], palette=palette)
axes[1].set_title("B. Test MAE in ₹ (Lower is Better)", fontweight="bold")
axes[1].set_ylim(0, max(res_df["Test MAE"]) * 1.25)
for p in axes[1].patches:
    axes[1].annotate(f"₹{p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")

# C: 5-Fold Cross Validation R2
sns.barplot(x="Model", y="CV R² (Mean)", data=res_df, ax=axes[2], palette=palette)
axes[2].set_title("C. 5-Fold Cross-Validation R²", fontweight="bold")
axes[2].set_ylim(0, 1.1)
for p in axes[2].patches:
    axes[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=10, fontweight="bold", xytext=(0, 4), textcoords="offset points")

for ax in axes:
    ax.tick_params(axis="x", rotation=15)

plt.suptitle("Model Benchmark Comparison: College Event Budget Forecasting (Task C)", fontsize=14, fontweight="bold", y=1.03)
fig.tight_layout()
cmp_plot_file = f"{REPORTS_DIR}/model_comparison_budget.png"
fig.savefig(cmp_plot_file, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  • {cmp_plot_file}")

# 2. Champion Model Residual & Parity Plot (2-panel)
y_best_pred = predictions[champion_name]
residuals = y_test - y_best_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].scatter(y_test, y_best_pred, alpha=0.6, color="#2ca02c", edgecolors="k", s=35)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Parity Line (y = x)")
axes[0].set_title(f"{champion_name}: Actual vs Predicted Budget", fontweight="bold")
axes[0].set_xlabel("Actual Budget (₹)")
axes[0].set_ylabel("Predicted Budget (₹)")
axes[0].legend()
axes[0].grid(True, linestyle=":", alpha=0.6)

sns.histplot(residuals, kde=True, ax=axes[1], color="#1f77b4", bins=25)
axes[1].axvline(0, color="red", linestyle="--", lw=2, label="Zero Error")
axes[1].set_title(f"{champion_name}: Residual Distribution (Mean Error: ₹{residuals.mean():.1f})", fontweight="bold")
axes[1].set_xlabel("Prediction Error (₹)")
axes[1].set_ylabel("Count")
axes[1].legend()

fig.tight_layout()
eval_plot_file = f"{REPORTS_DIR}/eval_reg_budget.png"
fig.savefig(eval_plot_file, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  • {eval_plot_file}")

# 3. Budget Feature Importances Plot
if hasattr(champion_model, "feature_importances_"):
    feat_imp = pd.Series(champion_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12)
elif hasattr(champion_model, "estimators_"):
    # If voting regressor, average feature importances across base tree estimators
    imp_list = [est.feature_importances_ for _, est in champion_model.estimators if hasattr(est, "feature_importances_")]
    if imp_list:
        feat_imp = pd.Series(np.mean(imp_list, axis=0), index=feature_cols).sort_values(ascending=False).head(12)
    else:
        feat_imp = None
else:
    feat_imp = None

if feat_imp is not None:
    plt.figure(figsize=(10, 5.5))
    feat_imp.sort_values().plot(kind="barh", color="#2ca02c", edgecolor="black")
    plt.title(f"Top 12 Drivers: Event Budget Allocation ({champion_name})", fontweight="bold", fontsize=12)
    plt.xlabel("Gini Feature Importance")
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    feat_imp_file = f"{REPORTS_DIR}/budget_feature_importance.png"
    plt.savefig(feat_imp_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  • {feat_imp_file}")

# ------------------------------------------------------------------------------
# 6. UPDATE CHAMPION METRICS JSON
# ------------------------------------------------------------------------------
metrics_json_path = f"{REPORTS_DIR}/champion_metrics.json"
try:
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r") as f:
            current_metrics = json.load(f)
    else:
        current_metrics = {}
except Exception:
    current_metrics = {}

current_metrics["budget_regression"] = {
    "champion_model": champion_name,
    "test_mae": float(champion_row["Test MAE"]),
    "test_rmse": float(champion_row["Test RMSE"]),
    "test_r2": float(champion_row["Test R²"]),
    "test_mape": float(champion_row["Test MAPE (%)"]),
    "cv_r2": float(champion_row["CV R² (Mean)"]),
    "model_artifact": model_artifact_path
}

with open(metrics_json_path, "w") as f:
    json.dump(current_metrics, f, indent=2)
print(f"✅ Updated {metrics_json_path} with Task C champion metrics.")

print("\n" + "=" * 75)
print("🎉 Event Budget Prediction Pipeline Executed Successfully!")
print("=" * 75)

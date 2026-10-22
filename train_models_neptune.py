"""
================================================================================
Model Training, Comparison & Neptune.ai Integration Pipeline
================================================================================
Integrated with:
  - lakeFS: For immutable data lake commit hashes and dataset version tracking
  - Neptune.ai: For experiment tracking, hyperparameter logging, metric curves,
                evaluation visualizations, and model artifact management.

Models Evaluated:
  1. Decision Tree
  2. Random Forest
  3. Gradient Boosting
  4. Voting Ensemble (Ensemble Combining DT, RF, and GB)

Tasks:
  - Task A: Attendance Regression (TARGET_registers)
  - Task B: Room Booking Approval Classification (TARGET_accept)
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

import neptune
from neptune.types import File

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from lakefs_versioning import LakeFSManager

# ------------------------------------------------------------------------------
# 0. CONFIGURATION, DIRECTORIES & LAKEFS CONTEXT
# ------------------------------------------------------------------------------
DATA_PATH = "features_ready.csv"
REPORTS_DIR = "reports"
MODELS_DIR = os.path.join(REPORTS_DIR, "models")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Fetch latest lakeFS data version metadata
lakefs_mgr = LakeFSManager()
latest_commit = lakefs_mgr.get_latest_commit()
lakefs_commit_id = latest_commit.get("id", "uncommitted")
lakefs_features_uri = latest_commit.get("objects", {}).get(
    "processed/features_ready.csv", {}
).get("lakefs_uri", f"lakefs://{lakefs_mgr.repository}/{lakefs_mgr.branch}/processed/features_ready.csv")

print("=" * 70)
print("🌊 lakeFS DATA LAKE VERSION CONTEXT")
print(f"  • Repository:  {lakefs_mgr.repository}")
print(f"  • Branch:      {lakefs_mgr.branch}")
print(f"  • Commit Hash: {lakefs_commit_id}")
print(f"  • Feature URI: {lakefs_features_uri}")
print("=" * 70)

# Configure Neptune.ai
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT", "workspace/college-event-ml")
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN", "")
NEPTUNE_MODE = "async" if NEPTUNE_API_TOKEN else "offline"

print(f"\n🔱 Neptune.ai configured in [{NEPTUNE_MODE.upper()}] mode")
if NEPTUNE_MODE == "offline":
    print("  ℹ️ Tip: Set NEPTUNE_API_TOKEN & NEPTUNE_PROJECT to stream directly to app.neptune.ai")
    print("  Offline runs are stored in '.neptune/' and can be synchronized via `neptune sync`.\n")

# Load engineered feature dataset
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} features from {DATA_PATH}.")

# Define feature sets strictly preventing data leakage
LEAKY_COLS = ["TARGET_registers", "TARGET_accept", "fill_ratio", "TARGET_budget"]
feature_cols_reg = [c for c in df.columns if c not in LEAKY_COLS]
feature_cols_cls = [c for c in df.columns if c not in ["TARGET_registers", "TARGET_accept", "TARGET_budget"]]

# ------------------------------------------------------------------------------
# 1. TASK A: ATTENDANCE REGRESSION (TARGET_registers)
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("🚀 TASK A: STUDENT ATTENDANCE FORECASTING (REGRESSION)")
print("=" * 70)

X_reg = df[feature_cols_reg]
y_reg = df["TARGET_registers"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

# Individual estimators
dt_reg = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)
gb_reg = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)

# Voting Regressor
voting_reg = VotingRegressor(
    estimators=[
        ("dt", dt_reg),
        ("rf", rf_reg),
        ("gb", gb_reg)
    ],
    weights=[1, 2, 2]
)

reg_models = {
    "Decision Tree": dt_reg,
    "Random Forest": rf_reg,
    "Gradient Boosting": gb_reg,
    "Voting Regressor": voting_reg
}

reg_metrics_list = []

for name, model in reg_models.items():
    # Initialize Neptune Run
    run = neptune.init_run(
        project=NEPTUNE_PROJECT,
        api_token=NEPTUNE_API_TOKEN if NEPTUNE_API_TOKEN else None,
        mode=NEPTUNE_MODE,
        name=f"Reg_{name.replace(' ', '_')}",
        tags=["task:regression", "college-events", f"model:{name.lower().replace(' ', '_')}", f"lakefs:{lakefs_commit_id}"]
    )
    
    # 5-Fold Cross Validation
    cv_scores = cross_val_score(model, X_train_r, y_train_r, cv=5, scoring="r2")
    
    # Fit & Predict
    model.fit(X_train_r, y_train_r)
    y_pred_train = model.predict(X_train_r)
    y_pred_test = model.predict(X_test_r)
    
    # Metrics
    tr_r2 = float(r2_score(y_train_r, y_pred_train))
    te_r2 = float(r2_score(y_test_r, y_pred_test))
    te_mae = float(mean_absolute_error(y_test_r, y_pred_test))
    te_rmse = float(np.sqrt(mean_squared_error(y_test_r, y_pred_test)))
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    
    # Log lakeFS lineage & metadata to Neptune
    run["lakefs/repository"] = lakefs_mgr.repository
    run["lakefs/branch"] = lakefs_mgr.branch
    run["lakefs/commit_id"] = lakefs_commit_id
    run["lakefs/features_uri"] = lakefs_features_uri
    
    # Log Parameters
    if hasattr(model, "get_params"):
        clean_params = {k: str(v) for k, v in model.get_params().items() if not k.startswith("estimators") and len(str(v)) < 250}
        run["parameters"] = clean_params
        
    # Log Metrics
    run["metrics/train_r2"] = tr_r2
    run["metrics/test_r2"] = te_r2
    run["metrics/test_mae"] = te_mae
    run["metrics/test_rmse"] = te_rmse
    run["metrics/cv_r2_mean"] = cv_mean
    run["metrics/cv_r2_std"] = cv_std
    
    # Save & Log Evaluation Visualizations
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(y_test_r, y_pred_test, alpha=0.6, color="#2b5c8f", edgecolors="k", s=30)
    ax[0].plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], "r--", lw=2)
    ax[0].set_title(f"{name}: Actual vs Predicted")
    ax[0].set_xlabel("Actual Turnout")
    ax[0].set_ylabel("Predicted Turnout")
    
    residuals = y_test_r - y_pred_test
    sns.histplot(residuals, kde=True, ax=ax[1], color="#d95f02", bins=25)
    ax[1].set_title(f"{name}: Residuals (Error Distribution)")
    ax[1].set_xlabel("Residual (Actual - Predicted)")
    
    fig.tight_layout()
    plot_path = f"{REPORTS_DIR}/eval_reg_{name.lower().replace(' ', '_')}.png"
    fig.savefig(plot_path)
    run["evaluation/plots"].upload(plot_path)
    plt.close(fig)
    
    # Serialize and Log Model Artifact
    model_save_path = os.path.join(MODELS_DIR, f"reg_{name.lower().replace(' ', '_')}.joblib")
    joblib.dump(model, model_save_path)
    run["model/saved_model"].upload(model_save_path)
    
    run_id = run["sys/id"].fetch() if NEPTUNE_MODE != "offline" else f"offline_{name.lower().replace(' ', '_')}"
    run.stop()
    
    reg_metrics_list.append({
        "Model": name,
        "Neptune Run ID": run_id,
        "CV R² (Mean)": round(cv_mean, 3),
        "CV R² (Std)": round(cv_std, 3),
        "Test R²": round(te_r2, 3),
        "Test MAE": round(te_mae, 2),
        "Test RMSE": round(te_rmse, 2),
        "Train R²": round(tr_r2, 3)
    })
    
    print(f"  • {name:20s} | Test R²: {te_r2:.3f} | Test MAE: {te_mae:5.2f} | Test RMSE: {te_rmse:5.2f} | 5-Fold CV R²: {cv_mean:.3f}")

reg_summary_df = pd.DataFrame(reg_metrics_list)

# ------------------------------------------------------------------------------
# 2. TASK B: BOOKING APPROVAL CLASSIFICATION (TARGET_accept)
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("🚀 TASK B: ROOM BOOKING APPROVAL (CLASSIFICATION)")
print("=" * 70)

X_cls = df[feature_cols_cls]
y_cls = df["TARGET_accept"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.20, random_state=42, stratify=y_cls
)

dt_cls = DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, class_weight="balanced", random_state=42)
rf_cls = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
gb_cls = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)

voting_cls = VotingClassifier(
    estimators=[
        ("dt", dt_cls),
        ("rf", rf_cls),
        ("gb", gb_cls)
    ],
    voting="soft",
    weights=[1, 2, 2]
)

cls_models = {
    "Decision Tree": dt_cls,
    "Random Forest": rf_cls,
    "Gradient Boosting": gb_cls,
    "Voting Classifier": voting_cls
}

cls_metrics_list = []

for name, model in cls_models.items():
    run = neptune.init_run(
        project=NEPTUNE_PROJECT,
        api_token=NEPTUNE_API_TOKEN if NEPTUNE_API_TOKEN else None,
        mode=NEPTUNE_MODE,
        name=f"Cls_{name.replace(' ', '_')}",
        tags=["task:classification", "college-events", f"model:{name.lower().replace(' ', '_')}", f"lakefs:{lakefs_commit_id}"]
    )
    
    # 5-Fold Cross Validation
    cv_scores = cross_val_score(model, X_train_c, y_train_c, cv=5, scoring="roc_auc")
    
    # Fit & Predict
    model.fit(X_train_c, y_train_c)
    y_pred_train = model.predict(X_train_c)
    y_pred_test = model.predict(X_test_c)
    
    if hasattr(model, "predict_proba"):
        y_prob_test = model.predict_proba(X_test_c)[:, 1]
    else:
        y_prob_test = y_pred_test
        
    acc = float(accuracy_score(y_test_c, y_pred_test))
    prec = float(precision_score(y_test_c, y_pred_test, zero_division=0))
    rec = float(recall_score(y_test_c, y_pred_test, zero_division=0))
    f1 = float(f1_score(y_test_c, y_pred_test, zero_division=0))
    auc = float(roc_auc_score(y_test_c, y_prob_test))
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    
    # Log lakeFS lineage & metadata to Neptune
    run["lakefs/repository"] = lakefs_mgr.repository
    run["lakefs/branch"] = lakefs_mgr.branch
    run["lakefs/commit_id"] = lakefs_commit_id
    run["lakefs/features_uri"] = lakefs_features_uri
    
    if hasattr(model, "get_params"):
        clean_params = {k: str(v) for k, v in model.get_params().items() if not k.startswith("estimators") and len(str(v)) < 250}
        run["parameters"] = clean_params
        
    run["metrics/accuracy"] = acc
    run["metrics/precision"] = prec
    run["metrics/recall"] = rec
    run["metrics/f1_score"] = f1
    run["metrics/roc_auc"] = auc
    run["metrics/cv_auc_mean"] = cv_mean
    run["metrics/cv_auc_std"] = cv_std
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test_c, y_pred_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Reject (0)", "Accept (1)"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{name}: Confusion Matrix")
    
    cm_path = f"{REPORTS_DIR}/cm_cls_{name.lower().replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(cm_path)
    run["evaluation/confusion_matrix"].upload(cm_path)
    plt.close(fig)
    
    # Serialize Model
    model_save_path = os.path.join(MODELS_DIR, f"cls_{name.lower().replace(' ', '_')}.joblib")
    joblib.dump(model, model_save_path)
    run["model/saved_model"].upload(model_save_path)
    
    run_id = run["sys/id"].fetch() if NEPTUNE_MODE != "offline" else f"offline_{name.lower().replace(' ', '_')}"
    run.stop()
    
    cls_metrics_list.append({
        "Model": name,
        "Neptune Run ID": run_id,
        "ROC-AUC": round(auc, 3),
        "CV ROC-AUC (Mean)": round(cv_mean, 3),
        "CV ROC-AUC (Std)": round(cv_std, 3),
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1-Score": round(f1, 3)
    })
    
    print(f"  • {name:20s} | ROC-AUC: {auc:.3f} | F1: {f1:.3f} | Accuracy: {acc:.3f} | 5-Fold CV AUC: {cv_mean:.3f}")

cls_summary_df = pd.DataFrame(cls_metrics_list)

# ------------------------------------------------------------------------------
# 3. GENERATE COMPARISON CHARTS & CSV SUMMARIES
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("📊 GENERATING MODEL COMPARISON CHARTS & METRIC REPORTS")
print("=" * 70)

# Regression comparison chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=reg_summary_df, x="Model", y="Test R²", hue="Model", legend=False, ax=ax1, palette="crest")
ax1.set_title("Task A: Attendance Turnout - Test R² Comparison (Higher is Better)", fontsize=11, fontweight="bold")
ax1.set_ylim(0, 0.70)
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

sns.barplot(data=reg_summary_df, x="Model", y="Test MAE", hue="Model", legend=False, ax=ax2, palette="flare")
ax2.set_title("Task A: Attendance Turnout - Test MAE (Lower is Better)", fontsize=11, fontweight="bold")
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
fig.tight_layout()
reg_plot_file = f"{REPORTS_DIR}/model_comparison_regression.png"
fig.savefig(reg_plot_file, dpi=120, bbox_inches="tight")
plt.close(fig)

# Classification comparison chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=cls_summary_df, x="Model", y="ROC-AUC", hue="Model", legend=False, ax=ax1, palette="viridis")
ax1.set_title("Task B: Booking Approval - ROC-AUC Score (Higher is Better)", fontsize=11, fontweight="bold")
ax1.set_ylim(0.70, 1.0)
for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

sns.barplot(data=cls_summary_df, x="Model", y="F1-Score", hue="Model", legend=False, ax=ax2, palette="mako")
ax2.set_title("Task B: Booking Approval - F1-Score (Higher is Better)", fontsize=11, fontweight="bold")
ax2.set_ylim(0.70, 1.0)
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
fig.tight_layout()
cls_plot_file = f"{REPORTS_DIR}/model_comparison_classification.png"
fig.savefig(cls_plot_file, dpi=120, bbox_inches="tight")
plt.close(fig)

# Save Comparison CSVs
reg_summary_df.to_csv(f"{REPORTS_DIR}/regression_comparison.csv", index=False)
cls_summary_df.to_csv(f"{REPORTS_DIR}/classification_comparison.csv", index=False)

# ------------------------------------------------------------------------------
# 4. CHAMPION MODEL SELECTION & NEPTUNE REGISTRY TAGGING
# ------------------------------------------------------------------------------
champion_reg = reg_summary_df.sort_values(by="Test MAE").iloc[0]
champion_cls = cls_summary_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]

print("\n" + "=" * 70)
print("🏆 CHAMPION MODEL SELECTION & NEPTUNE METADATA TAGGING")
print("=" * 70)
print(f"Regression Champion:     {champion_reg['Model']} (Test MAE: {champion_reg['Test MAE']}, Test R²: {champion_reg['Test R²']})")
print(f"Classification Champion: {champion_cls['Model']} (Test ROC-AUC: {champion_cls['ROC-AUC']}, F1: {champion_cls['F1-Score']})")

# Log Champion Summary Run to Neptune
champ_run = neptune.init_run(
    project=NEPTUNE_PROJECT,
    api_token=NEPTUNE_API_TOKEN if NEPTUNE_API_TOKEN else None,
    mode=NEPTUNE_MODE,
    name="Champion_Production_Summary",
    tags=["champion", "stage:production", "summary", f"lakefs:{lakefs_commit_id}"]
)

champ_run["lakefs/commit_id"] = lakefs_commit_id
champ_run["lakefs/branch"] = lakefs_mgr.branch
champ_run["champions/regression/model_name"] = champion_reg["Model"]
champ_run["champions/regression/test_mae"] = float(champion_reg["Test MAE"])
champ_run["champions/regression/test_r2"] = float(champion_reg["Test R²"])
champ_run["champions/classification/model_name"] = champion_cls["Model"]
champ_run["champions/classification/test_roc_auc"] = float(champion_cls["ROC-AUC"])
champ_run["champions/classification/test_f1"] = float(champion_cls["F1-Score"])

champ_run["reports/regression_plot"].upload(reg_plot_file)
champ_run["reports/classification_plot"].upload(cls_plot_file)
champ_run["reports/regression_table"].upload(f"{REPORTS_DIR}/regression_comparison.csv")
champ_run["reports/classification_table"].upload(f"{REPORTS_DIR}/classification_comparison.csv")
champ_run.stop()

# Overall metrics json
metrics_json_path = f"{REPORTS_DIR}/champion_metrics.json"
try:
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r") as f:
            metrics_out = json.load(f)
    else:
        metrics_out = {}
except Exception:
    metrics_out = {}

metrics_out["lakefs_metadata"] = {
    "repository": lakefs_mgr.repository,
    "branch": lakefs_mgr.branch,
    "commit_id": lakefs_commit_id,
    "features_lakefs_uri": lakefs_features_uri
}
metrics_out["attendance_regression"] = {
    "champion_model": champion_reg["Model"],
    "test_mae": float(champion_reg["Test MAE"]),
    "test_r2": float(champion_reg["Test R²"]),
    "cv_r2": float(champion_reg["CV R² (Mean)"])
}
metrics_out["booking_approval"] = {
    "champion_model": champion_cls["Model"],
    "test_roc_auc": float(champion_cls["ROC-AUC"]),
    "test_f1": float(champion_cls["F1-Score"]),
    "test_accuracy": float(champion_cls["Accuracy"])
}

with open(metrics_json_path, "w") as f:
    json.dump(metrics_out, f, indent=2)

print("\n🎉 Training, Comparison and Neptune.ai Tracking Completed Successfully!")
print(f"Saved artifacts:")
print(f"  • {reg_plot_file}")
print(f"  • {cls_plot_file}")
print(f"  • {metrics_json_path}")
print(f"  • {REPORTS_DIR}/regression_comparison.csv")
print(f"  • {REPORTS_DIR}/classification_comparison.csv")
print("\nTo view runs on Neptune.ai or sync offline runs:")
print("  neptune sync")

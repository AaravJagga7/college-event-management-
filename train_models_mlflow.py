"""
================================================================================
Model Training, Comparison & MLflow Integration Pipeline
================================================================================
Models Compared:
  1. Decision Tree
  2. Random Forest
  3. Gradient Boosting
  4. Voting Ensemble (Combining Decision Tree, Random Forest, Gradient Boosting)
  
Tasks:
  - Task A: Attendance Regression (TARGET_registers)
  - Task B: Approval Classification (TARGET_accept)
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

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ------------------------------------------------------------------------------
# 0. CONFIGURATION & MLFLOW SETUP
# ------------------------------------------------------------------------------
DATA_PATH = "features_ready.csv"
DB_PATH = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "college-event-models"
REPORTS_DIR = "reports"
MODELS_DIR = os.path.join(REPORTS_DIR, "models")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

mlflow.set_tracking_uri(DB_PATH)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

print(f"✅ MLflow configured with backend store: {DB_PATH}")
print(f"📁 Experiment name: '{EXPERIMENT_NAME}'\n")

# Load engineered feature dataset
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} features from {DATA_PATH}.")

# Define feature sets strictly preventing data leakage
LEAKY_COLS = ["TARGET_registers", "TARGET_accept", "fill_ratio", "TARGET_budget"]
feature_cols_reg = [c for c in df.columns if c not in LEAKY_COLS]
feature_cols_cls = [c for c in df.columns if c not in ["TARGET_registers", "TARGET_accept", "TARGET_budget"]]
feature_cols_bgt = [c for c in df.columns if c not in ["TARGET_budget", "TARGET_accept", "fill_ratio"]]

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

# Define individual estimators
dt_reg = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)
gb_reg = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)

# Define Voting Regressor
voting_reg = VotingRegressor(
    estimators=[
        ("dt", dt_reg),
        ("rf", rf_reg),
        ("gb", gb_reg)
    ],
    weights=[1, 2, 2] # give slightly higher weight to ensemble trees
)

reg_models = {
    "Decision Tree": dt_reg,
    "Random Forest": rf_reg,
    "Gradient Boosting": gb_reg,
    "Voting Regressor": voting_reg
}

reg_metrics_list = []

for name, model in reg_models.items():
    run_name = f"Reg_{name.replace(' ', '_')}"
    with mlflow.start_run(run_name=run_name) as run:
        # Cross validation
        cv_scores = cross_val_score(model, X_train_r, y_train_r, cv=5, scoring="r2")
        
        # Fit and predict
        model.fit(X_train_r, y_train_r)
        y_pred_train = model.predict(X_train_r)
        y_pred_test = model.predict(X_test_r)
        
        # Metrics
        tr_r2 = r2_score(y_train_r, y_pred_train)
        te_r2 = r2_score(y_test_r, y_pred_test)
        te_mae = mean_absolute_error(y_test_r, y_pred_test)
        te_rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_test))
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        
        # Log params & tags
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("target", "TARGET_registers")
        mlflow.set_tag("model_type", name)
        
        if hasattr(model, "get_params"):
            clean_params = {k: str(v) for k, v in model.get_params().items() if not k.startswith("estimators") and len(str(v)) < 250}
            mlflow.log_params(clean_params)
            
        # Log metrics
        metrics = {
            "train_r2": tr_r2,
            "test_r2": te_r2,
            "test_mae": te_mae,
            "test_rmse": te_rmse,
            "cv_r2_mean": cv_mean,
            "cv_r2_std": cv_std
        }
        mlflow.log_metrics(metrics)
        
        # Residual and prediction plot artifact
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
        mlflow.log_figure(fig, f"eval_plots_{name.replace(' ', '_')}.png")
        plt.close(fig)
        
        # Log model signature
        signature = infer_signature(X_train_r, y_pred_train)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train_r.head(2)
        )
        
        reg_metrics_list.append({
            "Model": name,
            "Run ID": run.info.run_id,
            "Model URI": model_info.model_uri,
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
    run_name = f"Cls_{name.replace(' ', '_')}"
    with mlflow.start_run(run_name=run_name) as run:
        # Cross validation ROC-AUC
        cv_scores = cross_val_score(model, X_train_c, y_train_c, cv=5, scoring="roc_auc")
        
        # Fit and predict
        model.fit(X_train_c, y_train_c)
        y_pred = model.predict(X_test_c)
        y_prob = model.predict_proba(X_test_c)[:, 1]
        
        acc = accuracy_score(y_test_c, y_pred)
        prec = precision_score(y_test_c, y_pred)
        rec = recall_score(y_test_c, y_pred)
        f1 = f1_score(y_test_c, y_pred)
        roc_auc = roc_auc_score(y_test_c, y_prob)
        cv_auc_mean = float(cv_scores.mean())
        cv_auc_std = float(cv_scores.std())
        
        mlflow.set_tag("task", "classification")
        mlflow.set_tag("target", "TARGET_accept")
        mlflow.set_tag("model_type", name)
        
        if hasattr(model, "get_params"):
            clean_params = {k: str(v) for k, v in model.get_params().items() if not k.startswith("estimators") and len(str(v)) < 250}
            mlflow.log_params(clean_params)
            
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "cv_roc_auc_mean": cv_auc_mean,
            "cv_roc_auc_std": cv_auc_std
        }
        mlflow.log_metrics(metrics)
        
        # Confusion matrix artifact
        cm = confusion_matrix(y_test_c, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["Reject (0)", "Accept (1)"]).plot(ax=ax, cmap="Blues")
        ax.set_title(f"{name}: Confusion Matrix")
        fig.tight_layout()
        plot_path = f"{REPORTS_DIR}/cm_{name.lower().replace(' ', '_')}.png"
        fig.savefig(plot_path)
        mlflow.log_figure(fig, f"confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close(fig)
        
        signature = infer_signature(X_train_c, y_pred)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train_c.head(2)
        )
        
        cls_metrics_list.append({
            "Model": name,
            "Run ID": run.info.run_id,
            "Model URI": model_info.model_uri,
            "Accuracy": round(acc, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1-Score": round(f1, 3),
            "ROC-AUC": round(roc_auc, 3),
            "CV ROC-AUC": round(cv_auc_mean, 3)
        })
        
        print(f"  • {name:20s} | ROC-AUC: {roc_auc:.3f} | F1: {f1:.3f} | Accuracy: {acc:.3f} | 5-Fold CV AUC: {cv_auc_mean:.3f}")

cls_summary_df = pd.DataFrame(cls_metrics_list)

# ------------------------------------------------------------------------------
# 3. TASK C: EVENT BUDGET FORECASTING (TARGET_budget - REGRESSION)
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("🚀 TASK C: EVENT BUDGET ALLOCATION FORECASTING (REGRESSION)")
print("=" * 70)

X_bgt = df[feature_cols_bgt]
y_bgt = df["TARGET_budget"]

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_bgt, y_bgt, test_size=0.20, random_state=42
)

# Define individual estimators for budget
dt_bgt = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42)
rf_bgt = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)
gb_bgt = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)

# Define Voting Regressor for budget
voting_bgt = VotingRegressor(
    estimators=[
        ("dt", dt_bgt),
        ("rf", rf_bgt),
        ("gb", gb_bgt)
    ],
    weights=[1, 2, 2]
)

bgt_models = {
    "Decision Tree": dt_bgt,
    "Random Forest": rf_bgt,
    "Gradient Boosting": gb_bgt,
    "Voting Regressor": voting_bgt
}

bgt_metrics_list = []
trained_bgt_models = {}

for name, model in bgt_models.items():
    run_name = f"Budget_{name.replace(' ', '_')}"
    with mlflow.start_run(run_name=run_name) as run:
        # Cross validation
        cv_scores = cross_val_score(model, X_train_b, y_train_b, cv=5, scoring="r2")
        
        # Fit and predict
        model.fit(X_train_b, y_train_b)
        y_pred_train = model.predict(X_train_b)
        y_pred_test = model.predict(X_test_b)
        
        # Metrics
        tr_r2 = r2_score(y_train_b, y_pred_train)
        te_r2 = r2_score(y_test_b, y_pred_test)
        te_mae = mean_absolute_error(y_test_b, y_pred_test)
        te_rmse = np.sqrt(mean_squared_error(y_test_b, y_pred_test))
        te_mape = mean_absolute_percentage_error(y_test_b, y_pred_test) * 100.0
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        
        trained_bgt_models[name] = model
        
        # Log params & tags
        mlflow.set_tag("task", "budget_regression")
        mlflow.set_tag("target", "TARGET_budget")
        mlflow.set_tag("model_type", name)
        
        if hasattr(model, "get_params"):
            clean_params = {k: str(v) for k, v in model.get_params().items() if not k.startswith("estimators") and len(str(v)) < 250}
            mlflow.log_params(clean_params)
            
        # Log metrics
        metrics = {
            "train_r2": tr_r2,
            "test_r2": te_r2,
            "test_mae": te_mae,
            "test_rmse": te_rmse,
            "test_mape": te_mape,
            "cv_r2_mean": cv_mean,
            "cv_r2_std": cv_std
        }
        mlflow.log_metrics(metrics)
        
        # Residual and prediction plot artifact
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].scatter(y_test_b, y_pred_test, alpha=0.6, color="#2ca02c", edgecolors="k", s=30)
        ax[0].plot([y_test_b.min(), y_test_b.max()], [y_test_b.min(), y_test_b.max()], "r--", lw=2)
        ax[0].set_title(f"{name}: Actual vs Predicted Budget")
        ax[0].set_xlabel("Actual Budget (₹)")
        ax[0].set_ylabel("Predicted Budget (₹)")
        
        residuals = y_test_b - y_pred_test
        sns.histplot(residuals, kde=True, ax=ax[1], color="#1f77b4", bins=20)
        ax[1].axvline(0, color="r", linestyle="--")
        ax[1].set_title(f"{name}: Residual Error Distribution")
        ax[1].set_xlabel("Error (₹)")
        
        plot_path = f"{REPORTS_DIR}/eval_reg_budget_{name.lower().replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(plot_path)
        
        # Log Model
        signature = infer_signature(X_train_b, y_pred_train)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train_b.head(2)
        )
        
        bgt_metrics_list.append({
            "Model": name,
            "Run ID": run.info.run_id,
            "Model URI": model_info.model_uri,
            "Test R²": round(te_r2, 4),
            "Test MAE": round(te_mae, 2),
            "Test RMSE": round(te_rmse, 2),
            "Test MAPE (%)": round(te_mape, 2),
            "Train R²": round(tr_r2, 4),
            "CV R² (Mean)": round(cv_mean, 4),
            "CV R² (Std)": round(cv_std, 4)
        })
        
        print(f"  • {name:20s} | Test R²: {te_r2:.4f} | Test MAE: ₹{te_mae:,.2f} | MAPE: {te_mape:.2f}% | 5-Fold CV R²: {cv_mean:.3f}")

bgt_summary_df = pd.DataFrame(bgt_metrics_list)

# ------------------------------------------------------------------------------
# 4. GENERATE COMPARISON CHARTS & LOG SUMMARY RUN TO MLFLOW
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("📊 GENERATING COMPREHENSIVE COMPARISON CHARTS FOR ALL 3 TASKS")
print("=" * 70)

palette = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2"]

# Task A: Regression Comparison Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sns.barplot(x="Model", y="Test R²", data=reg_summary_df, ax=axes[0], palette=palette)
axes[0].set_title("A. Test R² Comparison (Higher is Better)", fontweight="bold")
axes[0].set_ylim(0, max(reg_summary_df["Test R²"]) * 1.25)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontweight="bold")

sns.barplot(x="Model", y="Test MAE", data=reg_summary_df, ax=axes[1], palette=palette)
axes[1].set_title("B. Test MAE Comparison (Lower is Better)", fontweight="bold")
axes[1].set_ylim(0, max(reg_summary_df["Test MAE"]) * 1.25)
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontweight="bold")

sns.barplot(x="Model", y="CV R² (Mean)", data=reg_summary_df, ax=axes[2], palette=palette)
axes[2].set_title("C. 5-Fold Cross-Validation R²", fontweight="bold")
axes[2].set_ylim(0, max(reg_summary_df["CV R² (Mean)"]) * 1.25)
for p in axes[2].patches:
    axes[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontweight="bold")

for ax in axes:
    ax.tick_params(axis="x", rotation=15)

plt.suptitle("Model Benchmark Comparison: Student Attendance Forecasting (Task A)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
reg_plot_file = f"{REPORTS_DIR}/model_comparison_regression.png"
fig.savefig(reg_plot_file, dpi=120, bbox_inches="tight")
plt.close(fig)

# Task B: Classification Comparison Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sns.barplot(x="Model", y="ROC-AUC", data=cls_summary_df, ax=axes[0], palette=palette)
axes[0].set_title("A. Test ROC-AUC (Higher is Better)", fontweight="bold")
axes[0].set_ylim(0, 1.1)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontweight="bold")

sns.barplot(x="Model", y="F1-Score", data=cls_summary_df, ax=axes[1], palette=palette)
axes[1].set_title("B. Test F1-Score (Higher is Better)", fontweight="bold")
axes[1].set_ylim(0, 1.1)
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontweight="bold")

sns.barplot(x="Model", y="Accuracy", data=cls_summary_df, ax=axes[2], palette=palette)
axes[2].set_title("C. Classification Accuracy", fontweight="bold")
axes[2].set_ylim(0, 1.1)
for p in axes[2].patches:
    axes[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontweight="bold")

for ax in axes:
    ax.tick_params(axis="x", rotation=15)

plt.suptitle("Model Benchmark Comparison: Booking Approval (Task B)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
cls_plot_file = f"{REPORTS_DIR}/model_comparison_classification.png"
fig.savefig(cls_plot_file, dpi=120, bbox_inches="tight")
plt.close(fig)

# Task C: Budget Comparison Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sns.barplot(x="Model", y="Test R²", data=bgt_summary_df, ax=axes[0], palette=palette)
axes[0].set_title("A. Test R² (Higher is Better)", fontweight="bold")
axes[0].set_ylim(0, 1.1)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", xytext=(0, 4), textcoords="offset points", fontweight="bold")

sns.barplot(x="Model", y="Test MAE", data=bgt_summary_df, ax=axes[1], palette=palette)
axes[1].set_title("B. Test MAE in ₹ (Lower is Better)", fontweight="bold")
axes[1].set_ylim(0, max(bgt_summary_df["Test MAE"]) * 1.25)
for p in axes[1].patches:
    axes[1].annotate(f"₹{p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", xytext=(0, 4), textcoords="offset points", fontweight="bold")

sns.barplot(x="Model", y="CV R² (Mean)", data=bgt_summary_df, ax=axes[2], palette=palette)
axes[2].set_title("C. 5-Fold Cross-Validation R²", fontweight="bold")
axes[2].set_ylim(0, 1.1)
for p in axes[2].patches:
    axes[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", xytext=(0, 4), textcoords="offset points", fontweight="bold")

for ax in axes:
    ax.tick_params(axis="x", rotation=15)

plt.suptitle("Model Benchmark Comparison: Event Budget Forecasting (Task C)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
bgt_plot_file = f"{REPORTS_DIR}/model_comparison_budget.png"
fig.savefig(bgt_plot_file, dpi=120, bbox_inches="tight")
plt.close(fig)

# Save Comparison CSVs
reg_summary_df.to_csv(f"{REPORTS_DIR}/regression_comparison.csv", index=False)
cls_summary_df.to_csv(f"{REPORTS_DIR}/classification_comparison.csv", index=False)
bgt_summary_df.to_csv(f"{REPORTS_DIR}/budget_comparison.csv", index=False)

# ------------------------------------------------------------------------------
# 5. REGISTER CHAMPION MODELS IN MLFLOW MODEL REGISTRY
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("🏆 REGISTERING CHAMPION MODELS IN MLFLOW MODEL REGISTRY")
print("=" * 70)

# 1. Regression Champion (lowest MAE)
champion_reg = reg_summary_df.sort_values(by="Test MAE").iloc[0]
reg_model_name = "college-attendance-champion"
print(f"1. Attendance Champion: {champion_reg['Model']} (Test MAE: {champion_reg['Test MAE']}, Test R²: {champion_reg['Test R²']})")
reg_mv = mlflow.register_model(model_uri=champion_reg["Model URI"], name=reg_model_name)
client.set_registered_model_alias(name=reg_model_name, alias="champion", version=reg_mv.version)

# 2. Classification Champion (highest ROC-AUC)
champion_cls = cls_summary_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]
cls_model_name = "college-approval-champion"
print(f"2. Approval Champion:   {champion_cls['Model']} (Test ROC-AUC: {champion_cls['ROC-AUC']}, F1: {champion_cls['F1-Score']})")
cls_mv = mlflow.register_model(model_uri=champion_cls["Model URI"], name=cls_model_name)
client.set_registered_model_alias(name=cls_model_name, alias="champion", version=cls_mv.version)

# 3. Budget Champion (lowest MAE)
champion_bgt = bgt_summary_df.sort_values(by="Test MAE").iloc[0]
bgt_model_name = "college-budget-champion"
print(f"3. Budget Champion:     {champion_bgt['Model']} (Test MAE: ₹{champion_bgt['Test MAE']:,.2f}, Test R²: {champion_bgt['Test R²']})")
bgt_mv = mlflow.register_model(model_uri=champion_bgt["Model URI"], name=bgt_model_name)
client.set_registered_model_alias(name=bgt_model_name, alias="champion", version=bgt_mv.version)

# Serialize Champion Budget Model artifact (.joblib)
champion_bgt_obj = trained_bgt_models[champion_bgt["Model"]]
bgt_joblib_path = os.path.join(MODELS_DIR, "champion_budget_model.joblib")
joblib.dump(champion_bgt_obj, bgt_joblib_path)
print(f"💾 Serialized champion budget model to: {bgt_joblib_path}")

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
metrics_out["budget_regression"] = {
    "champion_model": champion_bgt["Model"],
    "test_mae": float(champion_bgt["Test MAE"]),
    "test_rmse": float(champion_bgt["Test RMSE"]),
    "test_r2": float(champion_bgt["Test R²"]),
    "test_mape": float(champion_bgt["Test MAPE (%)"]),
    "cv_r2": float(champion_bgt["CV R² (Mean)"]),
    "model_artifact": bgt_joblib_path
}

with open(metrics_json_path, "w") as f:
    json.dump(metrics_out, f, indent=2)

print("\n🎉 Triple-Task Training, Comparison and MLflow Registration Completed Successfully!")
print(f"Saved artifacts:")
print(f"  • {reg_plot_file}")
print(f"  • {cls_plot_file}")
print(f"  • {bgt_plot_file}")
print(f"  • {REPORTS_DIR}/budget_comparison.csv")
print(f"  • {bgt_joblib_path}")
print(f"  • {metrics_json_path}")
print("\nTo view the MLflow UI dashboard, run:")
print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")

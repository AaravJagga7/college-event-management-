"""
================================================================================
Comprehensive Model Training, Metrics Computation & MLflow Web Integration
================================================================================
Logs every metric:
  - Accuracy, Precision (Accept/Reject), Recall / Sensitivity, Specificity
  - F1-Score (Accept/Reject, Macro, Weighted)
  - Confusion Matrix (TP, TN, FP, FN)
  - ROC-AUC and 5-Fold Cross Validation
================================================================================
"""

import os
import json
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

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    r2_score, mean_absolute_error, mean_squared_error
)

# ------------------------------------------------------------------------------
# 1. SETUP MLFLOW TRACKING
# ------------------------------------------------------------------------------
DATA_PATH = "features_ready.csv"
DB_PATH = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "college-event-models"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

mlflow.set_tracking_uri(DB_PATH)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

print(f"✅ MLflow Backend Store: {DB_PATH}")
print(f"📊 Experiment: '{EXPERIMENT_NAME}'")

df = pd.read_csv(DATA_PATH)
feature_cols_cls = [c for c in df.columns if c not in ["TARGET_registers", "TARGET_accept"]]
feature_cols_reg = [c for c in df.columns if c not in ["TARGET_registers", "TARGET_accept", "fill_ratio"]]

# ------------------------------------------------------------------------------
# 2. CLASSIFICATION: BOOKING APPROVAL (TARGET_accept)
# ------------------------------------------------------------------------------
print("\n" + "=" * 75)
print("🚀 TRAINING & EVALUATING CLASSIFICATION MODELS (TARGET_accept)")
print("=" * 75)

X_cls = df[feature_cols_cls]
y_cls = df["TARGET_accept"]

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_cls, y_cls, test_size=0.20, random_state=42, stratify=y_cls
)

dt_cls = DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, class_weight="balanced", random_state=42)
rf_cls = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
gb_cls = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
voting_cls = VotingClassifier(
    estimators=[("dt", dt_cls), ("rf", rf_cls), ("gb", gb_cls)],
    voting="soft",
    weights=[1, 2, 2]
)

cls_models = {
    "Decision Tree": dt_cls,
    "Random Forest": rf_cls,
    "Gradient Boosting": gb_cls,
    "Voting Classifier": voting_cls
}

detailed_metrics = []

for name, model in cls_models.items():
    run_name = f"Classification_{name.replace(' ', '_')}"
    with mlflow.start_run(run_name=run_name) as run:
        # Cross validation
        cv_scores = cross_val_score(model, X_tr_c, y_tr_c, cv=5, scoring="roc_auc")
        
        # Fit model
        model.fit(X_tr_c, y_tr_c)
        y_pred = model.predict(X_te_c)
        y_prob = model.predict_proba(X_te_c)[:, 1]
        
        # Confusion matrix calculations
        cm = confusion_matrix(y_te_c, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Detailed score calculations
        acc = accuracy_score(y_te_c, y_pred)
        prec_accept = precision_score(y_te_c, y_pred, pos_label=1)
        rec_accept = recall_score(y_te_c, y_pred, pos_label=1)   # Sensitivity (True Positive Rate)
        spec_accept = tn / (tn + fp)                             # Specificity (True Negative Rate)
        f1_accept = f1_score(y_te_c, y_pred, pos_label=1)
        
        prec_reject = precision_score(y_te_c, y_pred, pos_label=0)
        rec_reject = recall_score(y_te_c, y_pred, pos_label=0)
        f1_reject = f1_score(y_te_c, y_pred, pos_label=0)
        
        macro_prec = precision_score(y_te_c, y_pred, average="macro")
        macro_rec = recall_score(y_te_c, y_pred, average="macro")
        macro_f1 = f1_score(y_te_c, y_pred, average="macro")
        weighted_f1 = f1_score(y_te_c, y_pred, average="weighted")
        roc_auc = roc_auc_score(y_te_c, y_prob)
        cv_auc_mean = float(cv_scores.mean())
        
        # Log tags & hyperparams
        mlflow.set_tag("task", "classification")
        mlflow.set_tag("target", "TARGET_accept")
        mlflow.set_tag("model_family", name)
        
        if hasattr(model, "get_params"):
            clean_params = {k: str(v) for k, v in model.get_params().items() if not k.startswith("estimators") and len(str(v)) < 250}
            mlflow.log_params(clean_params)
            
        # Log EVERY calculated score into MLflow
        mlflow.log_metrics({
            "accuracy": acc,
            "precision_accept": prec_accept,
            "sensitivity_accept": rec_accept,
            "recall_accept": rec_accept,
            "specificity": spec_accept,
            "f1_score_accept": f1_accept,
            "precision_reject": prec_reject,
            "recall_reject": rec_reject,
            "f1_score_reject": f1_reject,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "roc_auc": roc_auc,
            "cv_roc_auc_5fold": cv_auc_mean,
            "true_positives_tp": int(tp),
            "true_negatives_tn": int(tn),
            "false_positives_fp": int(fp),
            "false_negatives_fn": int(fn)
        })
        
        # Save & Log Confusion Matrix Figure
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["Reject (0)", "Accept (1)"]).plot(ax=ax, cmap="Blues")
        ax.set_title(f"{name}: Confusion Matrix\\n(Sensitivity={rec_accept:.1%}, Specificity={spec_accept:.1%})", fontsize=10, fontweight="bold")
        fig.tight_layout()
        cm_path = f"{REPORTS_DIR}/cm_{name.lower().replace(' ', '_')}.png"
        fig.savefig(cm_path, dpi=110)
        mlflow.log_figure(fig, f"confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close(fig)
        
        # Save Model Artifact with Signature
        signature = infer_signature(X_tr_c, y_pred)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_tr_c.head(2)
        )
        
        detailed_metrics.append({
            "Model": name,
            "Run ID": run.info.run_id,
            "Model URI": model_info.model_uri,
            "Accuracy": round(acc, 4),
            "Precision (Accept)": round(prec_accept, 4),
            "Recall / Sensitivity": round(rec_accept, 4),
            "Specificity": round(spec_accept, 4),
            "F1-Score (Accept)": round(f1_accept, 4),
            "Precision (Reject)": round(prec_reject, 4),
            "Recall (Reject)": round(rec_reject, 4),
            "F1-Score (Reject)": round(f1_reject, 4),
            "Macro F1": round(macro_f1, 4),
            "Weighted F1": round(weighted_f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        })
        
        print(f"  • {name:20s} | Acc: {acc:.3f} | Prec: {prec_accept:.3f} | Sensitivity: {rec_accept:.3f} | Specificity: {spec_accept:.3f} | F1: {f1_accept:.3f} | AUC: {roc_auc:.3f}")

metrics_df = pd.DataFrame(detailed_metrics)
metrics_df.to_csv(f"{REPORTS_DIR}/detailed_classification_metrics.csv", index=False)

# Register Champion in MLflow
champion = metrics_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]
model_reg_name = "college-approval-champion"
reg_mv = mlflow.register_model(model_uri=champion["Model URI"], name=model_reg_name)
client.set_registered_model_alias(name=model_reg_name, alias="champion", version=reg_mv.version)
print(f"\n🏆 Champion Model: '{champion['Model']}' registered as '{model_reg_name}' @champion")

# ------------------------------------------------------------------------------
# 3. REGRESSION: ATTENDANCE FORECASTING (TARGET_registers)
# ------------------------------------------------------------------------------
print("\n" + "=" * 75)
print("🚀 TRAINING & EVALUATING REGRESSION MODELS (TARGET_registers)")
print("=" * 75)

X_reg = df[feature_cols_reg]
y_reg = df["TARGET_registers"]

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.20, random_state=42)

dt_reg = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)
gb_reg = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
voting_reg = VotingRegressor(estimators=[("dt", dt_reg), ("rf", rf_reg), ("gb", gb_reg)], weights=[1, 2, 2])

for name, model in {"Decision Tree": dt_reg, "Random Forest": rf_reg, "Gradient Boosting": gb_reg, "Voting Regressor": voting_reg}.items():
    with mlflow.start_run(run_name=f"Regression_{name.replace(' ', '_')}") as run:
        cv_scores = cross_val_score(model, X_tr_r, y_tr_r, cv=5, scoring="r2")
        model.fit(X_tr_r, y_tr_r)
        preds = model.predict(X_te_r)
        
        r2 = r2_score(y_te_r, preds)
        mae = mean_absolute_error(y_te_r, preds)
        rmse = np.sqrt(mean_squared_error(y_te_r, preds))
        
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("target", "TARGET_registers")
        mlflow.set_tag("model_family", name)
        
        mlflow.log_metrics({
            "test_r2": r2,
            "test_mae": mae,
            "test_rmse": rmse,
            "cv_r2_5fold": float(cv_scores.mean())
        })
        
        signature = infer_signature(X_tr_r, preds)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", signature=signature, input_example=X_tr_r.head(2))
        print(f"  • {name:20s} | Test R²: {r2:.3f} | Test MAE: {mae:5.2f} | Test RMSE: {rmse:5.2f} | 5-Fold CV: {cv_scores.mean():.3f}")

print("\n🎉 All detailed metrics successfully logged to MLflow!")

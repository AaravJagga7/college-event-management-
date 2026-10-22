import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
)

# Load data
df = pd.read_csv("features_ready.csv")

# Exclude all target and leaky columns from classification
feature_cols_cls = [c for c in df.columns if c not in ["TARGET_registers", "TARGET_accept", "TARGET_budget"]]

X_c = df[feature_cols_cls]
y_c = df["TARGET_accept"] # 1 = Accept, 0 = Reject

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_c, y_c, test_size=0.20, random_state=42, stratify=y_c
)

# Task B Classification Models
dt_c = DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, class_weight="balanced", random_state=42)
rf_c = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
gb_c = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
voting_c = VotingClassifier(
    estimators=[("dt", dt_c), ("rf", rf_c), ("gb", gb_c)],
    voting="soft",
    weights=[1, 2, 2]
)

cls_models = {
    "Decision Tree": dt_c,
    "Random Forest": rf_c,
    "Gradient Boosting": gb_c,
    "Voting Classifier": voting_c
}

results_cls = []

for name, m in cls_models.items():
    m.fit(X_train_c, y_train_c)
    y_pred = m.predict(X_test_c)
    y_prob = m.predict_proba(X_test_c)[:, 1]
    
    cm = confusion_matrix(y_test_c, y_pred) # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test_c, y_pred)
    prec_1 = precision_score(y_test_c, y_pred, pos_label=1)
    rec_1 = recall_score(y_test_c, y_pred, pos_label=1)
    f1_1 = f1_score(y_test_c, y_pred, pos_label=1)
    
    prec_0 = precision_score(y_test_c, y_pred, pos_label=0)
    rec_0 = recall_score(y_test_c, y_pred, pos_label=0)
    f1_0 = f1_score(y_test_c, y_pred, pos_label=0)
    
    prec_macro = precision_score(y_test_c, y_pred, average="macro")
    rec_macro = recall_score(y_test_c, y_pred, average="macro")
    f1_macro = f1_score(y_test_c, y_pred, average="macro")
    f1_weighted = f1_score(y_test_c, y_pred, average="weighted")
    auc = roc_auc_score(y_test_c, y_prob)
    
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    results_cls.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision (Accept)": round(prec_1, 4),
        "Recall / Sensitivity (Accept)": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "F1-Score (Accept)": round(f1_1, 4),
        "Precision (Reject)": round(prec_0, 4),
        "Recall (Reject)": round(rec_0, 4),
        "F1-Score (Reject)": round(f1_0, 4),
        "Macro Precision": round(prec_macro, 4),
        "Macro Recall": round(rec_macro, 4),
        "Macro F1": round(f1_macro, 4),
        "Weighted F1": round(f1_weighted, 4),
        "ROC-AUC": round(auc, 4),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    })

res_cls_df = pd.DataFrame(results_cls)
res_cls_df.to_csv("reports/detailed_classification_metrics.csv", index=False)
print("=== TASK B: BOOKING APPROVAL CLASSIFICATION METRICS ===")
display_cols = ["Model", "Accuracy", "Precision (Accept)", "Recall / Sensitivity (Accept)", "Specificity", "F1-Score (Accept)", "Precision (Reject)", "Recall (Reject)", "F1-Score (Reject)", "ROC-AUC"]
print(res_cls_df[display_cols].to_string(index=False))

# Task C Budget Evaluation
if "TARGET_budget" in df.columns:
    print("\n=== TASK C: EVENT BUDGET REGRESSION METRICS ===")
    feat_bgt = [c for c in df.columns if c not in ["TARGET_budget", "TARGET_accept", "fill_ratio"]]
    X_b = df[feat_bgt]
    y_b = df["TARGET_budget"]
    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X_b, y_b, test_size=0.20, random_state=42)
    
    bgt_models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    }
    bgt_voting = VotingRegressor(
        estimators=[("dt", bgt_models["Decision Tree"]), ("rf", bgt_models["Random Forest"]), ("gb", bgt_models["Gradient Boosting"])],
        weights=[1, 2, 2]
    )
    bgt_models["Voting Regressor"] = bgt_voting
    
    results_bgt = []
    for name, m in bgt_models.items():
        m.fit(X_tr_b, y_tr_b)
        preds = m.predict(X_te_b)
        r2 = r2_score(y_te_b, preds)
        mae = mean_absolute_error(y_te_b, preds)
        rmse = np.sqrt(mean_squared_error(y_te_b, preds))
        mape = np.mean(np.abs((y_te_b - preds) / y_te_b)) * 100
        results_bgt.append({
            "Model": name,
            "Test R²": round(r2, 4),
            "Test MAE (₹)": round(mae, 2),
            "Test RMSE (₹)": round(rmse, 2),
            "MAPE (%)": round(mape, 2)
        })
    bgt_df = pd.DataFrame(results_bgt)
    print(bgt_df.to_string(index=False))

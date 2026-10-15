import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

# Load data
df = pd.read_csv("features_ready.csv")
feature_cols_cls = [c for c in df.columns if c not in ["TARGET_registers", "TARGET_accept"]]

X = df[feature_cols_cls]
y = df["TARGET_accept"] # 1 = Accept, 0 = Reject

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Models
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, class_weight="balanced", random_state=42)
rf = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
voting = VotingClassifier(
    estimators=[("dt", dt), ("rf", rf), ("gb", gb)],
    voting="soft",
    weights=[1, 2, 2]
)

models = {
    "Decision Tree": dt,
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "Voting Classifier": voting
}

results = []

for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    y_prob = m.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred) # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test, y_pred)
    
    # Class 1 (Accept) metrics
    prec_1 = precision_score(y_test, y_pred, pos_label=1)
    rec_1 = recall_score(y_test, y_pred, pos_label=1) # Sensitivity for Accept class
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    
    # Class 0 (Reject - Minority Critical class) metrics
    prec_0 = precision_score(y_test, y_pred, pos_label=0)
    rec_0 = recall_score(y_test, y_pred, pos_label=0) # Sensitivity for Reject class / Specificity for Accept class
    f1_0 = f1_score(y_test, y_pred, pos_label=0)
    
    # Macro and Weighted metrics
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_prob)
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp)
    # Sensitivity = TP / (TP + FN) = rec_1
    sensitivity = tp / (tp + fn)
    
    results.append({
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

res_df = pd.DataFrame(results)
res_df.to_csv("reports/detailed_classification_metrics.csv", index=False)
print("=== OVERALL METRICS TABLE ===")
display_cols = ["Model", "Accuracy", "Precision (Accept)", "Recall / Sensitivity (Accept)", "Specificity", "F1-Score (Accept)", "Precision (Reject)", "Recall (Reject)", "F1-Score (Reject)", "ROC-AUC"]
print(res_df[display_cols].to_string(index=False))

"""
Generates high-resolution, publication-quality visualizations for model performance:
1. reports/confusion_matrix_all_models.png (2x2 grid of confusion matrices with TP, TN, FP, FN annotations)
2. reports/model_performance_metrics.png (Comprehensive comparative bar charts of Accuracy, Precision, Recall, F1, ROC-AUC)
3. reports/master_model_performance_dashboard.png (Executive all-in-one performance dashboard combining ROC curves, confusion matrices, and metric comparisons)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# Set high-DPI matplotlib configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#D1D5DB'
plt.rcParams['axes.linewidth'] = 1.0

os.makedirs("reports", exist_ok=True)

# 1. Load Data
df = pd.read_csv("features_ready.csv")
feature_cols_cls = [c for c in df.columns if c not in ["TARGET_registers", "TARGET_accept", "TARGET_budget"]]

X = df[feature_cols_cls]
y = df["TARGET_accept"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Train Models
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
cms = {}
roc_data = {}

for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    y_prob = m.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cms[name] = (cm, tn, fp, fn, tp)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc_val)
    
    acc = accuracy_score(y_test, y_pred)
    prec_1 = precision_score(y_test, y_pred, pos_label=1)
    rec_1 = recall_score(y_test, y_pred, pos_label=1)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    
    prec_0 = precision_score(y_test, y_pred, pos_label=0)
    rec_0 = recall_score(y_test, y_pred, pos_label=0)
    f1_0 = f1_score(y_test, y_pred, pos_label=0)
    
    specificity = tn / (tn + fp)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec_1,
        "Recall": rec_1,
        "F1-Score": f1_1,
        "Specificity": specificity,
        "ROC-AUC": auc_val,
        "Precision (Reject)": prec_0,
        "Recall (Reject)": rec_0,
        "F1-Score (Reject)": f1_0,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    })

res_df = pd.DataFrame(results)
res_df.to_csv("reports/detailed_classification_metrics.csv", index=False)
print("Updated reports/detailed_classification_metrics.csv")

# ==============================================================================
# IMAGE 1: CONFUSION MATRIX ALL MODELS (2x2 Grid)
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
axes = axes.flatten()

model_order = ["Decision Tree", "Random Forest", "Gradient Boosting", "Voting Classifier"]
palette_cms = {
    "Decision Tree": "Blues",
    "Random Forest": "Greens",
    "Gradient Boosting": "Purples",
    "Voting Classifier": "YlOrBr"
}

for idx, name in enumerate(model_order):
    ax = axes[idx]
    cm, tn, fp, fn, tp = cms[name]
    row_data = res_df[res_df["Model"] == name].iloc[0]
    
    # Custom annotations with counts, percentages, and cell labels
    # cm layout: [[TN, FP], [FN, TP]]
    total = np.sum(cm)
    annot = [
        [f"TN: {tn}\n({tn/(tn+fp)*100:.1f}%)", f"FP: {fp}\n({fp/(tn+fp)*100:.1f}%)"],
        [f"FN: {fn}\n({fn/(fn+tp)*100:.1f}%)", f"TP: {tp}\n({tp/(fn+tp)*100:.1f}%)"]
    ]
    
    cmap_choice = palette_cms[name]
    sns.heatmap(
        cm, annot=np.array(annot), fmt="", cmap=cmap_choice, ax=ax,
        cbar=False, annot_kws={"size": 13, "weight": "bold", "color": "#111827"},
        linewidths=2, linecolor="#F3F4F6", square=True
    )
    
    # Title and metrics subtitle
    is_champ = (name == "Random Forest")
    title_text = f"[CHAMPION] {name} (Production Champion)" if is_champ else f"{name}"
    title_color = "#047857" if is_champ else "#1F2937"
    
    ax.set_title(title_text, fontsize=14, weight="bold", pad=12, color=title_color)
    subtitle = (
        f"Accuracy: {row_data['Accuracy']*100:.1f}%  |  "
        f"F1: {row_data['F1-Score']*100:.1f}%  |  "
        f"Precision: {row_data['Precision']*100:.1f}%  |  "
        f"ROC-AUC: {row_data['ROC-AUC']:.3f}"
    )
    ax.text(0.5, -0.18, subtitle, transform=ax.transAxes, ha="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9))
    
    ax.set_xlabel("Predicted Class", fontsize=11, weight="semibold", labelpad=8)
    ax.set_ylabel("Actual Class (Ground Truth)", fontsize=11, weight="semibold", labelpad=8)
    ax.set_xticklabels(["Reject (0)", "Accept (1)"], fontsize=10.5)
    ax.set_yticklabels(["Reject (0)", "Accept (1)"], fontsize=10.5, rotation=0)

plt.suptitle("RVCE 8th Mile — Task B: Booking Approval Confusion Matrices\nTest Set Evaluation (N = 240 samples: 196 Accepts, 44 Rejections)",
             fontsize=16, weight="bold", y=0.99, color="#111827")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
cm_output_path = "reports/confusion_matrix_all_models.png"
plt.savefig(cm_output_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {cm_output_path}")


# ==============================================================================
# IMAGE 2: ALL MODEL PERFORMANCES (Accuracy, Precision, Recall, F1, ROC-AUC)
# ==============================================================================
fig, (ax_main, ax_detailed) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

# Panel A: Core Key Performance Indicators (Accuracy, Precision, Recall, F1, ROC-AUC)
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(metrics_to_plot))
width = 0.18

model_colors = {
    "Decision Tree": "#3B82F6",       # Blue
    "Random Forest": "#10B981",       # Emerald Green (Champion)
    "Gradient Boosting": "#8B5CF6",   # Purple
    "Voting Classifier": "#F59E0B"    # Amber
}

for i, model_name in enumerate(model_order):
    m_data = res_df[res_df["Model"] == model_name].iloc[0]
    vals = [m_data[m] for m in metrics_to_plot]
    pos = x + (i - 1.5) * width
    bars = ax_main.bar(pos, vals, width, label=model_name + (" [CHAMPION]" if model_name == "Random Forest" else ""),
                       color=model_colors[model_name], edgecolor="#1F2937", linewidth=0.8, alpha=0.9)
    
    # Add numerical labels on top of bars
    for bar in bars:
        h = bar.get_height()
        ax_main.text(bar.get_x() + bar.get_width() / 2., h + 0.012,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, weight="bold", rotation=0)

ax_main.set_title("Core Performance Metrics by Architecture", fontsize=14, weight="bold", pad=12)
ax_main.set_xticks(x)
ax_main.set_xticklabels(metrics_to_plot, fontsize=11, weight="semibold")
ax_main.set_ylim(0, 1.08)
ax_main.set_ylabel("Score (0.0 to 1.0)", fontsize=11, weight="semibold")
ax_main.axhline(0.9, color="#10B981", linestyle="--", alpha=0.4, label="90% Benchmark")
ax_main.grid(axis="y", linestyle=":", alpha=0.6)
ax_main.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="lower right", fontsize=10)

# Panel B: Class-Wise & Balance Breakdown (Accept vs. Reject Minority Class)
class_metrics = ["F1-Score", "Precision", "Recall", "Precision (Reject)", "Recall (Reject)", "F1-Score (Reject)"]
x2 = np.arange(len(class_metrics))
width2 = 0.18

for i, model_name in enumerate(model_order):
    m_data = res_df[res_df["Model"] == model_name].iloc[0]
    vals = [m_data[m] for m in class_metrics]
    pos = x2 + (i - 1.5) * width2
    bars = ax_detailed.bar(pos, vals, width2, label=model_name,
                           color=model_colors[model_name], edgecolor="#1F2937", linewidth=0.8, alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax_detailed.text(bar.get_x() + bar.get_width() / 2., h + 0.012,
                         f"{h:.2f}", ha="center", va="bottom", fontsize=8, weight="bold")

ax_detailed.set_title("Class-Specific Breakdown (Accept Class vs. Reject Minority)", fontsize=14, weight="bold", pad=12)
ax_detailed.set_xticks(x2)
ax_detailed.set_xticklabels(["F1 (Acc)", "Prec (Acc)", "Rec (Acc)", "Prec (Rej)", "Rec (Rej)", "F1 (Rej)"],
                            fontsize=10.5, weight="semibold")
ax_detailed.set_ylim(0, 1.08)
ax_detailed.set_ylabel("Score (0.0 to 1.0)", fontsize=11, weight="semibold")
ax_detailed.grid(axis="y", linestyle=":", alpha=0.6)
ax_detailed.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", loc="lower right", fontsize=10)

plt.suptitle("RVCE 8th Mile — Model Performance Comparison (F1-Score, Precision, Recall, Accuracy, ROC-AUC)",
             fontsize=16, weight="bold", y=0.99, color="#111827")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
perf_output_path = "reports/model_performance_metrics.png"
plt.savefig(perf_output_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {perf_output_path}")


# ==============================================================================
# IMAGE 3: MASTER MODEL PERFORMANCE DASHBOARD (All-In-One Executive View)
# ==============================================================================
fig = plt.figure(figsize=(22, 13), dpi=300, facecolor="#F8FAFC")
gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.32, wspace=0.25)

# 1. Top Left (Span 2 cols): Main Performance Bar Chart (Accuracy, Precision, Recall, F1, ROC-AUC)
ax_bar = fig.add_subplot(gs[0, 0:2])
x = np.arange(len(metrics_to_plot))
width = 0.19
for i, model_name in enumerate(model_order):
    m_data = res_df[res_df["Model"] == model_name].iloc[0]
    vals = [m_data[m] for m in metrics_to_plot]
    pos = x + (i - 1.5) * width
    bars = ax_bar.bar(pos, vals, width, label=model_name + (" [CHAMPION]" if model_name == "Random Forest" else ""),
                      color=model_colors[model_name], edgecolor="#1F2937", linewidth=0.7, alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2., h + 0.015,
                    f"{h*100:.1f}%" if "AUC" not in metrics_to_plot[int(round(bar.get_x()))] else f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8.5, weight="bold")

ax_bar.set_title("A. Cross-Model Metric Benchmarks (N=240 Test Set)", fontsize=13, weight="bold", color="#111827", pad=10)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metrics_to_plot, fontsize=11, weight="bold")
ax_bar.set_ylim(0, 1.1)
ax_bar.set_ylabel("Score / Performance", fontsize=10.5, weight="bold")
ax_bar.grid(axis="y", linestyle=":", alpha=0.6)
ax_bar.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.5)

# 2. Top Right (Span 2 cols): ROC Curves Comparison
ax_roc = fig.add_subplot(gs[0, 2:4])
for name in model_order:
    fpr, tpr, auc_val = roc_data[name]
    is_champ = (name == "Random Forest")
    lw = 3.0 if is_champ else 1.8
    ax_roc.plot(fpr, tpr, color=model_colors[name], lw=lw,
                label=f"{name} (AUC = {auc_val:.3f}){' [CHAMPION]' if is_champ else ''}")

ax_roc.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.500)")
ax_roc.set_title("B. Receiver Operating Characteristic (ROC) Curves", fontsize=13, weight="bold", color="#111827", pad=10)
ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10.5, weight="bold")
ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10.5, weight="bold")
ax_roc.set_xlim([-0.02, 1.02])
ax_roc.set_ylim([-0.02, 1.05])
ax_roc.grid(True, linestyle=":", alpha=0.6)
ax_roc.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=10)

# 3. Bottom Row: 4 Confusion Matrices Side-by-Side
for i, name in enumerate(model_order):
    ax_cm = fig.add_subplot(gs[1, i])
    cm, tn, fp, fn, tp = cms[name]
    row_data = res_df[res_df["Model"] == name].iloc[0]
    
    annot = [
        [f"TN: {tn}\n({tn/(tn+fp)*100:.1f}%)", f"FP: {fp}\n({fp/(tn+fp)*100:.1f}%)"],
        [f"FN: {fn}\n({fn/(fn+tp)*100:.1f}%)", f"TP: {tp}\n({tp/(fn+tp)*100:.1f}%)"]
    ]
    
    cmap_choice = palette_cms[name]
    sns.heatmap(
        cm, annot=np.array(annot), fmt="", cmap=cmap_choice, ax=ax_cm,
        cbar=False, annot_kws={"size": 11, "weight": "bold", "color": "#111827"},
        linewidths=1.5, linecolor="#F3F4F6", square=True
    )
    
    is_champ = (name == "Random Forest")
    title_text = f"[CHAMPION] {name}\n(Champion)" if is_champ else f"{name}"
    title_color = "#047857" if is_champ else "#1F2937"
    ax_cm.set_title(title_text, fontsize=11.5, weight="bold", pad=8, color=title_color)
    
    ax_cm.set_xlabel("Predicted", fontsize=9.5, weight="semibold")
    if i == 0:
        ax_cm.set_ylabel("Actual (Ground Truth)", fontsize=10, weight="bold")
    else:
        ax_cm.set_ylabel("")
        
    ax_cm.set_xticklabels(["Reject (0)", "Accept (1)"], fontsize=9)
    ax_cm.set_yticklabels(["Reject (0)", "Accept (1)"], fontsize=9, rotation=0)
    
    # Sub-box with accuracy & F1
    ax_cm.text(0.5, -0.22, f"Acc: {row_data['Accuracy']*100:.1f}% | F1: {row_data['F1-Score']*100:.1f}%\nPrec: {row_data['Precision']*100:.1f}% | Rec: {row_data['Recall']*100:.1f}%",
               transform=ax_cm.transAxes, ha="center", fontsize=8.5,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.9))

plt.suptitle("RVCE 8th Mile Festival — Master Model Evaluation Dashboard\nComprehensive Performance (F1-Score, Precision, Recall, Accuracy, ROC-AUC) & Confusion Matrices",
             fontsize=17, weight="bold", y=0.98, color="#0F172A")

master_output_path = "reports/master_model_performance_dashboard.png"
plt.savefig(master_output_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {master_output_path}")

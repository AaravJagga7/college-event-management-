"""
================================================================================
College Event Management System: EDA & Feature Extraction Pipeline
================================================================================
# Inputs:  rvce_8th_mile_data.csv
# Outputs: features_ready.csv, eda_visualization.png, feature_importance.png
# ================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ------------------------------------------------------------------------------
# 1. LOAD RAW DATA
# ------------------------------------------------------------------------------
RAW_DATA_PATH = "rvce_8th_mile_data.csv"
print(f">>> [1/5] Loading raw data: {RAW_DATA_PATH} (RV College of Engineering - 8th Mile Festival)...")
df_raw = pd.read_csv(RAW_DATA_PATH)
print(f"Loaded {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.\n")

# ------------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
print(">>> [2/5] Performing Exploratory Data Analysis (EDA)...")
print("-" * 50)
print("A. Missing values per column:")
missing = df_raw.isnull().sum()
print(missing[missing > 0])
print("\nB. Constant / Zero-variance columns:")
constant_cols = [c for c in df_raw.columns if df_raw[c].nunique() == 1]
print(constant_cols)
print("\nC. Target variable statistics ('Registers'):")
print(df_raw["Registers"].describe().round(2))
print("\nD. Approval target ('Accept/Reject'):")
print(df_raw["Accept/Reject"].value_counts(normalize=True).round(3))
print("\nE. Rejection Reasons Breakdown:")
print(df_raw["Reject_Reason"].value_counts(dropna=False))
print("-" * 50)

# Generate EDA Visualization Figure
df_temp = df_raw.copy()
df_temp["Date"] = pd.to_datetime(df_temp["Request Date"], format="%m/%d/%Y")
df_temp["Submitted"] = pd.to_datetime(df_temp["Request Submitted On"], format="%m/%d/%Y")
df_temp["Month"] = df_temp["Date"].dt.month
df_temp["Lead_Days"] = (df_temp["Date"] - df_temp["Submitted"]).dt.days

sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(2, 3, figsize=(19, 10))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# 1. Distribution of Registers
sns.histplot(df_temp["Registers"], bins=30, kde=True, ax=axes[0, 0], color="#2b5c8f")
axes[0, 0].set_title("1. Turnout Distribution (RVCE 8th Mile)", fontweight="bold")
axes[0, 0].axvline(df_temp["Registers"].median(), color="red", linestyle="--", label=f"Median: {df_temp['Registers'].median():.0f}")
axes[0, 0].legend()

# 2. Registers by Event Type
sns.boxplot(x="Type", y="Registers", data=df_temp, order=["Normal", "Important", "Very important"], ax=axes[0, 1], palette="Blues_r", hue="Type", legend=False)
axes[0, 1].set_title("2. Turnout by Priority Level", fontweight="bold")

# 3. Registers across Event Categories
sns.barplot(x="Event_Category", y="Registers", data=df_temp, ax=axes[0, 2], palette="viridis", errorbar=None, hue="Event_Category", legend=False)
axes[0, 2].set_title("3. Mean Registrations across Categories", fontweight="bold")
axes[0, 2].tick_params(axis="x", rotation=20)

# 4. Monthly Seasonality
monthly = df_temp.groupby("Month")["Registers"].mean().reset_index()
sns.lineplot(x="Month", y="Registers", data=monthly, marker="o", color="#d95f02", linewidth=2.5, ax=axes[1, 0])
axes[1, 0].set_title("4. Attendance Seasonality (Sept - Dec Fest Cycle)", fontweight="bold")
axes[1, 0].set_xticks([9, 10, 11, 12])
axes[1, 0].set_xticklabels(["Sep", "Oct", "Nov", "Dec"])

# 5. Rejection Rate by Venue
venue_col = "Requested Venue" if "Requested Venue" in df_temp.columns else "Requested Room No"
rej_by_room = df_temp.assign(rej=(df_temp["Accept/Reject"] == "Reject").astype(int))\
                     .groupby(venue_col)["rej"].mean().mul(100).reset_index()
sns.barplot(x=venue_col, y="rej", data=rej_by_room, ax=axes[1, 1], palette="magma", hue=venue_col, legend=False)
axes[1, 1].set_title(f"5. Rejection Rate (%) by Venue", fontweight="bold")
axes[1, 1].tick_params(axis="x", rotation=45, labelsize=8)

# 6. Rejection Reasons Pie Chart
rej_counts = df_raw["Reject_Reason"].dropna().value_counts()
if len(rej_counts) > 0:
    axes[1, 2].pie(rej_counts, labels=rej_counts.index, autopct="%1.1f%%", colors=["#e41a1c", "#377eb8", "#4daf4a"], startangle=140)
axes[1, 2].set_title("6. Primary Causes for Booking Rejections", fontweight="bold")

plt.suptitle("Exploratory Data Analysis: RV College of Engineering — 8th Mile Festival", fontsize=15, fontweight="bold", y=0.98)
plt.savefig("eda_visualization.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved visualization: eda_visualization.png\n")

# ------------------------------------------------------------------------------
# 3. FEATURE EXTRACTION PIPELINE
# ------------------------------------------------------------------------------
print(">>> [3/5] Running Feature Extraction Pipeline...")

def extract_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    
    # Parse dates and times
    dt_event = pd.to_datetime(df["Request Date"], format="%m/%d/%Y")
    dt_submit = pd.to_datetime(df["Request Submitted On"], format="%m/%d/%Y")
    t_event = pd.to_datetime(df["Requested Time"], format="%I:%M%p")
    
    fe = pd.DataFrame(index=df.index)
    
    # 1. Temporal & Calendar Features
    fe["month"] = dt_event.dt.month
    fe["day_of_week"] = dt_event.dt.dayofweek            # 0=Monday, 6=Sunday
    fe["is_saturday"] = (fe["day_of_week"] == 5).astype(int)
    fe["week_of_year"] = dt_event.dt.isocalendar().week.astype(int)
    fe["is_exam_season"] = fe["month"].isin([4, 5, 11, 12]).astype(int) # College exam period
    
    # 2. Time of Day & Duration Features
    fe["hour"] = t_event.dt.hour                          # 24-hr format
    fe["is_afternoon"] = (fe["hour"] >= 14).astype(int)   # >= 2:00 PM (post-class slot)
    fe["duration_min"] = df["Duration"].str.replace("min", "").astype(int)
    
    # 3. Booking Behavior
    fe["lead_days"] = (dt_event - dt_submit).dt.days      # Planning / promo window
    
    # 4. Ordinal Encodings (Hierarchy preserved)
    fe["type_level"] = df["Type"].map({"Normal": 1, "Important": 2, "Very important": 3})
    fe["designation_level"] = df["Faculty_Designation"].map({
        "Assistant Professor": 1, 
        "Associate Professor": 2, 
        "Professor": 3
    })
    
    # 5. Room Physical Attributes
    fe["room_no"] = df["Requested Room No"]
    fe["room_capacity"] = df["Room_Capacity"]
    
    # 6. Entity Frequency Encodings
    fe["faculty_event_count"] = df.groupby("Faculty_id")["Event"].transform("count")
    fe["room_usage_count"] = df.groupby("Requested Room No")["Event"].transform("count")
    
    # 7. Spatio-Temporal Interval Overlap (Room Clash Engine)
    # Checks prior chronological requests for the same room & date with overlapping times
    start = t_event.dt.hour * 60 + t_event.dt.minute
    end = start + fe["duration_min"]
    clash = np.zeros(len(df), dtype=int)
    seen = {}
    for i in df["Request Submitted On"].argsort(kind="stable").values:
        key = (df.at[i, "Requested Room No"], df.at[i, "Request Date"])
        clash[i] = sum(1 for s, e in seen.get(key, []) if start[i] < e and s < end[i])
        seen.setdefault(key, []).append((start[i], end[i]))
    fe["earlier_clash_count"] = clash
    
    # 8. Nominal One-Hot Encodings (Fixed vocabulary)
    for c in ["academic", "cultural", "fun_club", "sports", "technical"]:
        fe[f"cat_{c}"] = (df["Event_Category"] == c).astype(int)
        
    # 9. Targets & Interaction Metric
    fe["TARGET_registers"] = df["Registers"]
    fe["TARGET_accept"] = (df["Accept/Reject"] == "Accept").astype(int)
    fe["fill_ratio"] = df["Registers"] / df["Room_Capacity"]
    if "Budget" in df.columns:
        fe["TARGET_budget"] = df["Budget"]
    
    return fe

features_df = extract_features(df_raw)
features_df.to_csv("features_ready.csv", index=False)
print(f"Extracted {features_df.shape[1]} features across {features_df.shape[0]} rows.")
print("Saved: features_ready.csv\n")

# ------------------------------------------------------------------------------
# 4. PREDICTIVE VALIDATION & FEATURE IMPORTANCE
# ------------------------------------------------------------------------------
print(">>> [4/5] Training Random Forest Models for Feature Importance...")

# Task A: Predict Registers (Attendance Regression) - NO TARGET LEAKAGE
drop_reg = [c for c in ["TARGET_registers", "TARGET_accept", "fill_ratio", "TARGET_budget"] if c in features_df.columns]
X_reg = features_df.drop(columns=drop_reg)
y_reg = features_df["TARGET_registers"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_r, y_train_r)

y_pred_r = rf_reg.predict(X_test_r)
r2_a = r2_score(y_test_r, y_pred_r)
mae_a = mean_absolute_error(y_test_r, y_pred_r)
print(f"Task A (Attendance Regression): R² = {r2_a:.3f}, MAE = {mae_a:.2f}")

# Task B: Predict Approval (Classification)
drop_cls = [c for c in ["TARGET_registers", "TARGET_accept", "TARGET_budget"] if c in features_df.columns]
X_cls = features_df.drop(columns=drop_cls)
y_cls = features_df["TARGET_accept"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cls.fit(X_train_c, y_train_c)

y_pred_c = rf_cls.predict(X_test_c)
y_prob_c = rf_cls.predict_proba(X_test_c)[:, 1]
roc_auc = roc_auc_score(y_test_c, y_prob_c)

# Confusion Matrix Computation
cm = confusion_matrix(y_test_c, y_pred_c)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
accuracy = (tp + tn) / len(y_test_c)

print(f"Task B (Approval Classification): ROC-AUC = {roc_auc:.3f}, Accuracy = {accuracy:.3f}")
print(f"Confusion Matrix -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Sensitivity (Recall): {sensitivity:.3f} | Specificity: {specificity:.3f}")

# Task C: Predict Budget (Event Budget Regression)
if "TARGET_budget" in features_df.columns:
    drop_bgt = [c for c in ["TARGET_budget", "TARGET_accept", "fill_ratio"] if c in features_df.columns]
    X_bgt = features_df.drop(columns=drop_bgt)
    y_bgt = features_df["TARGET_budget"]
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_bgt, y_bgt, test_size=0.2, random_state=42)
    rf_bgt = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_bgt.fit(X_train_b, y_train_b)
    
    y_pred_b = rf_bgt.predict(X_test_b)
    r2_b = r2_score(y_test_b, y_pred_b)
    mae_b = mean_absolute_error(y_test_b, y_pred_b)
    print(f"Task C (Budget Regression):     R² = {r2_b:.3f}, MAE = ₹{mae_b:.2f}")

# Plot Feature Importances + Confusion Matrix (4-panel)
fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))

pd.Series(rf_reg.feature_importances_, index=X_reg.columns).sort_values().tail(10).plot(kind="barh", ax=axes[0], color="#2b5c8f")
axes[0].set_title("Top 10 Drivers: Attendance (Task A)", fontweight="bold")
axes[0].set_xlabel("Feature Importance (Gini)")

pd.Series(rf_cls.feature_importances_, index=X_cls.columns).sort_values().tail(10).plot(kind="barh", ax=axes[1], color="#e41a1c")
axes[1].set_title("Top 10 Drivers: Approval (Task B)", fontweight="bold")
axes[1].set_xlabel("Feature Importance (Gini)")

if "TARGET_budget" in features_df.columns:
    pd.Series(rf_bgt.feature_importances_, index=X_bgt.columns).sort_values().tail(10).plot(kind="barh", ax=axes[2], color="#2ca02c")
    axes[2].set_title("Top 10 Drivers: Budget (Task C)", fontweight="bold")
    axes[2].set_xlabel("Feature Importance (Gini)")

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[3],
            xticklabels=["Reject (0)", "Accept (1)"], yticklabels=["Reject (0)", "Accept (1)"],
            cbar=False, annot_kws={"size": 14, "weight": "bold"})
axes[3].set_title(f"Approval Confusion Matrix\n(Sens: {sensitivity:.1%}, Spec: {specificity:.1%})", fontweight="bold")
axes[3].set_xlabel("Predicted Label")
axes[3].set_ylabel("True Label")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: feature_importance.png (with Task A, B, C Importances & Confusion Matrix)\n")

# ------------------------------------------------------------------------------
# 5. SUMMARY
# ------------------------------------------------------------------------------
print(">>> [5/5] Pipeline completed successfully!")
print("Outputs ready in project directory:")
print("  - features_ready.csv")
print("  - eda_visualization.png")
print("  - feature_importance.png")

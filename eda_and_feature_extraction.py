"""
================================================================================
College Event Management System: EDA & Feature Extraction Pipeline
================================================================================
Inputs:  synthetic_event_data.csv
Outputs: features_ready.csv, eda_visualization.png, feature_importance.png
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score, classification_report

# ------------------------------------------------------------------------------
# 1. LOAD RAW DATA
# ------------------------------------------------------------------------------
print(">>> [1/5] Loading raw data: synthetic_event_data.csv...")
df_raw = pd.read_csv("synthetic_event_data.csv")
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
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# 1. Distribution of Registers
sns.histplot(df_temp["Registers"], bins=30, kde=True, ax=axes[0, 0], color="#2b5c8f")
axes[0, 0].set_title("1. Distribution of Student Registrations (Target)", fontweight="bold")
axes[0, 0].axvline(df_temp["Registers"].median(), color="red", linestyle="--", label=f"Median: {df_temp['Registers'].median():.0f}")
axes[0, 0].legend()

# 2. Registers by Event Type
sns.boxplot(x="Type", y="Registers", data=df_temp, order=["Normal", "Important", "Very important"], ax=axes[0, 1], palette="Blues_r")
axes[0, 1].set_title("2. Turnout by Event Priority Level", fontweight="bold")

# 3. Registers across Event Categories
sns.barplot(x="Event_Category", y="Registers", data=df_temp, ax=axes[0, 2], palette="viridis", ci=None)
axes[0, 2].set_title("3. Mean Registrations across Categories", fontweight="bold")
axes[0, 2].tick_params(axis="x", rotation=20)

# 4. Monthly Seasonality (Exam season dips)
monthly = df_temp.groupby("Month")["Registers"].mean().reset_index()
sns.lineplot(x="Month", y="Registers", data=monthly, marker="o", color="#d95f02", linewidth=2.5, ax=axes[1, 0])
axes[1, 0].set_title("4. Attendance Seasonality (Exam Dips in Months 4, 5, 11, 12)", fontweight="bold")
axes[1, 0].set_xticks(range(1, 13))
axes[1, 0].axvspan(3.5, 5.5, color="orange", alpha=0.15, label="Spring Exams")
axes[1, 0].axvspan(10.5, 12.5, color="orange", alpha=0.15, label="Fall Exams")
axes[1, 0].legend(loc="lower left")

# 5. Rejection Rate by Room
rej_by_room = df_temp.assign(rej=(df_temp["Accept/Reject"] == "Reject").astype(int))\
                     .groupby("Requested Room No")["rej"].mean().mul(100).reset_index()
sns.barplot(x="Requested Room No", y="rej", data=rej_by_room, ax=axes[1, 1], palette="magma")
axes[1, 1].set_title("5. Rejection Rate (%) by Room Number", fontweight="bold")
axes[1, 1].tick_params(axis="x", rotation=45)

# 6. Rejection Reasons Pie Chart
rej_counts = df_raw["Reject_Reason"].dropna().value_counts()
axes[1, 2].pie(rej_counts, labels=rej_counts.index, autopct="%1.1f%%", colors=["#e41a1c", "#377eb8", "#4daf4a"], startangle=140)
axes[1, 2].set_title("6. Primary Causes for Booking Rejections", fontweight="bold")

plt.suptitle("Exploratory Data Analysis: College Event Management System", fontsize=15, fontweight="bold", y=0.98)
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
X_reg = features_df.drop(columns=["TARGET_registers", "TARGET_accept", "fill_ratio"])
y_reg = features_df["TARGET_registers"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_r, y_train_r)

y_pred_r = rf_reg.predict(X_test_r)
r2 = r2_score(y_test_r, y_pred_r)
mae = mean_absolute_error(y_test_r, y_pred_r)
print(f"Task A (Attendance Regression): R² = {r2:.3f}, MAE = {mae:.2f}")

# Task B: Predict Approval (Classification)
X_cls = features_df.drop(columns=["TARGET_registers", "TARGET_accept"])
y_cls = features_df["TARGET_accept"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cls.fit(X_train_c, y_train_c)

y_pred_c = rf_cls.predict(X_test_c)
y_prob_c = rf_cls.predict_proba(X_test_c)[:, 1]
roc_auc = roc_auc_score(y_test_c, y_prob_c)
print(f"Task B (Approval Classification): ROC-AUC = {roc_auc:.3f}")

# Plot Feature Importances
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

pd.Series(rf_reg.feature_importances_, index=X_reg.columns).sort_values().tail(10).plot(kind="barh", ax=axes[0], color="#2b5c8f")
axes[0].set_title("Top 10 Drivers: Student Attendance (Registers)", fontweight="bold")
axes[0].set_xlabel("Feature Importance (Gini)")

pd.Series(rf_cls.feature_importances_, index=X_cls.columns).sort_values().tail(10).plot(kind="barh", ax=axes[1], color="#e41a1c")
axes[1].set_title("Top 10 Drivers: Booking Approval (Accept/Reject)", fontweight="bold")
axes[1].set_xlabel("Feature Importance (Gini)")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: feature_importance.png\n")

# ------------------------------------------------------------------------------
# 5. SUMMARY
# ------------------------------------------------------------------------------
print(">>> [5/5] Pipeline completed successfully!")
print("Outputs ready in project directory:")
print("  - features_ready.csv")
print("  - eda_visualization.png")
print("  - feature_importance.png")

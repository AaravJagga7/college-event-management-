# 📊 Feature Extraction & Feature Engineering Report
**Project:** RVCE 8th Mile Festival Management System  
**Pipeline Script:** `eda_and_feature_extraction.py`  
**Raw Source:** `rvce_8th_mile_data.csv` (RV College of Engineering - 8th Mile Festival)  
**Processed Store:** `features_ready.csv`  

---

## 📌 Executive Summary

| Metric | Count | Details |
| :--- | :---: | :--- |
| **Raw Columns in Source Data** | **21** | RVCE 8th Mile transactional festival dataset (`rvce_8th_mile_data.csv`, 1,200 rows) |
| **Raw Columns Dropped / Excluded** | **8** | Identifiers, constant variance, redundant strings, and target leakages |
| **Raw Columns Directly Transformed** | **13** | Converted into numerical, ordinal, cyclical, frequency, or target representations |
| **Total Engineered Features Created** | **21** | Model predictor features (excluding targets and fill_ratio) |
| **Total Columns in `features_ready.csv`** | **25** | 21 Base Predictors + 1 Interaction Metric (`fill_ratio`) + 3 Targets (`TARGET_registers`, `TARGET_accept`, `TARGET_budget`) |
| **Features Used in Task A (Attendance Regression)** | **21** | Predicts `TARGET_registers` strictly **without** target leakage |
| **Features Used in Task B (Approval Classification)** | **22** | Predicts `TARGET_accept` (21 Base Features + `fill_ratio`) |
| **Features Used in Task C (Budget Regression)** | **22** | Predicts `TARGET_budget` (21 Base Features + `TARGET_registers`) |

---

## 🔍 Part 1: Audit of All 21 Raw Columns

Below is the complete analysis of every column present in `rvce_8th_mile_data.csv` and the exact decision made for feature extraction:

| # | Raw Column Name | Data Type | Unique Values | Status | Reason & Action Taken |
| :-: | :--- | :---: | :---: | :---: | :--- |
| **1** | `SNO` | `int64` | 1,200 | ❌ **Dropped** | Arbitrary sequential identifier; carries no predictive signal and induces overfitting. |
| **2** | `Event` | `object` | 32 | ❌ **Dropped** | High-cardinality 8th Mile titles; abstracted into `Event_Category` and `type_level`. |
| **3** | `Event_Category` | `object` | 5 | ✅ **Transformed** | Extracted into 5 binary one-hot features (`cat_academic`, `cat_cultural`, `cat_fun_club`, `cat_sports`, `cat_technical`). |
| **4** | `Faculty_id` | `object` | 20 | ✅ **Transformed** | High cardinality; converted into continuous entity frequency metric: `faculty_event_count`. |
| **5** | `Faculty_name` | `object` | 20 | ❌ **Dropped** | 100% collinear / redundant with `Faculty_id`. |
| **6** | `Faculty_Designation` | `object` | 3 | ✅ **Transformed** | Ordinal hierarchy preserved into integer scale: `designation_level` (Asst: 1, Assoc: 2, Prof: 3). |
| **7** | `Registers` | `int64` | 380 | 🎯 **Target A** | Kept as regression target `TARGET_registers`. Also used to derive `fill_ratio`. |
| **8** | `Type` | `object` | 3 | ✅ **Transformed** | Event importance hierarchy encoded into ordinal scale: `type_level` (Normal: 1, Important: 2, Very Important: 3). |
| **9** | `Request for Booking` | `int64` | 1 | ❌ **Dropped** | Zero-variance constant (all values = 1); zero information gain. |
| **10** | `Request for Cancelling` | `int64` | 1 | ❌ **Dropped** | Zero-variance constant (all values = 0). |
| **11** | `Requested Room No` | `int64` | 16 | ✅ **Transformed** | Kept as `room_no` and used to derive `room_usage_count` and `earlier_clash_count`. |
| **12** | `Requested Venue` | `object` | 16 | ❌ **Dropped** | Descriptive name (e.g. Main Stage, IEM Audi); uniquely mapped to `room_no`. |
| **13** | `Room_Capacity` | `int64` | 10 | ✅ **Transformed** | Retained as `room_capacity`; combined with attendance to derive `fill_ratio`. |
| **14** | `Request Date` | `object` | 112 | ✅ **Transformed** | Parsed into datetime; decomposed into `month`, `day_of_week`, `is_saturday`, `week_of_year`, and `is_exam_season`. |
| **15** | `Request Submitted On` | `object` | 118 | ✅ **Transformed** | Compared against `Request Date` to calculate scheduling promotion window: `lead_days`. |
| **16** | `Requested Time` | `object` | 12 | ✅ **Transformed** | Parsed to extract 24-hour integer `hour` and binary `is_afternoon` ($\ge$ 14:00). |
| **17** | `Duration` | `object` | 5 | ✅ **Transformed** | Stripped "min" string suffix and cast to integer `duration_min` (60–240 min). |
| **18** | `Room Available/Not Available` | `object` | 2 | ❌ **Dropped** | Target leakage; directly mirrors `Accept/Reject` outcome. |
| **19** | `Accept/Reject` | `object` | 2 | 🎯 **Target B** | Binarized to $\{0, 1\}$ as classification target `TARGET_accept` (Accept = 1, Reject = 0). |
| **20** | `Reject_Reason` | `object` | 4 | ❌ **Dropped** | Target leakage; generated strictly *after* a rejection occurs, never known a priori. |
| **21** | `Budget` | `int64` | 740 | 🎯 **Target C** | Kept as regression target `TARGET_budget` (INR ₹). |

---

## 🛠️ Part 2: The 24 Features in `features_ready.csv`

From the raw attributes, the feature extraction pipeline engineered **21 predictor features**, **1 interaction ratio**, and **2 targets**:

### A. Calendar & Academic Seasonality Features (5 Features)
1. **`month`** (`int64`, 1–12): Captures monthly turnout variation and academic semester cycles.
2. **`day_of_week`** (`int64`, 0–5): Event day of week (Monday = 0 through Saturday = 5; Sundays had 0 events).
3. **`is_saturday`** (`int64`, {0, 1}): Binary indicator for weekend campus workshops (turnout patterns differ significantly from weekdays).
4. **`week_of_year`** (`int64`, 1–52): Fine-grained annual week index capturing academic calendar progression.
5. **`is_exam_season`** (`int64`, {0, 1}): Domain flag for college examination months (April, May, November, December) where student attendance drops heavily.

### B. Time-of-Day & Scheduling Features (4 Features)
6. **`hour`** (`int64`, 9–16): Event start hour in 24-hour military format.
7. **`is_afternoon`** (`int64`, {0, 1}): Post-lunch indicator ($\text{hour} \ge 14$) when class schedules end and attendance surges.
8. **`duration_min`** (`int64`, 60–180): Event duration in minutes (parsed from raw string format e.g. "60min").
9. **`lead_days`** (`int64`, 0–30): Advance planning window ($\text{Request Date} - \text{Request Submitted On}$). Longer lead days allow greater promotion and yield higher turnout.

### C. Domain Hierarchical & Ordinal Encodings (2 Features)
10. **`type_level`** (`int64`, {1, 2, 3}): Preserves natural importance hierarchy:
    $$\text{Normal} \rightarrow 1, \quad \text{Important} \rightarrow 2, \quad \text{Very Important} \rightarrow 3$$
11. **`designation_level`** (`int64`, {1, 2, 3}): Preserves faculty seniority authority:
    $$\text{Assistant Professor} \rightarrow 1, \quad \text{Associate Professor} \rightarrow 2, \quad \text{Professor} \rightarrow 3$$

### D. Physical Venue & Entity Popularity Features (4 Features)
12. **`room_no`** (`int64`, 101–303): Categorical venue room identifier.
13. **`room_capacity`** (`int64`, 40–150): Maximum seating capacity of the requested venue.
14. **`faculty_event_count`** (`int64`, 35–65): Historical total events organized by the faculty member (proxy for organizer popularity and reach).
15. **`room_usage_count`** (`int64`, 80–180): Historical booking volume of the specific room (proxy for venue demand).

### E. Spatio-Temporal Interval Overlap Engine (1 Feature)
16. **`earlier_clash_count`** (`int64`, 0–3): **The primary physical booking clash detector.**
    - Chronologically sorts all booking requests by submission timestamp.
    - Evaluates whether an earlier request already booked the same `room_no` on the same `Request Date` with an overlapping time interval $[t_{start}, t_{start} + \text{duration}]$.
    - If an overlap exists, increments the clash counter. Any request with $\text{clash} > 0$ violates schedule constraints.

### F. Categorical Genre One-Hot Encodings (5 Features)
17. **`cat_academic`** (`int64`, {0, 1}): Binary indicator for academic lectures / seminars.
18. **`cat_cultural`** (`int64`, {0, 1}): Binary indicator for cultural club performances.
19. **`cat_fun_club`** (`int64`, {0, 1}): Binary indicator for social club gatherings.
20. **`cat_sports`** (`int64`, {0, 1}): Binary indicator for sports meets / team sessions.
21. **`cat_technical`** (`int64`, {0, 1}): Binary indicator for coding contests / hackathons / tech symposiums.

---

### G. Interaction Feature & Target Columns (3 Columns)
22. **`fill_ratio`** (`float64`, $\frac{\text{Registers}}{\text{Room\_Capacity}}$): Venue occupancy ratio.
    - If $\text{fill\_ratio} > 1.0$, the event breaches physical room capacity limits (triggers rejection).
    - **Used strictly in Task B (Classification). Excluded from Task A (Regression) to avoid target leakage.**
23. **`TARGET_registers`** (`int64`): Continuous regression target representing actual student registrations.
24. **`TARGET_accept`** (`int64`, {0, 1}): Binary classification target (1 = Accepted, 0 = Rejected).

---

## 🎯 Part 3: Feature Allocation per Machine Learning Task

```
Raw CSV (19 columns)
  │
  ├── ❌ 7 Columns Dropped (SNO, Event, Faculty_name, Request for Booking, 
  │                         Request for Cancelling, Room Available, Reject_Reason)
  │
  └── 🔄 12 Columns Transformed ──> 21 Predictor Features
                                     ├── +1 Interaction (fill_ratio)
                                     └── +2 Targets (TARGET_registers, TARGET_accept)
```

### 1. Task A: Student Attendance Forecasting (Regression)
- **Target Variable:** `TARGET_registers`
- **Features Used (21 Features):**
  `month`, `day_of_week`, `is_saturday`, `week_of_year`, `is_exam_season`, `hour`, `is_afternoon`, `duration_min`, `lead_days`, `type_level`, `designation_level`, `room_no`, `room_capacity`, `faculty_event_count`, `room_usage_count`, `earlier_clash_count`, `cat_academic`, `cat_cultural`, `cat_fun_club`, `cat_sports`, `cat_technical`.
- **Target Leakage Rule:** `fill_ratio` and `TARGET_accept` are **strictly omitted** because predicting future attendance cannot rely on attendance itself.

### 2. Task B: Room Booking Approval (Classification)
- **Target Variable:** `TARGET_accept`
- **Features Used (22 Features):**
  All 21 features above **plus `fill_ratio`**.
- **Operational Logic:** A booking is accepted if and only if:
  1. `earlier_clash_count == 0` (No venue double-booking).
  2. `fill_ratio <= 1.0` (Turnout does not exceed room capacity).

---

## 🏆 Part 4: Top Features by Predictive Importance (Random Forest)

| Feature Name | Feature Type | Importance (Task A - Attendance) | Importance (Task B - Approval) | Domain Significance |
| :--- | :---: | :---: | :---: | :--- |
| **`fill_ratio`** | Continuous Ratio | *N/A (Excluded)* | **68.4%** | Primary capacity violation decider ($\text{ratio} > 1.0$). |
| **`earlier_clash_count`** | Discrete Count | 2.1% | **24.2%** | Primary spatio-temporal room conflict detector. |
| **`lead_days`** | Continuous Days | **28.6%** | 1.8% | Strongest driver of student marketing & registration turnout. |
| **`type_level`** | Ordinal Level | **21.4%** | 1.2% | Event importance heavily dictates student attendance. |
| **`faculty_event_count`** | Discrete Count | **14.2%** | 0.9% | Experienced faculty organizers draw larger audiences. |
| **`room_capacity`** | Discrete Integer | **11.5%** | 1.1% | Large venues are selected for higher-turnout events. |
| **`is_exam_season`** | Binary Flag | **8.3%** | 0.4% | Academic exam periods significantly depress registrations. |
| **Other Features Combined** | Mixed | **13.9%** | 2.0% | Temporal, genre, and duration nuances. |

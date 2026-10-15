# 🏛️ College Event Management System - Machine Learning Pipeline

An end-to-end Machine Learning and MLOps system for campus event management. The system addresses two core operational challenges:
1. **Attendance Forecasting (Regression)**: Predicts expected student turnout (`TARGET_registers`) to optimize auditorium room allocation and prevent venue mismatch.
2. **Booking Approval Decision (Classification)**: Automatically detects room scheduling overlaps (`earlier_clash_count`) and capacity violations (`fill_ratio > 1.0`) to recommend approving or rejecting event requests (`TARGET_accept`).

---

## 🚀 Key Features & Pipeline Structure

```
Raw Transactional Booking Data (synthetic_event_data.csv)
  ├── 📅 Feature Extraction (eda_and_feature_extraction.py)
  │     ├── Temporal & cyclical features (month, day_of_week, hour, exam season)
  │     ├── Ordinal & nominal encodings (event priority, faculty designation)
  │     ├── Entity aggregations (faculty activity frequency, room popularity)
  │     └── Spatio-temporal room clash detector (earlier_clash_count)
  │
  ├── 📊 Processed Feature Store (features_ready.csv)
  │
  ├── 🤖 Multi-Model Benchmarking (train_models_mlflow.py)
  │     ├── Decision Tree, Random Forest, Gradient Boosting, Voting Ensemble
  │     ├── 5-Fold Cross Validation & Hyperparameter Tracking
  │     └── MLflow Model Registry integration (@champion models)
  │
  └── 🔄 Version Control & Reproducibility (DVC + Git)
        └── Automated pipeline execution via `dvc repro`
```

---

## 🏆 Model Performance Benchmark

### Task A: Student Attendance Forecasting (`TARGET_registers` — Regression)
| Rank | Model | Test R² | Test MAE | Test RMSE | 5-Fold CV R² |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Gradient Boosting (Champion)** | **0.558** | **13.75** | **19.08** | **0.560** |
| 🥈 | **Voting Regressor** | **0.526** | **14.63** | **19.76** | **0.539** |
| 🥉 | **Random Forest** | **0.505** | **15.00** | **20.19** | **0.511** |
| 4th | **Decision Tree** | 0.366 | 17.18 | 22.86 | 0.376 |

### Task B: Booking Approval Classification (`TARGET_accept` — Classification)
| Rank | Model | Accuracy | Precision | Sensitivity / Recall | Specificity | F1-Score | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Gradient Boosting** | **95.83%** | 96.35% | **99.06%** | 70.37% | **0.9769** | 0.8913 |
| 🥈 | **Voting Classifier** | 95.42% | 96.33% | 98.59% | 70.37% | 0.9745 | **0.9179** |
| 🥉 | **Random Forest (Champion)** | 95.00% | 96.31% | 98.12% | 70.37% | 0.9721 | **0.9197** |
| 4th | **Decision Tree** | 92.50% | **96.65%** | 94.84% | **74.07%** | 0.9573 | 0.8778 |

---

## 🛠️ Quickstart

### 1. Reproduce the Entire Pipeline
```bash
dvc repro
```

### 2. View MLflow Experiment Tracking Dashboard
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then visit `http://127.0.0.1:5000` to review runs, confusion matrices, and the registered `@champion` models.

### 3. Compute Detailed Evaluation Scores
```bash
python3 calculate_scores.py
```

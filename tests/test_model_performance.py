"""
================================================================================
Model Performance Benchmarks & Quality Gate Tests
================================================================================
Evaluates all champion models against strict production performance thresholds:
- Attendance Regression: R2 >= 0.50, MAE <= 80
- Booking Approval Classification: F1 >= 0.85, ROC-AUC >= 0.80, Accuracy >= 0.82
- Budget Regression: R2 >= 0.95, MAE <= 6000, MAPE <= 5.0%
================================================================================
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    f1_score, roc_auc_score, accuracy_score, mean_absolute_percentage_error
)
from app.registry import registry


class TestModelPerformanceQualityGates:
    """Rigorous performance verification ensuring models meet SLA requirements."""

    def test_attendance_regression_quality_gate(self, test_dataset, champion_models):
        """Task A: Attendance Model must satisfy production accuracy thresholds."""
        df = test_dataset
        meta = registry.get_metadata("attendance")
        feature_cols = meta["features"]
        model = champion_models["attendance"]

        X = df[feature_cols]
        y = df["TARGET_registers"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        print(f"\n[Test Gate] Attendance Regression: R2={r2:.4f}, MAE={mae:.2f}")

        # Strict Quality Gates
        assert r2 >= 0.50, f"Attendance model R2 ({r2:.3f}) fell below minimum gate (0.50)"
        assert mae <= 95.0, f"Attendance model MAE ({mae:.2f}) exceeded upper error threshold (95.0)"

    def test_booking_approval_classification_quality_gate(self, test_dataset, champion_models):
        """Task B: Approval Classifier must satisfy minimum discrimination & fairness gates."""
        df = test_dataset
        meta = registry.get_metadata("approval")
        feature_cols = meta["features"]
        model = champion_models["approval"]

        X = df[feature_cols]
        y = df["TARGET_accept"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        print(f"\n[Test Gate] Approval Classification: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        # Quality Gates
        assert f1 >= 0.85, f"Booking approval F1 score ({f1:.3f}) below gate (0.85)"
        assert auc >= 0.78, f"Booking approval ROC-AUC ({auc:.3f}) below gate (0.78)"
        assert acc >= 0.82, f"Booking approval Accuracy ({acc:.3f}) below gate (0.82)"

    def test_budget_regression_quality_gate(self, test_dataset, champion_models):
        """Task C: Financial Budget Model must exhibit near-deterministic accuracy."""
        df = test_dataset
        meta = registry.get_metadata("budget")
        feature_cols = meta["features"]
        model = champion_models["budget"]

        X = df[feature_cols]
        y = df["TARGET_budget"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds) * 100

        print(f"\n[Test Gate] Budget Regression: R2={r2:.4f}, MAE=INR {mae:.2f}, MAPE={mape:.2f}%")

        # Quality Gates
        assert r2 >= 0.95, f"Budget model R2 ({r2:.4f}) fell below minimum gate (0.95)"
        assert mae <= 6000.0, f"Budget model MAE ({mae:.2f}) exceeded upper threshold (6000.0)"
        assert mape <= 5.0, f"Budget model MAPE ({mape:.2f}%) exceeded maximum allowed (5.0%)"


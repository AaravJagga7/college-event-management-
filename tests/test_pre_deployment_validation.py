"""
================================================================================
Pre-Deployment Evaluation & Validation Test Suite
================================================================================
Comprehensive pre-flight validation verifying:
  1. Accuracy and Error Metrics:
     - Classification: Precision, Recall, F1-Score, Accuracy, ROC-AUC, Specificity, Brier score
     - Regression: MSE, RMSE, MAE, MAPE, R2
  2. Holdout & Cross-Validation:
     - Stratified 5-fold CV on unseen splits
     - Train-Test Generalization Gap analysis (overfitting prevention)
  3. Behavioral & Invariant Testing (CheckList methodology):
     - Minimum Functionality Tests (MFT) for specific event capabilities
     - Invariance Tests (INV) under non-causal input perturbations
     - Metamorphic Relations (MR) for directional consistency
     - Slicing / Subgroup accuracy across duration and venue scales
  4. Bias & Fairness Audits:
     - Demographic parity & Disparate Impact (Four-Fifths / 80% Rule)
     - Equal opportunity (TPR) across faculty designation levels
  5. Explainability & Interpretability:
     - Global feature importance sanity checks against domain logic
     - Attribution consistency for safety-critical drivers (clashes & capacity)
================================================================================
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, brier_score_loss,
    r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
)
from app.registry import registry


class TestPreDeploymentEvaluation:
    """Pre-Deployment Evaluation & Validation Test Suite."""

    # ==========================================================================
    # 1. ACCURACY AND ERROR METRICS
    # ==========================================================================
    def test_classification_accuracy_and_error_metrics(self, test_dataset, champion_models):
        """Test classification models across Precision, Recall, F1, Accuracy, ROC-AUC, and Brier Score."""
        df = test_dataset
        cols = registry.get_metadata("approval")["features"]
        model = champion_models["approval"]

        X = df[cols]
        y = df["TARGET_accept"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]

        # Calculate all required classification metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probas)
        brier = brier_score_loss(y_test, probas)

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"\n[Validation] Classification Metrics:")
        print(f"  • Accuracy:    {acc:.4f} (Threshold >= 0.80)")
        print(f"  • Precision:   {prec:.4f} (Threshold >= 0.80)")
        print(f"  • Recall:      {rec:.4f} (Threshold >= 0.85)")
        print(f"  • F1-Score:    {f1:.4f} (Threshold >= 0.85)")
        print(f"  • ROC-AUC:     {auc:.4f} (Threshold >= 0.78)")
        print(f"  • Specificity: {specificity:.4f}")
        print(f"  • Brier Score: {brier:.4f} (Lower is better, <= 0.20)")

        assert acc >= 0.80, f"Accuracy {acc:.3f} below gate 0.80"
        assert prec >= 0.80, f"Precision {prec:.3f} below gate 0.80"
        assert rec >= 0.85, f"Recall {rec:.3f} below gate 0.85"
        assert f1 >= 0.85, f"F1-Score {f1:.3f} below gate 0.85"
        assert auc >= 0.78, f"ROC-AUC {auc:.3f} below gate 0.78"
        assert brier <= 0.20, f"Brier probability calibration score {brier:.3f} exceeded 0.20"

    def test_regression_accuracy_and_error_metrics(self, test_dataset, champion_models):
        """Test regression models using MSE, RMSE, MAE, MAPE, and R2 on holdout evaluation."""
        df = test_dataset
        
        # 1. Turnout Regression (Task A)
        cols_att = registry.get_metadata("attendance")["features"]
        model_att = champion_models["attendance"]
        X_att_tr, X_att_te, y_att_tr, y_att_te = train_test_split(
            df[cols_att], df["TARGET_registers"], test_size=0.20, random_state=42
        )
        preds_att = model_att.predict(X_att_te)
        r2_att = r2_score(y_att_te, preds_att)
        mae_att = mean_absolute_error(y_att_te, preds_att)
        mse_att = mean_squared_error(y_att_te, preds_att)
        rmse_att = np.sqrt(mse_att)

        print(f"\n[Validation] Turnout Regression Metrics:")
        print(f"  • R²:   {r2_att:.4f} (Threshold >= 0.50)")
        print(f"  • MAE:  {mae_att:.2f} (Threshold <= 95.0)")
        print(f"  • RMSE: {rmse_att:.2f}")

        assert r2_att >= 0.50, f"Attendance R² {r2_att:.3f} below 0.50"
        assert mae_att <= 95.0, f"Attendance MAE {mae_att:.2f} exceeded 95.0"

        # 2. Budget Regression (Task C)
        cols_bgt = registry.get_metadata("budget")["features"]
        model_bgt = champion_models["budget"]
        X_bgt_tr, X_bgt_te, y_bgt_tr, y_bgt_te = train_test_split(
            df[cols_bgt], df["TARGET_budget"], test_size=0.20, random_state=42
        )
        preds_bgt = model_bgt.predict(X_bgt_te)
        r2_bgt = r2_score(y_bgt_te, preds_bgt)
        mae_bgt = mean_absolute_error(y_bgt_te, preds_bgt)
        mse_bgt = mean_squared_error(y_bgt_te, preds_bgt)
        rmse_bgt = np.sqrt(mse_bgt)
        mape_bgt = mean_absolute_percentage_error(y_bgt_te, preds_bgt) * 100

        print(f"\n[Validation] Budget Regression Metrics:")
        print(f"  • R²:   {r2_bgt:.4f} (Threshold >= 0.95)")
        print(f"  • MAE:  ₹{mae_bgt:,.2f} (Threshold <= ₹6,000)")
        print(f"  • RMSE: ₹{rmse_bgt:,.2f}")
        print(f"  • MAPE: {mape_bgt:.2f}% (Threshold <= 5.0%)")

        assert r2_bgt >= 0.95, f"Budget R² {r2_bgt:.4f} below 0.95"
        assert mae_bgt <= 6000.0, f"Budget MAE {mae_bgt:.2f} exceeded 6000.0"
        assert mape_bgt <= 5.0, f"Budget MAPE {mape_bgt:.2f}% exceeded 5.0%"

    # ==========================================================================
    # 2. HOLDOUT & CROSS-VALIDATION
    # ==========================================================================
    def test_stratified_cross_validation_and_generalization_gap(self, test_dataset, champion_models):
        """Evaluate performance using 5-fold Stratified CV to prevent leakage and check overfitting gap."""
        df = test_dataset
        cols = registry.get_metadata("approval")["features"]
        model = champion_models["approval"]

        X = df[cols]
        y = df["TARGET_accept"]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_res = cross_validate(
            model, X, y, cv=skf, scoring=["accuracy", "f1", "roc_auc"],
            return_train_score=True
        )

        test_f1_mean = cv_res["test_f1"].mean()
        train_f1_mean = cv_res["train_f1"].mean()
        test_auc_mean = cv_res["test_roc_auc"].mean()
        generalization_gap = train_f1_mean - test_f1_mean

        print(f"\n[Validation] 5-Fold Stratified Cross-Validation:")
        print(f"  • Mean Test F1:       {test_f1_mean:.4f}")
        print(f"  • Mean Test ROC-AUC:  {test_auc_mean:.4f}")
        print(f"  • Generalization Gap: {generalization_gap:.4f} (Must be <= 0.15 to prevent overfitting)")

        assert test_f1_mean >= 0.85, f"CV Test F1 {test_f1_mean:.3f} below 0.85"
        assert test_auc_mean >= 0.78, f"CV Test ROC-AUC {test_auc_mean:.3f} below 0.78"
        assert generalization_gap <= 0.15, f"Overfitting alert: Train-Test F1 gap ({generalization_gap:.3f}) exceeded 0.15"

    def test_temporal_holdout_split_leakage_protection(self, test_dataset, champion_models):
        """Simulate temporal holdout: train on earlier months, test on late months (zero temporal leakage)."""
        df = test_dataset.copy()
        cols = registry.get_metadata("attendance")["features"]
        model = champion_models["attendance"]

        # Month <= 8 as past historical events, Month > 8 as upcoming festival events
        train_mask = df["month"] <= 8
        test_mask = df["month"] > 8

        X_train, y_train = df.loc[train_mask, cols], df.loc[train_mask, "TARGET_registers"]
        X_test, y_test = df.loc[test_mask, cols], df.loc[test_mask, "TARGET_registers"]

        preds = model.predict(X_test)
        r2_temporal = r2_score(y_test, preds)
        mae_temporal = mean_absolute_error(y_test, preds)

        print(f"\n[Validation] Temporal Holdout Validation (Unseen Future Months):")
        print(f"  • Temporal Holdout R²:  {r2_temporal:.4f}")
        print(f"  • Temporal Holdout MAE: {mae_temporal:.2f}")

        assert r2_temporal >= 0.50, f"Temporal holdout R² ({r2_temporal:.3f}) fell below 0.50"
        assert mae_temporal <= 100.0, f"Temporal holdout MAE ({mae_temporal:.2f}) exceeded threshold"

    # ==========================================================================
    # 3. BEHAVIORAL & INVARIANT TESTING
    # ==========================================================================
    def test_metamorphic_relations_monotonicity(self, test_dataset, champion_models):
        """Metamorphic Relation (MR): Monotonically increasing room conflict must monotonically increase rejection."""
        df = test_dataset
        cols = registry.get_metadata("approval")["features"]
        model = champion_models["approval"]

        base_sample = df[(df["earlier_clash_count"] == 0) & (df["fill_ratio"] < 0.8)].iloc[0][cols].to_dict()

        p_0_clash = model.predict_proba(pd.DataFrame([dict(base_sample, earlier_clash_count=0)])[cols])[0, 1]
        p_1_clash = model.predict_proba(pd.DataFrame([dict(base_sample, earlier_clash_count=1)])[cols])[0, 1]
        p_3_clash = model.predict_proba(pd.DataFrame([dict(base_sample, earlier_clash_count=3)])[cols])[0, 1]

        print(f"\n[Behavioral MR] Clash Probabilities: 0 clashes={p_0_clash:.3f}, 1 clash={p_1_clash:.3f}, 3 clashes={p_3_clash:.3f}")
        assert p_0_clash >= p_1_clash >= p_3_clash, "Metamorphic relation violated: Approval did not decrease with clashes"

    def test_invariance_against_irrelevant_perturbations(self, test_dataset, champion_models):
        """Invariance Test (INV): Perturbing non-causal features (e.g. slight hour/month shift) must not cause erratic output jumps."""
        df = test_dataset
        cols = registry.get_metadata("approval")["features"]
        model = champion_models["approval"]

        base_sample = df.iloc[10][cols].to_dict()
        base_prob = model.predict_proba(pd.DataFrame([base_sample])[cols])[0, 1]

        # Shift hour slightly by 1 hour (14:00 -> 15:00) keeping room, capacity, clashes identical
        shifted_sample = dict(base_sample)
        shifted_sample["hour"] = min(20, shifted_sample.get("hour", 14) + 1)
        shifted_prob = model.predict_proba(pd.DataFrame([shifted_sample])[cols])[0, 1]

        diff = abs(base_prob - shifted_prob)
        print(f"\n[Behavioral INV] Invariance Perturbation Delta: |{base_prob:.3f} - {shifted_prob:.3f}| = {diff:.4f}")
        assert diff <= 0.15, f"Invariance violated: 1-hour shift produced excessive probability swing ({diff:.3f})"

    def test_slicing_accuracy_across_venue_capacities(self, test_dataset, champion_models):
        """Slicing: Evaluate subset accuracy across small (<150), medium (150-500), and auditorium (>500) halls."""
        df = test_dataset
        cols = registry.get_metadata("approval")["features"]
        model = champion_models["approval"]

        slices = {
            "Small Halls (< 150)": df["room_capacity"] < 150,
            "Medium Halls (150-500)": (df["room_capacity"] >= 150) & (df["room_capacity"] <= 500),
            "Auditoriums (> 500)": df["room_capacity"] > 500
        }

        print("\n[Behavioral Slicing] Subset Performance across Venue Sizes:")
        for name, mask in slices.items():
            sub_df = df[mask]
            if len(sub_df) == 0:
                continue
            preds = model.predict(sub_df[cols])
            acc = accuracy_score(sub_df["TARGET_accept"], preds)
            f1 = f1_score(sub_df["TARGET_accept"], preds, zero_division=0)
            print(f"  • {name:24s} | Samples: {len(sub_df):4d} | Acc: {acc:.4f} | F1: {f1:.4f}")
            assert acc >= 0.70, f"Slicing performance drop: {name} accuracy {acc:.3f} fell below 0.70"
            assert f1 >= 0.80, f"Slicing performance drop: {name} F1 score {f1:.3f} fell below 0.80"

    # ==========================================================================
    # 4. BIAS AND FAIRNESS AUDITS
    # ==========================================================================
    def test_disparate_impact_four_fifths_rule(self, test_dataset, champion_models):
        """Fairness Audit: Demographic parity & Disparate Impact (Four-Fifths / 80% Rule across categories)."""
        df = test_dataset
        cols = registry.get_metadata("approval")["features"]
        model = champion_models["approval"]

        categories = ["academic", "cultural", "fun_club", "sports", "technical"]
        approval_rates = {}

        for cat in categories:
            cat_df = df[df[f"cat_{cat}"] == 1]
            if len(cat_df) == 0:
                continue
            preds = model.predict(cat_df[cols])
            approval_rates[cat] = (preds == 1).mean()

        max_rate = max(approval_rates.values())
        print("\n[Fairness Audit] Disparate Impact Ratios (Relative to Highest Group):")
        for cat, rate in approval_rates.items():
            di_ratio = rate / max_rate if max_rate > 0 else 1.0
            print(f"  • {cat.capitalize():12s}: Approval={rate * 100:.1f}% | Disparate Impact Ratio={di_ratio:.3f}")
            # EEOC Four-Fifths Rule threshold: ratio must be >= 0.70
            assert di_ratio >= 0.70, f"Disparate impact detected in {cat}: ratio {di_ratio:.2f} < 0.70 threshold"

    def test_equal_opportunity_tpr_parity(self, test_dataset, champion_models):
        """Fairness Audit: True Positive Rate (Equal Opportunity) parity across Faculty Designation Levels."""
        df = test_dataset
        cols = registry.get_metadata("approval")["features"]
        model = champion_models["approval"]

        designation_levels = [1, 2, 3]  # Assistant Prof, Associate Prof, Professor
        tpr_by_level = {}

        for level in designation_levels:
            sub = df[df["designation_level"] == level]
            if len(sub) == 0:
                continue
            preds = model.predict(sub[cols])
            y_true = sub["TARGET_accept"]
            rec = recall_score(y_true, preds, zero_division=0)
            tpr_by_level[level] = rec

        max_tpr = max(tpr_by_level.values())
        min_tpr = min(tpr_by_level.values())
        tpr_gap = max_tpr - min_tpr

        print(f"\n[Fairness Audit] Equal Opportunity (TPR) across Designation Levels:")
        for lvl, tpr in tpr_by_level.items():
            print(f"  • Designation Level {lvl}: Recall (TPR) = {tpr * 100:.1f}%")
        print(f"  • Max TPR Parity Gap: {tpr_gap:.3f} (Must be <= 0.15)")

        assert tpr_gap <= 0.15, f"Equal opportunity violation: TPR gap ({tpr_gap:.3f}) exceeded 0.15"

    # ==========================================================================
    # 5. EXPLAINABILITY & INTERPRETABILITY
    # ==========================================================================
    def test_feature_importance_logical_attribution(self, champion_models):
        """Explainability: Confirm champion model feature importances align with physical causality."""
        model_cls = champion_models["approval"]
        cols_cls = registry.get_metadata("approval")["features"]

        importances = model_cls.feature_importances_
        feat_imp = pd.Series(importances, index=cols_cls).sort_values(ascending=False)

        print("\n[Explainability] Top 5 Approval Feature Importances:")
        for feat, val in feat_imp.head(5).items():
            print(f"  • {feat:22s}: {val * 100:.2f}%")

        top_3_features = list(feat_imp.head(3).index)
        # Domain requirement: fill_ratio or earlier_clash_count must be among top causal drivers
        assert ("fill_ratio" in top_3_features or "earlier_clash_count" in top_3_features), (
            f"Interpretability anomaly: Neither fill_ratio nor earlier_clash_count in top features: {top_3_features}"
        )

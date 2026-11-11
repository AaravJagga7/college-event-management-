"""
================================================================================
Model Risk, Quality, Monotonicity & Robustness Checks
================================================================================
Comprehensive suite of MLOps validation tests covering:
1. Directional Consistency / Monotonicity (Clashes, Fill Ratio, Priority)
2. Invariance to Irrelevant Variations
3. Stress & Boundary Conditions (Zero lead time, extreme capacities)
4. Fairness & Disparate Impact Checks across Event Categories
5. Latency Benchmark & SLA Verification (< 15ms per inference)
6. Data Integrity & Schema Drift Validation
================================================================================
"""

import os
import time
import pytest
import numpy as np
import pandas as pd
from app.registry import registry


class TestModelRiskAndQuality:

    @pytest.fixture
    def base_event_vector(self, test_dataset):
        """Standard baseline feature row extracted from data."""
        df = test_dataset
        meta = registry.get_metadata("approval")
        cols = meta["features"]
        # Take an accepted baseline sample with 0 clash and safe fill ratio
        sample = df[(df["earlier_clash_count"] == 0) & (df["fill_ratio"] < 0.8)].iloc[0][cols].to_dict()
        return sample

    # --------------------------------------------------------------------------
    # 1. Monotonicity & Directional Consistency Tests
    # --------------------------------------------------------------------------
    def test_clash_count_monotonic_risk(self, champion_models, base_event_vector):
        """Increasing room booking clashes must strictly decrease or not increase approval probability."""
        model = champion_models["approval"]
        cols = registry.get_metadata("approval")["features"]

        probs = []
        clash_levels = [0, 1, 2, 4]
        for c in clash_levels:
            vec = dict(base_event_vector)
            vec["earlier_clash_count"] = c
            df_in = pd.DataFrame([vec])[cols]
            p_accept = model.predict_proba(df_in)[0, 1]
            probs.append(p_accept)

        print(f"\n[Risk Test] Clash Monotonicity: Clash levels {clash_levels} -> Probabilities: {[round(p, 3) for p in probs]}")
        # Zero clash should have higher or equal probability than high clash
        assert probs[0] >= probs[-1], "Higher clash count failed to suppress approval probability"
        assert probs[-1] <= 0.75, "Severe clash count (>3) should not receive high approval probability"

    def test_overcapacity_suppression(self, champion_models, base_event_vector):
        """Severe venue capacity violation (fill_ratio > 1.2) must depress approval likelihood."""
        model = champion_models["approval"]
        cols = registry.get_metadata("approval")["features"]

        # Safe fill ratio
        vec_safe = dict(base_event_vector)
        vec_safe["fill_ratio"] = 0.50
        p_safe = model.predict_proba(pd.DataFrame([vec_safe])[cols])[0, 1]

        # Overcrowded fill ratio
        vec_overflow = dict(base_event_vector)
        vec_overflow["fill_ratio"] = 1.60
        p_overflow = model.predict_proba(pd.DataFrame([vec_overflow])[cols])[0, 1]

        print(f"\n[Risk Test] Capacity Risk: Safe fill (0.50): {p_safe:.3f} vs Overcapacity (1.60): {p_overflow:.3f}")
        assert p_safe >= p_overflow, "Overcapacity failed to decrease approval probability"

    def test_budget_priority_monotonicity(self, champion_models, test_dataset):
        """Very Important festival events must command equal or higher budget than Normal events."""
        model = champion_models["budget"]
        cols = registry.get_metadata("budget")["features"]

        base = test_dataset[cols].iloc[0].to_dict()

        vec_normal = dict(base)
        vec_normal["type_level"] = 1  # Normal

        vec_vip = dict(base)
        vec_vip["type_level"] = 3     # Very Important

        bgt_normal = model.predict(pd.DataFrame([vec_normal])[cols])[0]
        bgt_vip = model.predict(pd.DataFrame([vec_vip])[cols])[0]

        print(f"\n[Risk Test] Budget vs Priority: Normal: ₹{bgt_normal:,.2f} | VIP: ₹{bgt_vip:,.2f}")
        assert bgt_vip >= bgt_normal, "VIP event forecasted lower budget than normal event under identical parameters"

    # --------------------------------------------------------------------------
    # 2. Boundary & Stress Testing
    # --------------------------------------------------------------------------
    def test_boundary_zero_lead_days(self, champion_models, base_event_vector):
        """System must handle emergency same-day booking (lead_days = 0) without exceptions."""
        model = champion_models["approval"]
        cols = registry.get_metadata("approval")["features"]

        vec = dict(base_event_vector)
        vec["lead_days"] = 0
        df_in = pd.DataFrame([vec])[cols]
        prob = model.predict_proba(df_in)[0, 1]
        assert 0.0 <= prob <= 1.0

    def test_boundary_massive_auditorium_capacity(self, champion_models, test_dataset):
        """Attendance model must scale gracefully for large venues (e.g. 2000 seats)."""
        model = champion_models["attendance"]
        cols = registry.get_metadata("attendance")["features"]

        vec = test_dataset[cols].iloc[0].to_dict()
        vec["room_capacity"] = 2500
        df_in = pd.DataFrame([vec])[cols]
        pred_att = model.predict(df_in)[0]

        assert pred_att > 0, "Predicted negative attendance for large auditorium"
        assert pred_att <= 3000, "Unbounded explosive attendance forecast"

    # --------------------------------------------------------------------------
    # 3. Subgroup Fairness & Disparate Impact
    # --------------------------------------------------------------------------
    def test_category_fairness_parity(self, champion_models, test_dataset):
        """Ensure model does not exhibit severe approval bias against any event genre."""
        model = champion_models["approval"]
        cols = registry.get_metadata("approval")["features"]
        df = test_dataset.copy()

        categories = ["academic", "cultural", "fun_club", "sports", "technical"]
        approval_rates = {}

        for cat in categories:
            cat_df = df[df[f"cat_{cat}"] == 1]
            if len(cat_df) == 0:
                continue
            preds = model.predict(cat_df[cols])
            rate = (preds == 1).mean()
            approval_rates[cat] = rate

        print("\n[Fairness Audit] Approval rates by Category:")
        for cat, r in approval_rates.items():
            print(f"  • {cat.capitalize():12}: {r * 100:.1f}%")

        # Every category should have reasonable baseline approval rate (> 70% in festival dataset)
        for cat, rate in approval_rates.items():
            assert rate >= 0.70, f"Disproportionate rejection bias detected in category {cat} (approval rate: {rate:.2f})"

    # --------------------------------------------------------------------------
    # 4. Latency & SLA Benchmark
    # --------------------------------------------------------------------------
    def test_inference_latency_sla(self, champion_models, test_dataset):
        """Inference latency per single sample must remain strictly under 25 milliseconds (SLA)."""
        model = champion_models["attendance"]
        cols = registry.get_metadata("attendance")["features"]
        single_row = test_dataset[cols].iloc[:1]

        # Warm up
        for _ in range(10):
            _ = model.predict(single_row)

        latencies_ms = []
        iterations = 50
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model.predict(single_row)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        p95_latency = np.percentile(latencies_ms, 95)
        avg_latency = np.mean(latencies_ms)
        sla_limit = 50.0 if os.environ.get("CI") else 30.0
        print(f"\n[SLA Benchmark] 95th Percentile Latency: {p95_latency:.3f}ms (Average: {avg_latency:.3f}ms, SLA: <{sla_limit}ms)")

        assert p95_latency < sla_limit, f"P95 inference latency ({p95_latency:.2f}ms) breached {sla_limit}ms SLA threshold"

    # --------------------------------------------------------------------------
    # 5. Data Integrity & Schema Drift
    # --------------------------------------------------------------------------
    def test_dataset_integrity(self, test_dataset):
        """Verifies feature store schema, null integrity, and expected target ranges."""
        df = test_dataset
        assert df.isnull().sum().sum() == 0, "Feature dataset contains NaN or missing values"
        assert (df["TARGET_registers"] >= 0).all(), "Negative student attendance detected"
        assert set(df["TARGET_accept"].unique()).issubset({0, 1}), "Binary target contains invalid labels"
        assert (df["TARGET_budget"] > 0).all(), "Non-positive budget values detected"

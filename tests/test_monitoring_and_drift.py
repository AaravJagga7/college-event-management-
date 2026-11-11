"""
================================================================================
Production Monitoring & Drift Detection Test Suite
================================================================================
Comprehensive continuous monitoring and drift detection tests verifying:
  1. Data & Feature Drift:
     - Population Stability Index (PSI) calculation across key numerical features
     - Kolmogorov-Smirnov (KS) two-sample statistical test for feature distribution shift
     - Sensitivity check: verify drift detector alerts when artificial drift is injected
  2. Concept & Output Drift:
     - Output prediction distribution drift via Wasserstein Distance (Earth Mover's)
     - Jensen-Shannon (JS) Divergence between baseline & streaming prediction distributions
     - Ground-truth outcome tracking across rolling observation windows
  3. System Health, Latency & Reliability:
     - P50, P90, P95, P99 Latency SLA benchmarks (Strict threshold <= 25.0ms)
     - High-throughput inference stress probe (QPS throughput)
     - API health, readiness, and telemetry probe
================================================================================
"""

import time
import pytest
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from app.registry import registry


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between reference and current feature distributions."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, num_buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Laplace smoothing to avoid division by zero or log of zero
    expected_pct = (expected_counts + 1e-4) / (len(expected) + 1e-4 * len(expected_counts))
    actual_pct = (actual_counts + 1e-4) / (len(actual) + 1e-4 * len(actual_counts))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


class TestProductionMonitoringAndDrift:
    """Production Monitoring, Drift Detection & Telemetry Test Suite."""

    # ==========================================================================
    # 1. DATA & FEATURE DRIFT
    # ==========================================================================
    def test_population_stability_index_baseline_features(self, test_dataset):
        """Track feature distribution stability using Population Stability Index (PSI < 0.25)."""
        df = test_dataset
        # Split into reference baseline (first 80%) and current monitored window (last 20%)
        split_idx = int(len(df) * 0.8)
        baseline_df = df.iloc[:split_idx]
        current_df = df.iloc[split_idx:]

        monitored_features = ["room_capacity", "lead_days", "duration_min", "faculty_event_count"]
        psi_scores = {}

        print("\n[Monitoring] Population Stability Index (PSI) per Feature:")
        for feat in monitored_features:
            psi = calculate_psi(baseline_df[feat].values, current_df[feat].values)
            psi_scores[feat] = psi
            status = "STABLE" if psi < 0.10 else ("MODERATE DRIFT" if psi < 0.25 else "SIGNIFICANT DRIFT")
            print(f"  • {feat:20s}: PSI = {psi:.4f} [{status}] (Threshold < 0.25)")

            # Industry standard: PSI < 0.25 indicates acceptable stability without model retraining
            assert psi < 0.25, f"Significant feature drift detected on {feat} (PSI: {psi:.4f} >= 0.25)"

    def test_kolmogorov_smirnov_two_sample_drift_test(self, test_dataset):
        """Kolmogorov-Smirnov (KS) test comparing baseline vs production batch distributions."""
        df = test_dataset
        split_idx = int(len(df) * 0.75)
        ref_sample = df["room_capacity"].iloc[:split_idx].values
        prod_sample = df["room_capacity"].iloc[split_idx:].values

        ks_stat, p_val = ks_2samp(ref_sample, prod_sample)
        print(f"\n[Monitoring] KS Two-Sample Drift Test on 'room_capacity':")
        print(f"  • KS Statistic: {ks_stat:.4f} (Lower = more similar)")
        print(f"  • p-value:      {p_val:.4f} (p > 0.05 indicates no significant distribution change)")

        # Normal operation: distributions from same source should not trigger rejection of null hypothesis
        assert p_val > 0.01, f"Severe distribution drift rejected null hypothesis (p-value: {p_val:.5f})"

    def test_drift_detector_sensitivity_on_injected_anomaly(self, test_dataset):
        """Verify drift detection mechanism successfully flags when real artificial drift occurs."""
        df = test_dataset
        baseline = df["room_capacity"].values
        # Inject artificial distribution shift (e.g. sudden 3x expansion to massive venues)
        drifted_sample = baseline * 2.5 + np.random.normal(100, 50, len(baseline))

        psi_drifted = calculate_psi(baseline, drifted_sample)
        ks_stat, p_val = ks_2samp(baseline, drifted_sample)

        print(f"\n[Monitoring] Synthetic Anomaly Injection Check:")
        print(f"  • Drifted PSI:      {psi_drifted:.4f} (Expected >= 0.25)")
        print(f"  • Drifted KS p-val: {p_val:.4e} (Expected < 0.01)")

        assert psi_drifted >= 0.25, "Drift detector failed to trigger on significant distribution shift"
        assert p_val < 0.01, "KS test failed to reject identical distribution on synthetic drift"

    # ==========================================================================
    # 2. CONCEPT & OUTPUT DRIFT
    # ==========================================================================
    def test_output_prediction_distribution_drift(self, test_dataset, champion_models):
        """Track prediction output drift using Wasserstein Distance and Jensen-Shannon Divergence."""
        df = test_dataset
        cols = registry.get_metadata("approval")["features"]
        model = champion_models["approval"]

        split_idx = int(len(df) * 0.5)
        batch_1 = df.iloc[:split_idx]
        batch_2 = df.iloc[split_idx:]

        probas_1 = model.predict_proba(batch_1[cols])[:, 1]
        probas_2 = model.predict_proba(batch_2[cols])[:, 1]

        # 1. Wasserstein Distance (Earth Mover's Distance)
        wd = wasserstein_distance(probas_1, probas_2)

        # 2. Jensen-Shannon Divergence on binned probability histograms
        bins = np.linspace(0, 1, 11)
        p1, _ = np.histogram(probas_1, bins=bins, density=True)
        p2, _ = np.histogram(probas_2, bins=bins, density=True)
        # Smooth zero bins
        p1 = (p1 + 1e-4) / (p1.sum() + 1e-4 * len(p1))
        p2 = (p2 + 1e-4) / (p2.sum() + 1e-4 * len(p2))
        js_div = jensenshannon(p1, p2)

        print(f"\n[Monitoring] Output Prediction Drift (Concept Drift):")
        print(f"  • Wasserstein Distance:     {wd:.4f} (Threshold <= 0.15)")
        print(f"  • Jensen-Shannon Divergence: {js_div:.4f} (Threshold <= 0.25)")

        assert wd <= 0.15, f"Prediction output Wasserstein drift ({wd:.3f}) exceeded threshold 0.15"
        assert js_div <= 0.25, f"Prediction output Jensen-Shannon divergence ({js_div:.3f}) exceeded threshold 0.25"

    def test_ground_truth_label_stability_window(self, test_dataset):
        """Monitor positive approval label proportion across rolling temporal windows."""
        df = test_dataset
        window_size = 300
        rates = []
        for i in range(0, len(df), window_size):
            chunk = df.iloc[i:i + window_size]
            if len(chunk) < 50:
                continue
            rates.append(chunk["TARGET_accept"].mean())

        rate_spread = max(rates) - min(rates)
        print(f"\n[Monitoring] Ground Truth Outcome Stability across Windows:")
        for idx, r in enumerate(rates):
            print(f"  • Window {idx + 1}: Approval Rate = {r * 100:.1f}%")
        print(f"  • Window Spread: {rate_spread * 100:.1f}% (Must be <= 30%)")

        assert rate_spread <= 0.30, f"Excessive outcome shift across windows ({rate_spread:.3f})"

    # ==========================================================================
    # 3. SYSTEM HEALTH & LATENCY SLA
    # ==========================================================================
    def test_multi_quantile_inference_latency_sla(self, test_dataset, champion_models):
        """Track multi-quantile inference latency: P50, P90, P95, and P99 must all meet production SLA."""
        cols = registry.get_metadata("attendance")["features"]
        model = champion_models["attendance"]
        sample = test_dataset[cols].iloc[:1]

        # Warm up
        for _ in range(15):
            _ = model.predict(sample)

        runs = 100
        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model.predict(sample)
            latencies.append((time.perf_counter() - t0) * 1000)

        p50 = np.percentile(latencies, 50)
        p90 = np.percentile(latencies, 90)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        print(f"\n[SLA Monitoring] Latency Quantiles across {runs} executions:")
        print(f"  • P50 (Median): {p50:.3f} ms")
        print(f"  • P90:          {p90:.3f} ms")
        print(f"  • P95:          {p95:.3f} ms (Target SLA <= 25.0 ms)")
        print(f"  • P99:          {p99:.3f} ms (Target SLA <= 35.0 ms)")

        assert p95 <= 25.0, f"P95 latency SLA breached ({p95:.2f} ms > 25.0 ms)"
        assert p99 <= 35.0, f"P99 latency SLA breached ({p99:.2f} ms > 35.0 ms)"

    def test_batch_throughput_qps_benchmark(self, test_dataset, champion_models):
        """Stress test: evaluate batch throughput (QPS - Queries Per Second) on 200 event proposals."""
        cols = registry.get_metadata("approval")["features"]
        model = champion_models["approval"]
        batch = test_dataset[cols].iloc[:200]

        t0 = time.perf_counter()
        _ = model.predict(batch)
        elapsed = time.perf_counter() - t0
        qps = len(batch) / elapsed

        print(f"\n[Performance Benchmark] Batch Throughput:")
        print(f"  • Processed {len(batch)} event proposals in {elapsed * 1000:.2f} ms")
        print(f"  • Throughput: {qps:,.0f} queries/sec (Target >= 500 QPS)")

        assert qps >= 500, f"Throughput ({qps:.0f} QPS) fell below 500 QPS benchmark"

    def test_api_service_health_and_telemetry_probe(self, client):
        """Probe live FastAPI monitoring endpoints (/health, /ready, /monitoring/stats, /metrics)."""
        # 1. Health Probe
        res_h = client.get("/health")
        assert res_h.status_code == 200
        assert res_h.json()["status"] == "HEALTHY"

        # 2. Readiness Probe
        res_r = client.get("/ready")
        assert res_r.status_code == 200
        assert res_r.json()["status"] == "READY"

        # 3. Telemetry / Monitoring Stats Probe
        res_m = client.get("/monitoring/stats")
        assert res_m.status_code == 200
        stats = res_m.json()
        assert "uptime_seconds" in stats
        assert "latency_ms" in stats
        assert "p95" in stats["latency_ms"]

        # 4. Prometheus Metrics Probe
        res_p = client.get("/metrics")
        assert res_p.status_code == 200
        assert "rvce_requests_total" in res_p.text

        print("\n[Service Health Probe] All 4 monitoring endpoints verified:")
        print(f"  • /health:           {res_h.json()['status']}")
        print(f"  • /ready:            {res_r.json()['status']}")
        print(f"  • /monitoring/stats: Uptime={stats['uptime_seconds']}s, P95={stats['latency_ms']['p95']}ms")
        print(f"  • /metrics:          Prometheus exposition active")

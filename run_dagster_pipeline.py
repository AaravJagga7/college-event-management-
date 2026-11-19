"""
================================================================================
Dagster Execution Runner: RVCE 8th Mile Event Intelligence & lakeFS
================================================================================
Executes full Dagster asset materialization in-process and outputs formatted
provenance, model metrics, and lakeFS commit metadata.
================================================================================
"""

import sys
import time
from dagster import materialize
from dagster_orchestration.definitions import all_mlops_assets
from dagster_orchestration.resources import LakeFSResource

def run():
    print("=" * 80)
    print("🚀 RVCE 8th Mile — Dagster Model Orchestration & lakeFS Integration")
    print("=" * 80)
    print("📦 Pipeline: rvce_event_intelligence_dagster_pipeline")
    print("🌊 lakeFS Integration: Active (branch: main)")
    print(f"🔄 Materializing {len(all_mlops_assets)} Software-Defined Assets (SDAs)...")
    print("-" * 80)

    t0 = time.time()
    lakefs_res = LakeFSResource()
    result = materialize(
        all_mlops_assets,
        resources={"lakefs": lakefs_res}
    )
    elapsed = time.time() - t0

    print("-" * 80)
    if result.success:
        print(f"🎉 DAGSTER RUN SUCCEEDED in {elapsed:.2f} seconds!")
        print("=" * 80)
        print("📊 Asset Materialization Summary:")

        for event in result.all_events:
            if event.is_step_success:
                print(f"  ✅ Step: {event.step_key:36} [SUCCESS]")

        # Print lakeFS commit details from latest ledger
        latest = lakefs_res.get_latest_commit()

        print("\n" + "=" * 80)
        print(f"🌊 Verified lakeFS Cryptographic Commit:")
        print("=" * 80)
        print(f"  • Commit ID:     {latest.get('id')}")
        print(f"  • Branch:        {latest.get('branch', 'main')}")
        print(f"  • Message:       {latest.get('message')}")
        print(f"  • Timestamp:     {latest.get('timestamp')}")
        print(f"  • Tracked Objects:")
        for k, v in latest.get("objects", {}).items():
            print(f"      - {k:30} -> {v.get('lakefs_uri')}")
        print("=" * 80)
        return 0
    else:
        print("❌ DAGSTER RUN FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(run())

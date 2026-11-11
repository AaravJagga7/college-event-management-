"""
================================================================================
Legacy MLflow Shim -> Kubeflow Model Registry Migration
================================================================================
MLflow has been switched to Kubeflow Model Registry (KFMR).
Running this file delegates to `train_models_kubeflow.py`.
================================================================================
"""

import sys
import subprocess

if __name__ == "__main__":
    print("ℹ️ Note: MLflow has been migrated to Kubeflow Model Registry.")
    print("🚀 Invoking train_models_kubeflow.py...\n")
    sys.exit(subprocess.call([sys.executable, "train_models_kubeflow.py"]))

import os
import sys

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(PROJECT_DIR, "mlflow.db")
db_uri = f"sqlite:///{db_path}"
artifacts_dir = os.path.join(PROJECT_DIR, "mlruns")
os.makedirs(artifacts_dir, exist_ok=True)

# Required internal environment variables for MLflow server app
os.environ["_MLFLOW_SERVER_FILE_STORE"] = db_uri
os.environ["_MLFLOW_SERVER_REGISTRY_STORE"] = db_uri
os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"] = artifacts_dir
os.environ["MLFLOW_TRACKING_URI"] = db_uri

# Import app AFTER setting the server store environment variables
from mlflow.server import app
from werkzeug.serving import run_simple

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"🚀 MLflow Server active on http://127.0.0.1:{port}")
    print(f"📦 Backend & Registry Store: {db_uri}")
    print(f"📁 Artifact Root:            {artifacts_dir}")
    run_simple("127.0.0.1", port, app, threaded=True)

import os
import sys

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
os.environ["AIRFLOW_HOME"] = os.path.join(PROJECT_DIR, "airflow")
os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"

from airflow.www.app import create_app
from werkzeug.serving import run_simple

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    print(f"🌪️ Starting Airflow Web UI server on http://127.0.0.1:{port}...")
    app = create_app()
    run_simple("127.0.0.1", port, app, threaded=True)


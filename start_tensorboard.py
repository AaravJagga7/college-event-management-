"""
================================================================================
TensorBoard Server Launcher
================================================================================
Runs TensorBoard server on specified port (default 6006) pointing to runs/.
================================================================================
"""

import sys
import os
from tensorboard import program

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 6006
    logdir = sys.argv[2] if len(sys.argv) > 2 else "runs/college_event_tracking"

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", logdir, "--port", str(port), "--bind_all"])
    url = tb.launch()
    print(f"🎉 TensorBoard server running at: {url}")
    print(f"📊 Tracking logdir: {logdir}")
    sys.stdout.flush()

    # Keep alive
    import time
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()

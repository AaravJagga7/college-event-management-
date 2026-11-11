"""
================================================================================
Vercel Serverless Entrypoint for RVCE Event Intelligence Engine
================================================================================
Exposes the FastAPI application as a serverless handler for Vercel deployment.
================================================================================
"""

import sys
import os

# Ensure the root directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

# Vercel looks for 'app' or WSGI/ASGI handler in this module
# FastAPI 'app' is an ASGI callable


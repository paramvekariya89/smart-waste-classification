"""
Vercel Serverless Entrypoint
Imports the Flask application from app.py
"""
import sys
import os

# Add parent directory to path so app can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app

# Vercel WSGI entry point
app = app

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import sys
import os

# Add parent directory to path to import from main.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import app

# Wrap FastAPI app with Mangum for serverless
handler = Mangum(app)

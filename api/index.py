import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import just the app, don't execute anything
try:
    from main import app
    from mangum import Mangum

    # Wrap with Mangum for serverless
    handler = Mangum(app, lifespan="off")
except Exception as e:
    print(f"Error loading app: {e}")
    raise

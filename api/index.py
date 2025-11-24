from mangum import Mangum
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import app WITHOUT lifespan
from main import create_app

# Create app instance
app = create_app()

# Wrap with Mangum (disable lifespan)
handler = Mangum(app, lifespan="off")

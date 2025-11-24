from mangum import Mangum

# Import app factory
from main import create_app

# Create app instance
app = create_app()

# Wrap with Mangum for serverless
handler = Mangum(app, lifespan="off")
